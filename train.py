import os
import warnings
from pathlib import Path

import torch
import torch.nn
import torchmetrics

from datasets import load_dataset, config, tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_weights_file_path, get_config
from dataset import BilingualDataset, causal_mask
from model import build_transformer


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for the target(decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate the output: (batch_size, tgt_seq_len, d_model)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Project the hidden state of the latest generated token to vocabulary logits
        prob = model.project(out[:, -1])

        # select the token with the max prob (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)

        # Append the selected vocabulary token ID to the end of the current decoder sequence (extend seq_len by 1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                                  dim=1)

        if next_word == eos_idx:
            break

        # Remove the batch dimension (batch_size=1) to return a 1D sequence of token IDs
    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer,
                   num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    # Build the path where the tokenizer file will be stored or loaded from
    # Example: "tokenizer_{}.json".format("en") -> "tokenizer_en.json"
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    # If the tokenizer file does not exist, we need to train a new one
    if not Path.exists(tokenizer_path):

        # Create a WordLevel tokenizer
        # WordLevel means: one word corresponds to one token
        # unk_token='[UNK]' is used for words that are not in the vocabulary
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))

        # Define how the text is split before tokenization
        # Whitespace means splitting by spaces: "I love NLP" -> ["I", "love", "NLP"]
        tokenizer.pre_tokenizer = Whitespace()

        # Create a trainer that will build the vocabulary
        # special_tokens: tokens that must always be included in the vocab
        # min_frequency=2: words appearing less than 2 times are discarded and mapped to [UNK]
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        # Train the tokenizer on all sentences in the dataset for the given language
        # This scans the dataset, counts word frequencies, and builds the vocabulary
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        # Save the trained tokenizer to disk so we can reuse it later
        tokenizer.save(str(tokenizer_path))

    else:
        # If the tokenizer file already exists, just load it from disk
        # This avoids retraining and ensures consistency across runs
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Return the tokenizer (either newly trained or loaded)
    return tokenizer


def get_ds(config):
    # Load parallel translation dataset (source ↔ target)
    ds_raw = load_dataset(
        'opus_books',
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split='train'
    )

    # Build or load tokenizers:
    # tokenizer_src → used by the encoder
    # tokenizer_tgt → used by the decoder and labels
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split dataset into train (90%) and validation (10%)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Wrap raw datasets with BilingualDataset:
    # add special tokens, padding, and attention masks
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )

    # Compute max token length in raw data (for choosing seq_len)
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        # Encoder side length
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]
        ).ids

        # Decoder side length
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config['lang_tgt']]
        ).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max source length: {max_len_src}")
    print(f"Max target length: {max_len_tgt}")

    # DataLoader wraps the dataset and creates mini-batches for training
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Validation DataLoader:
    # batch_size=1 makes it easier to inspect individual translations
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True
    )

    # Return everything needed for training and inference:
    # - train_dataloader: for model training
    # - val_dataloader: for evaluation
    # - tokenizer_src: encoder tokenizer
    # - tokenizer_tgt: decoder tokenizer
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])

    return model


def train_model(config):
    # Pick device: use GPU (cuda) if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device {device}")

    # Create folder for saving checkpoints (weights/)
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Build dataloaders + tokenizers (train/val split happens inside get_ds)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Build model using vocab sizes (src vocab for encoder, tgt vocab for decoder)
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    # TensorBoard logger
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer (Adam is standard for Transformer)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Resume training bookkeeping
    initial_epoch = 0
    global_step = 0

    # If preload is set, load checkpoint and continue training
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']

    # Cross-entropy over vocab:
    # - ignore PAD positions so padding doesn't contribute to loss
    # - label_smoothing helps regularize / stabilize training
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    # Main training loop over epochs
    for epoch in range(initial_epoch, config['num_epochs']):

        # tqdm just shows a progress bar
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            model.train()

            # Batch tensors (B = batch size)
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)

            # Attention masks:
            # encoder_mask: blocks PAD tokens in encoder self-attention
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)

            # decoder_mask: blocks PAD tokens + blocks future tokens (causal)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Forward pass through Transformer:
            # 1) encode source sequence -> encoder_output (memory)
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)

            # 2) decode with teacher forcing using decoder_input,
            #    attending to encoder_output
            decoder_output = model.decode(
                encoder_output,
                encoder_mask,
                decoder_input,
                decoder_mask
            )  # (B, seq_len, d_model)

            proj_output = model.project(decoder_output)  # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device)  # (B, seq_len)

            # CrossEntropyLoss expects:
            #   input  : (N, C)  -> N samples, C classes
            #   target : (N,)    -> one class index per sample
            #
            # Here, each token position is treated as one training sample.
            # We have:
            #   B sentences
            #   seq_len tokens per sentence
            # So total samples N = B * seq_len
            #
            # Flatten (batch_dim, time_dim) into a single dimension:
            # (B, seq_len, vocab_size) -> (B * seq_len, vocab_size)
            # (B, seq_len)            -> (B * seq_len)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagate
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename
        )
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
