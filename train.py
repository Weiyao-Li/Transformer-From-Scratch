from pathlib import Path

import torch
import torch.nn

from datasets import load_dataset, config
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import random_split

from dataset import BilingualDataset


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
