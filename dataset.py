from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

'''
https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-it?row=3
'''


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        """
        Initialize the bilingual dataset.

        Args:
            ds: Raw dataset (e.g., HuggingFace dataset) containing translation pairs.
            tokenizer_src: Tokenizer for the source language.
            tokenizer_tgt: Tokenizer for the target language.
            src_lang: Source language key in the dataset (e.g., "en").
            tgt_lang: Target language key in the dataset (e.g., "it").
            seq_len: Fixed sequence length for encoder/decoder inputs.
        """
        super().__init__()

        # Raw dataset
        self.ds = ds

        # Tokenizers for source and target languages
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        # Language keys
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Maximum sequence length
        self.seq_len = seq_len

        # Special tokens as tensors (used for building input sequences)
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")],
            dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")],
            dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")],
            dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        # Get one sample from the raw dataset
        src_target_pair = self.ds[index]

        # Extract source and target sentences by language key
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize source sentence and convert tokens to token IDs
        # "I love you."
        # → tokens: ["I", "love", "you", "."]
        # → ids:    [45, 123, 87, 9]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Number of PAD tokens for encoder input (reserve 2 slots for [SOS] and [EOS])
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # Number of PAD tokens for decoder input (reserve 1 slot for [SOS])
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long!')

        # add SOS and EOS to the source text
        # [SOS] + tokens + [EOS] + [PAD] * n
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)

            "decoder_input": decoder_input,  # (seq_len)

            # Encoder padding mask:
            # 1 means "this position is a real token", 0 means "this is PAD"
            # Shape: (1, 1, seq_len), will be broadcast to (batch, heads, query_len, key_len)
            # Used to prevent the encoder from attending to PAD tokens.
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),

            # Decoder mask = padding mask AND causal mask:
            # - Padding mask: prevents attending to PAD tokens
            # - Causal mask: prevents attending to future tokens (look-ahead)
            #
            # (decoder_input != PAD) → (1, seq_len)
            # causal_mask(seq_len)  → (1, seq_len, seq_len)
            # After broadcasting & AND:
            # final shape → (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
                            .unsqueeze(0)
                            .int()
                            & causal_mask(decoder_input.size(0)),

            "label": label,  # (seq_len)

            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    # Upper triangular matrix:
    # 1 above the diagonal, 0 on and below the diagonal
    # Shape: (1, size, size)
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)

    # Convert to boolean mask:
    # True  → position is allowed (can attend)
    # False → position is blocked (future positions)
    #
    # So each position i can only attend to positions <= i
    return mask == 0
