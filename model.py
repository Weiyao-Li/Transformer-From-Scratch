import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# convey positional info to the model
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (batch=1, seq_len, d_model)

        # not to be trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # added

    def forward(self, x):
        # keep dim: originally cancel the dim, but keep it
        # from (1, 6, 512) to (1, 6, 1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (b, l, d_model) -> (b, l, d_ff) -> (b, l, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.attentionScores = None
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (b, h, l, dk) @ (b, h, dk, l) -> (b, h, l, l)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attn = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attn = dropout(attn)

        # (b, h, l, l) @ (b, h, l, dk) -> (b, h, l, dk)
        return (attn @ value), attn

    def forward(self, q, k, v, mask):
        # x (b, l, d) @ (d, d) = (b, l, d)
        query = self.w_q(q)  # (b, l, d)
        key = self.w_k(k)  # (b, l, d)
        value = self.w_v(v)  # (b, l, d)

        # (b, l, d) -> (b, l, h, dk) -> (b, h, l, dk)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # (b, h, l, dk)
        x, self.attentionScores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (b, h, l, dk) -> (b, l, h, dk) -> (b, l, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.h * self.d_k)

        # (b, l, d_model) @ (d_model, d_model) -> (b, l, d_model)
        return self.w_o(x)



