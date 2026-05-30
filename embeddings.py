"""
embeddings.py — Token and positional embeddings.

- TokenEmbedding: maps token IDs → dense vectors, scaled by √d_model (as per paper).
- LearnedPositionalEncoding: learned vectors per position (0, 1, 2, ...), added to token
  embeddings element-wise.

Usage:
    tok_emb = TokenEmbedding(config)
    pos_emb = LearnedPositionalEncoding(config)

    x = tok_emb(token_ids)          # (B, S) -> (B, S, d_model)
    x = x + pos_emb(token_ids)      # broadcast add positional info
"""

import torch
import torch.nn as nn
from config import TransformerConfig


class TokenEmbedding(nn.Module):
    """
    Learned token embedding with √d_model scaling.

    Shapes:
        Input:  (batch, seq_len)           token IDs
        Output: (batch, seq_len, d_model)  dense vectors
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding added element-wise to token embeddings.

    Positions range from 0 to max_seq_len-1. The same positional vector is broadcast
    across the batch dimension.

    Shapes:
        Input:  (batch, seq_len)           (only seq_len is used, actual IDs ignored)
        Output: (batch, seq_len, d_model)  positional vectors ready to add
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
