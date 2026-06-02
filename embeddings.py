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
    Learned token embedding with √d_model scaling (paper §3.4).

    Two-part scale recipe:
      1. Initialize weights with std = 1/√d_model, i.e. N(0, 1/d_model)
         → row L2 norm ≈ 1
      2. Multiply output by √d_model
         → row L2 norm ≈ √d_model

    Both halves are required. With PyTorch's default N(0, 1) init, rows
    already have norm ≈ √d_model, so step (2) overshoots to norm ≈ d_model
    — a factor of √d_model too large — drowning out the sinusoidal
    positional encodings (norm ≈ √(d_model/2)).

    Shapes:
        Input:  (batch, seq_len)           token IDs
        Output: (batch, seq_len, d_model)  dense vectors
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.scale = config.d_model ** 0.5
        # Part 1 of the paper's scale recipe: init N(0, 1/√d).
        # Without this, default N(0,1) init + ×√d overshoots to norm ≈ d
        # and token embeddings drown out positional encodings ~12×.
        nn.init.normal_(self.embedding.weight, mean=0.0,
                        std=1.0 / self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding added element-wise to token embeddings.

    Positions range from 0 to max_seq_len-1. The same positional vector is
    broadcast across the batch dimension — only seq_len matters, not the
    actual token IDs passed in.

    Shapes:
        Input:  (batch, seq_len)           (only seq_len is used, actual IDs ignored)
        Output: (batch, seq_len, d_model)  positional vectors ready to add
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds "
                f"max_seq_len ({self.position_embedding.num_embeddings}). "
                f"Increase TransformerConfig.max_seq_len or truncate your input."
            )
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
        return self.position_embedding(positions).unsqueeze(0)  # (1, seq_len, d_model)
