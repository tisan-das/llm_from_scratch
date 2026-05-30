"""
layers.py — Feed-forward network, encoder block, and decoder block.

- FeedForward: two linear layers with ReLU activation (as per paper), applied
  position-wise (same weights for every position in the sequence).
- EncoderBlock: self-attention → add & norm → feed-forward → add & norm.
- DecoderBlock: masked self-attention → add & norm → cross-attention → add & norm →
  feed-forward → add & norm.

Layer norm is applied AFTER the residual add (Post-LN), following the original paper:
    LayerNorm(x + Sublayer(x))

Usage:
    ff = FeedForward(config)
    enc_block = EncoderBlock(config)
    dec_block = DecoderBlock(config)

    # Encoder block forward
    out, self_attn_weights = enc_block(x, src_mask)

    # Decoder block forward
    out, self_attn_w, cross_attn_w = dec_block(x, encoder_output, src_mask, tgt_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TransformerConfig
from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Shapes:
        Input:  (batch, seq_len, d_model)
        Hidden: (batch, seq_len, d_ff)
        Output: (batch, seq_len, d_model)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EncoderBlock(nn.Module):
    """
    A single encoder layer.

    Sublayers:
        1. Multi-head self-attention (bidirectional — no causal mask by default)
        2. LayerNorm(x + Dropout(attn_out))
        3. Feed-forward
        4. LayerNorm(x + Dropout(ff_out))

    Returns attention weights for visualization.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:        (batch, src_seq_len, d_model)
            src_mask: (batch, 1, 1, src_seq_len) padding mask for keys, or None

        Returns:
            x:              (batch, src_seq_len, d_model)
            attn_weights:   (batch, num_heads, src_seq_len, src_seq_len)
        """
        raise NotImplementedError


class DecoderBlock(nn.Module):
    """
    A single decoder layer.

    Sublayers:
        1. Masked multi-head self-attention (causal — can't see future positions)
        2. LayerNorm(x + Dropout(attn_out_1))
        3. Multi-head cross-attention to encoder output (Q from decoder, K,V from encoder)
        4. LayerNorm(x + Dropout(attn_out_2))
        5. Feed-forward
        6. LayerNorm(x + Dropout(ff_out))

    Returns both self-attention and cross-attention weights for visualization.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               (batch, tgt_seq_len, d_model)      decoder hidden states
            encoder_output:  (batch, src_seq_len, d_model)      encoder output (K,V for cross-attn)
            src_mask:        (batch, 1, 1, src_seq_len)         padding mask for cross-attn keys
            tgt_mask:        (1, 1, tgt_seq_len, tgt_seq_len)   causal mask for self-attn

        Returns:
            x:                  (batch, tgt_seq_len, d_model)
            self_attn_weights:  (batch, num_heads, tgt_seq_len, tgt_seq_len)
            cross_attn_weights: (batch, num_heads, tgt_seq_len, src_seq_len)
        """
        raise NotImplementedError
