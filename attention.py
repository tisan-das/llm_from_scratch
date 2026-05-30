"""
attention.py — Scaled dot-product attention and multi-head attention.

- scaled_dot_product_attention(): the core Q·K^T/√d_k → softmax → weighted V formula.
- MultiHeadAttention: splits d_model into num_heads, applies attention in parallel,
  then concatenates and projects with W_o.  Works for both self-attention (Q==K==V)
  and cross-attention (Q from decoder, K,V from encoder).

Mask convention: 0 = attend, -inf = mask out.  Masks are added to attention scores
before softmax, so masked positions get exp(-inf) = 0 weight.

Usage:
    # Self-attention
    mha = MultiHeadAttention(config)
    out, attn_weights = mha(query=x, key=x, value=x, mask=causal_mask)

    # Cross-attention
    out, attn_weights = mha(query=decoder_hidden, key=encoder_output, value=encoder_output)

    # Standalone function (rarely used directly)
    attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TransformerConfig


# ---------------------------------------------------------------------------
# Core attention function
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

    Args:
        Q: Queries  — (batch, num_heads, seq_len_q, d_k)
        K: Keys     — (batch, num_heads, seq_len_k, d_k)
        V: Values   — (batch, num_heads, seq_len_v, d_k)  [seq_len_k == seq_len_v]
        mask:       — (batch, 1, seq_len_q, seq_len_k) or broadcastable.
                      0=attend, -inf=mask out.  None means no masking.
        dropout:    — Optional nn.Dropout applied to attention weights.

    Returns:
        output:         — (batch, num_heads, seq_len_q, d_k)  weighted sum of values
        attn_weights:   — (batch, num_heads, seq_len_q, seq_len_k)  softmax scores
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Multi-head attention module
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.

    Projects input to Q, K, V (each of shape d_model) via linear layers,
    splits into num_heads heads of size d_k, applies scaled dot-product attention,
    concatenates heads back, and projects with W_o.

    Shapes:
        query:  (batch, seq_len_q, d_model)
        key:    (batch, seq_len_k, d_model)
        value:  (batch, seq_len_v, d_model)  [seq_len_k == seq_len_v]
        mask:   (batch, 1, seq_len_q, seq_len_k) or None
        output: (batch, seq_len_q, d_model)

    For self-attention: query == key == value (same tensor).
    For cross-attention: query from decoder, key & value from encoder.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:       (batch, seq_len_q, d_model)
            attn_weights: (batch, num_heads, seq_len_q, seq_len_k)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mask construction utilities
# ---------------------------------------------------------------------------

def create_padding_mask(
    seq: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    """
    Create a mask that hides <pad> tokens in the key/value sequence.

    Returns mask of shape (batch, 1, 1, seq_len) where positions with pad_token_id
    are -inf, others are 0.  Added to attention scores before softmax.

    Usage:
        src_mask = create_padding_mask(src_tokens, config.pad_token_id)
        # Pass as mask to encoder self-attention and decoder cross-attention
    """
    raise NotImplementedError


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create an upper-triangular mask so position i can only attend to positions ≤ i.

    Returns mask of shape (1, 1, seq_len, seq_len) where upper triangle = -inf,
    lower triangle (including diagonal) = 0.

    Usage:
        tgt_mask = create_causal_mask(tgt_len)
        # Pass as mask to decoder self-attention
    """
    raise NotImplementedError
