"""
transformer.py — Encoder, Decoder, and top-level Transformer.

- Encoder: token embed + positional encode + N × EncoderBlock stack.
- Decoder: token embed + positional encode + N × DecoderBlock stack.
- Transformer: ties Encoder + Decoder + final LM head (Linear → vocab logits).

Usage:
    model = Transformer(config)

    # Training (teacher forcing)
    logits, enc_attn, dec_attn, cross_attn = model(
        src_tokens, tgt_input, src_mask, tgt_mask
    )
    loss = F.cross_entropy(logits.view(-1, V), tgt_output.view(-1))

    # Inference (autoregressive — see generate.py)
"""

import torch
import torch.nn as nn
from config import TransformerConfig
from embeddings import TokenEmbedding, LearnedPositionalEncoding
from layers import EncoderBlock, DecoderBlock


class Encoder(nn.Module):
    """
    Transformer encoder: embed → add positional encoding → stack of encoder blocks.

    Shapes:
        src_tokens: (batch, src_seq_len)
        output:     (batch, src_seq_len, d_model)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self, src_tokens: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns:
            output:               (batch, src_seq_len, d_model)
            all_self_attn_weights: list of (batch, num_heads, src_seq_len, src_seq_len)
                                   one per layer, for visualization
        """
        raise NotImplementedError


class Decoder(nn.Module):
    """
    Transformer decoder: embed → add positional encoding → stack of decoder blocks.

    Shapes:
        tgt_tokens:      (batch, tgt_seq_len)
        encoder_output:  (batch, src_seq_len, d_model)
        output:          (batch, tgt_seq_len, d_model)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns:
            output:                   (batch, tgt_seq_len, d_model)
            all_self_attn_weights:    list of (batch, num_heads, tgt_seq_len, tgt_seq_len)
            all_cross_attn_weights:   list of (batch, num_heads, tgt_seq_len, src_seq_len)
        """
        raise NotImplementedError


class Transformer(nn.Module):
    """
    The full encoder-decoder Transformer.

    encoder(src) → encoder_output
    decoder(tgt, encoder_output) → hidden states
    lm_head(hidden) → logits over vocabulary

    Shapes:
        src_tokens:  (batch, src_seq_len)
        tgt_tokens:  (batch, tgt_seq_len)
        logits:      (batch, tgt_seq_len, vocab_size)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        Returns:
            logits:           (batch, tgt_seq_len, vocab_size)
            enc_self_attn:    list of encoder self-attention weights per layer
            dec_self_attn:    list of decoder self-attention weights per layer
            cross_attn:       list of decoder cross-attention weights per layer
        """
        raise NotImplementedError
