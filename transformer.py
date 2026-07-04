"""
transformer.py — Encoder, Decoder, and top-level Transformer.

- Encoder: token embed + positional encode + N x EncoderBlock stack.
  Supports Post-LN (EncoderBlock) and Pre-LN hybrid (EncoderBlockPreLN)
  blocks. Applies GPT-2 style 1/sqrt(2*num_layers) residual scaling at
  init for Post-LN stability and deep-stack variance control. Collects
  per-layer attention weights for visualization.
- Decoder: token embed + positional encode + N x DecoderBlock stack.
- Transformer: ties Encoder + Decoder + final LM head (Linear -> vocab logits).

Usage:
    model = Transformer(config)

    # Training (teacher forcing)
    logits, enc_attn, dec_attn, cross_attn = model(
        src_tokens, tgt_input, src_mask, tgt_mask
    )
    loss = F.cross_entropy(logits.view(-1, V), tgt_output.view(-1))

    # Inference (autoregressive — see generate.py)
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
from config import TransformerConfig
from embeddings import TokenEmbedding, LearnedPositionalEncoding
from layers import EncoderBlock, EncoderBlockPreLN, DecoderBlock


class Encoder(nn.Module):
    """
    Transformer encoder: token embed + positional encode + N x EncoderBlock stack.

    Shapes:
        src_tokens: (batch, src_seq_len)
        output:     (batch, src_seq_len, d_model)

    Owns the embedding layer, positional encoding, embedding dropout, and the
    block stack. Applies GPT-2 style 1/sqrt(2*num_layers) scaling to residual
    output projections (W_o in MHA, linear2 in FFN) at init — this is important
    for Post-LN stability (reduces gradient amplification through the LN
    backward) and deep Pre-LN variance control.

    Both EncoderBlock (Post-LN) and EncoderBlockPreLN (Pre-LN hybrid) self-
    normalize their output, so `final_norm=False` by default. Set it True only
    if you swap in pure Pre-LN blocks (x + Sublayer(LN(x)), no per-block output
    norm), whose last-layer output is un-normalized.

    ----

    CONCEPT: GPT-2 scaled init (1/sqrt(2*num_layers))
    ──────────────────────────────────────────────────

    Without scaling, each block's residual output has variance ~O(1), and
    summing N blocks gives output variance ~N. This amplifies the signal
    through the stack, making early training unstable in Post-LN (gradients
    explode through LayerNorm backward) and causing variance drift in deep
    Pre-LN. The fix: scale the residual OUTPUT projections (attention W_o
    and FFN linear2) by 1/sqrt(2*num_layers) so that after N blocks, the
    total residual variance stays ~O(1).

    This is a GPT-2 trick (Radford et al., 2019) — it's NOT in the original
    Transformer paper, but virtually every modern implementation uses it.
    Post-LN absolutely needs it; Pre-LN benefits from it at depth.

    ----

    CONCEPT: final_norm — when you need it
    ──────────────────────────────────────

    Our EncoderBlock (Post-LN) and EncoderBlockPreLN (hybrid) both normalize
    their own output (norm2 and norm3 respectively), so the stack output is
    already normalized — no final LayerNorm needed.

    Pure Pre-LN blocks (GPT-2 style, x + Sublayer(LN(x))) do NOT normalize
    their output — variance grows ~N and the decoder receives un-normalized
    features. For those blocks, set `final_norm=True` to add a LayerNorm after
    the block loop. This is what GPT-2 calls `ln_f`.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        block_cls: type[nn.Module] = EncoderBlock,
        final_norm: bool = False,
        scaled_init: bool = True,
    ):
        super().__init__()

        # -----------------------------------------------------------
        # Embeddings — map token IDs to d_model vectors, add
        # positional information, apply dropout.
        # -----------------------------------------------------------
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = LearnedPositionalEncoding(config)
        self.embed_dropout = nn.Dropout(config.dropout)

        # -----------------------------------------------------------
        # Block stack — N identical encoder blocks.
        # Uses kwarg-only block_cls so you can pass EncoderBlockPreLN
        # (or any future variant) without changing code.
        # -----------------------------------------------------------
        n_layers = config.num_encoder_layers
        self.layers = nn.ModuleList(block_cls(config) for _ in range(n_layers))
        self.num_layers = n_layers

        # -----------------------------------------------------------
        # Final LayerNorm — only needed for PURE Pre-LN blocks whose
        # output is un-normalized (x + Sublayer(LN(x))). Both our
        # block variants self-normalize: Post-LN via norm2 after the
        # FFN residual, Pre-LN hybrid via norm3 at block output.
        # Defaults to None — set True for pure Pre-LN blocks.
        # -----------------------------------------------------------
        self.final_norm = nn.LayerNorm(config.d_model) if final_norm else None

        # -----------------------------------------------------------
        # Scaled init — shrink residual output projections so the
        # total variance across N blocks stays ~O(1). Applied once
        # at construction time; no runtime overhead.
        # -----------------------------------------------------------
        if scaled_init:
            self._apply_scaled_init()

    # ---------------------------------------------------------------
    # Scaled initialization (applied once, at construction)
    # ---------------------------------------------------------------

    @torch.no_grad()
    def _apply_scaled_init(self) -> None:
        """
        Scale residual output projections by 1/sqrt(2 * num_layers).

        Targets:
          - Attention output projection (W_o in MultiHeadAttention)
          - FFN second linear layer (linear2 in FeedForward)

        These are the two places where a sublayer's result enters the
        residual stream. Scaling them keeps the summed residual variance
        ~O(1) after N blocks.

        If a block doesn't have the expected attribute names (e.g., custom
        MHA or FFN), we warn rather than silently no-op so you can add
        your own attribute names to the probe lists.
        """
        scale = 1.0 / math.sqrt(2 * self.num_layers)

        # Attribute names to probe. Our MHA uses W_o; our FFN uses linear2.
        # Additional names cover common variants (nn.TransformerEncoderLayer,
        # HuggingFace, etc.).
        attn_out_names = ("W_o", "out_proj", "wo", "o_proj", "output_proj")
        ffn_out_names = ("linear2", "w2", "fc2", "down_proj", "proj_out")

        missed = False
        for blk in self.layers:
            if not self._scale_first_match(
                getattr(blk, "self_attn", None), attn_out_names, scale
            ):
                missed = True
            if not self._scale_first_match(
                getattr(blk, "ffn", None), ffn_out_names, scale
            ):
                missed = True

        if missed:
            warnings.warn(
                "Encoder._apply_scaled_init: could not find a residual "
                "output projection to scale on at least one block. Add "
                "your attribute names to attn_out_names/ffn_out_names, "
                "or pass scaled_init=False.",
                stacklevel=2,
            )

    @staticmethod
    def _scale_first_match(
        module: nn.Module | None, names: tuple[str, ...], scale: float
    ) -> bool:
        """Try each attribute name in `names`; scale the first match found."""
        if module is None:
            return False
        for name in names:
            proj = getattr(module, name, None)
            if isinstance(proj, nn.Linear):
                proj.weight.mul_(scale)
                return True
        return False

    # ---------------------------------------------------------------
    # FORWARD PASS
    # ---------------------------------------------------------------

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        *,
        need_weights: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Args:
            src_tokens:    (batch, src_seq_len) — int tensor of token IDs.
            src_mask:      (batch, 1, 1, src_seq_len) or None — padding mask
                           applied to self-attention keys in every block.
                           Pass None for unpadded sequences.
            need_weights:  if False, skip collecting per-layer attention maps.
                           Saves memory during inference when you don't need
                           heatmaps. (The attention computation itself still
                           produces weights; this flag only controls whether
                           we RETAIN them.)

        Returns:
            x:          (batch, src_seq_len, d_model) — final hidden states.
            attn_maps:  list of (batch, num_heads, src_seq_len, src_seq_len)
                        — one per layer — or None if need_weights=False.
        """
        # ---- Step 1: Token embedding + positional encoding ----
        # TokenEmbedding applies sqrt(d_model) scaling per the paper.
        # Positional encoding is broadcast-added element-wise.
        x = self.token_embedding(src_tokens)      # (B, S, d_model)
        x = x + self.position_encoding(src_tokens)  # (B, S, d_model)
        x = self.embed_dropout(x)                  # (B, S, d_model)

        # ---- Step 2: Pass through each encoder block ----
        attn_maps: list[torch.Tensor] | None = [] if need_weights else None

        for layer in self.layers:
            x, attn = layer(x, src_mask)
            if attn_maps is not None:
                attn_maps.append(attn)

        # ---- Step 3: Final LayerNorm (only if enabled) ----
        # Both our block variants (Post-LN and Pre-LN hybrid) self-
        # normalize their output, so this is usually skipped. It's here
        # for pure Pre-LN blocks or custom variants that don't.
        if self.final_norm is not None:
            x = self.final_norm(x)

        return x, attn_maps

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, "
            f"d_model={self.token_embedding.embedding.embedding_dim}, "
            f"final_norm={self.final_norm is not None}"
        )


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
