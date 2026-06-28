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

        FFN(x) = ReLU(x @ W₁ + b₁) @ W₂ + b₂

    "Position-wise" means the same two linear transformations are applied
    independently and identically to every position in the sequence.  There is
    no mixing between positions — that is the attention sub-layer's job.

    The first linear layer expands from d_model to d_ff (a 4× expansion in the
    paper and in our default config: 128 → 512).  The ReLU then creates non-linear
    features in this higher-dimensional space.  The second linear layer projects
    back to d_model so the output can re-enter the residual stream.

    Shapes:
        Input:  (batch, seq_len, d_model)
        Hidden: (batch, seq_len, d_ff)
        Output: (batch, seq_len, d_model)

    Parameter count (d_model=128, d_ff=512):
        W₁: 128x512=65,536   b₁: 512   →   66,048
        W₂: 512x128=65,536   b₂: 128   →   65,664
        Total: 131,712  (~2x the multi-head attention sitting next to it)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # ---------------------------------------------------------------
        # Linear layer 1 — expand from d_model to d_ff.
        #
        # The paper uses a 4× expansion (d_model=512 → d_ff=2048).  Our
        # tiny config keeps the same ratio: 128 → 512.  The expansion is
        # not a theoretical requirement — d_ff = d_model would still work
        # (the ReLU still prevents the two linear layers from collapsing
        # into one affine map).  But expanding into a higher-dimensional
        # space gives the ReLU much richer features to work with, similar
        # to the kernel trick in SVMs: a hyperplane in a high-dimensional
        # space can separate patterns that are not linearly separable in
        # the original space.
        #
        # The 4× ratio itself is empirical — the paper doesn't justify it.
        # It makes the FFN roughly twice the size of the attention sub-layer
        # (which has 4 × d_model² params ≈ 4×128² = 65,536, vs. FFN's
        # 2 × d_model × d_ff ≈ 2×128×512 = 131,072).  Most Transformer
        # variants preserve this ~2:1 capacity ratio between FFN and
        # attention, though the exact multiplier varies (2×, 2.7×, 8×).
        #
        # bias=True is already nn.Linear's default; we state it explicitly
        # because the paper's formula calls the bias terms out by name:
        #   FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
        #
        # Weight shape: (d_ff, d_model) — maps each position's d_model
        # vector to a higher-dimensional d_ff vector.
        # ---------------------------------------------------------------
        self.linear1 = nn.Linear(config.d_model, config.d_ff, bias=True)

        # ---------------------------------------------------------------
        # Linear layer 2 — project back from d_ff to d_model.
        #
        # After the ReLU creates a rich d_ff-dimensional hidden
        # representation, this layer compresses it back to d_model so the
        # output can be added to the residual stream.  The residual
        # connection expects the same shape as the sub-layer input, so we
        # must land exactly at d_model.
        #
        # Weight shape: (d_model, d_ff)
        # ---------------------------------------------------------------
        self.linear2 = nn.Linear(config.d_ff, config.d_model, bias=True)

        # ---------------------------------------------------------------
        # Note on initialization — nn.Linear's defaults are correct here.
        #
        # nn.Linear applies Kaiming uniform init (a=√5) to weights and a
        # small uniform init to biases.  Kaiming init is designed for layers
        # followed by ReLU — it preserves activation variance through the
        # non-linearity, preventing the vanishing/exploding signal problem.
        #
        # Contrast with embeddings.py, where we manually set
        # std = 1/√d_model to satisfy the paper's √d_model scaling recipe.
        # Here, no custom init is needed — the default is the right one.
        #
        # For linear2 (which is NOT followed by ReLU), Kaiming init is
        # slightly suboptimal in theory (Xavier/Glorot would be the ideal
        # for a layer without non-linearity), but the difference is
        # negligible in practice and PyTorch uses Kaiming uniformly for
        # all Linear layers.
        # ---------------------------------------------------------------

        # ---------------------------------------------------------------
        # NOTE on dropout — this module deliberately stores none.
        #
        # Two *different* dropouts can sit around an FFN; keeping them
        # straight matters:
        #
        #   (a) Sub-layer-output dropout — applied to the FFN's d_model
        #       output, just before it is added to the residual stream.
        #       The paper (§5.4) specifies exactly this: dropout is applied
        #       "to the output of each sub-layer, before it is added to the
        #       sub-layer input and normalized."  That is the block's
        #       Add & Norm responsibility, so it lives in the
        #       EncoderBlock / DecoderBlock that wraps this call — NOT here.
        #       The block does:  ff_out = self.ffn(x)
        #                        x = self.norm(x + self.dropout(ff_out))
        #
        #   (b) Inner hidden dropout — applied to the d_ff activations
        #       between ReLU and linear2.  This is a SEPARATE regularizer on
        #       a DIFFERENT tensor at a DIFFERENT point in the computation;
        #       it is not the same dropout as (a), so having both is not
        #       redundant.  The paper's literal text only mandates (a),
        #       so we omit (b) to stay paper-faithful and keep this module
        #       focused on a single responsibility: the linear + activation
        #       transformation.
        #
        # Many canonical implementations (the Annotated Transformer,
        # tensor2tensor, PyTorch's own TransformerEncoderLayer) DO include
        # (b):  linear2(dropout(relu(linear1(x)))).  If you want it, store
        #   self.dropout = nn.Dropout(config.dropout)
        # here and apply it after the ReLU in forward().  This is an
        # optional design choice, not a correctness fix — the model trains
        # fine either way.
        # ---------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position-wise feed-forward transformation.

        Every position (token) in the sequence is processed independently
        with the SAME two linear layers and ReLU.  Only the last axis
        (d_model → d_ff → d_model) changes; the (batch, seq_len) axes ride
        along untouched — that is what "position-wise" means in practice.

        Args:
            x: (batch, seq_len, d_model) — hidden states coming from the
               previous Add & Norm (which wrapped the self-attention or
               cross-attention output).

        Returns:
            (batch, seq_len, d_model) — transformed hidden states, same
            shape as input so they can be added to the residual stream.

        Shape trace (d_model=128, d_ff=512):
            (batch, seq_len, 128)  →  linear1  →  (batch, seq_len, 512)
            (batch, seq_len, 512)  →  ReLU     →  (batch, seq_len, 512)
            (batch, seq_len, 512)  →  linear2  →  (batch, seq_len, 128)
        """
        # ---- Step 1: Expand to hidden dimension ----
        # Each position's 128-dim vector is projected to a 512-dim hidden
        # space.  The weight matrix W₁ has shape (512, 128) — each of the
        # 512 output features is a learned linear combination of all 128
        # input features for that position.
        #
        #   x @ W₁^T + b₁  =  (B, S, 128) @ (512, 128)^T + (512,)
        #                    =  (B, S, 512)
        #
        # nn.Linear applies the transform to the LAST axis only, so the
        # (batch, seq_len) leading dimensions pass through unchanged — this
        # is the tensor-level mechanism that makes the FFN position-wise.
        x = self.linear1(x)

        # ---- Step 2: Apply ReLU non-linearity ----
        # ReLU(x) = max(0, x) — element-wise, zeroing out negative values.
        #
        # Why a non-linearity is necessary:
        #   Without ReLU, the two linear layers collapse into one:
        #     linear2(linear1(x))
        #     = (x @ W₁^T + b₁) @ W₂^T + b₂
        #     = x @ (W₁^T @ W₂^T) + (b₁ @ W₂^T + b₂)
        #     = x @ W_equiv^T + b_equiv          ← just a single affine map!
        #   A stack of purely linear layers can always be replaced by a
        #   single equivalent layer — no extra expressive power.  The
        #   non-linearity between them is what makes depth meaningful.
        #
        # Why ReLU specifically:
        #   The paper (§3.3) uses ReLU.  This was the standard activation
        #   in 2017.  GELU (Hendrycks & Gimpel 2016) existed but wasn't
        #   widely adopted until BERT (2018) and GPT-2 (2019) used it.
        #   The Transformer paper using ReLU was the norm, not a deliberate
        #   choice against better alternatives.
        #
        # Modern variants — if you want to experiment:
        #   - GELU (F.gelu):  Smoother around zero, better gradients.
        #     TRUE drop-in — swap `F.relu(x)` → `F.gelu(x)`.  Done.
        #     Used by BERT, GPT-2, and most modern Transformers.
        #
        #   - SwiGLU (Silu-gated):  (Swish(x@W_gate) * (x@W_up)) @ W_down.
        #     NOT a drop-in — it adds a THIRD projection matrix (a gating
        #     branch), so you must change both __init__ and forward.
        #     d_ff is also typically cut to ~2/3 (e.g. 8d/3 instead of 4d)
        #     to keep the total parameter count comparable.  Used by
        #     LLaMA, PaLM, and most modern large-scale LLMs.
        x = F.relu(x)

        # ---- Step 3: Project back to d_model ----
        # Compress from the 512-dim hidden space back to 128-dim so the
        # output can enter the residual connection and the next encoder
        # or decoder block.
        #
        #   x @ W₂^T + b₂  =  (B, S, 512) @ (128, 512)^T + (128,)
        #                    =  (B, S, 128)
        x = self.linear2(x)

        return x


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
