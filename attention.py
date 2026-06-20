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

    This is the core operation behind every Transformer.  For each query position,
    we compute how much it should "attend to" every key position, producing a set
    of attention weights (one scalar per query–key pair).  Those weights then
    select a weighted combination of value vectors.

    The scaling factor 1/√d_k keeps the dot-product variance near 1.0 regardless
    of head dimension — without it, large d_k would push the softmax toward
    one-hot (saturation), killing gradient flow.

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
    # ---------------------------------------------------------------
    # Step 1: Compute raw attention scores.
    #
    # For each (batch, head, query_pos, key_pos) we take the dot product
    # of the query vector at query_pos with the key vector at key_pos.
    # The result is a (seq_len_q × seq_len_k) score matrix per head.
    #
    # K.transpose(-2, -1) swaps the last two dimensions so that we get
    #   Q @ K^T = (..., seq_len_q, d_k) @ (..., d_k, seq_len_k)
    #            = (..., seq_len_q, seq_len_k)
    #
    # Dividing by √d_k is the "scaled" part — it prevents the dot-product
    # variance from growing with d_k, which would otherwise push the
    # softmax into near-one-hot territory and kill gradient flow (§3.2.1).
    #
    # Shapes:
    #   Q:      (batch, num_heads, seq_len_q, d_k)
    #   K^T:    (batch, num_heads, d_k, seq_len_k)
    #   scores: (batch, num_heads, seq_len_q, seq_len_k)
    # ---------------------------------------------------------------
    d_k = Q.size(-1)  # head dimension — same for Q, K, V
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # ---------------------------------------------------------------
    # Step 2: Apply the attention mask (if provided).
    #
    # Mask convention (additive):
    #    0     → score is unchanged → position is attended to.
    #   -inf   → score becomes -inf → softmax output = exp(-inf) = 0.
    #
    # The mask broadcasts from (batch, 1, seq_len_q, seq_len_k) to
    # (batch, num_heads, seq_len_q, seq_len_k) — the same mask is
    # shared across all heads, which is the standard Transformer convention.
    #
    # Two common mask types used in this codebase:
    #   create_padding_mask() → hides <pad> tokens in the key sequence.
    #   create_causal_mask()  → prevents attending to future positions.
    # ---------------------------------------------------------------
    # if mask is not None:
    #     scores = scores + mask
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores + mask

    # ---------------------------------------------------------------
    # Step 3: Softmax over the key dimension (dim=-1).
    #
    # This converts raw similarity scores into a proper probability
    # distribution per query position: every row sums to 1.0.
    #
    # Masked positions (score = -inf) produce exp(-inf) = 0, so they
    # contribute nothing to the output and do not steal probability
    # mass from unmasked positions.
    #
    # Edge case — all-masked row:
    #   If every key position is masked (score = -inf everywhere),
    #   softmax evaluates 0/0 and produces NaN.  This never happens
    #   in practice: causal masks always leave the diagonal unmasked,
    #   and padding masks are only applied when at least one real token
    #   exists.  We deliberately do NOT guard against this case because
    #   silently returning uniform weights would hide genuine bugs.
    # ---------------------------------------------------------------
    attn_weights = F.softmax(scores, dim=-1)

    # ---------------------------------------------------------------
    # Step 4: Apply dropout to attention weights (optional).
    #
    # Dropout is applied element-wise to the attention weights after
    # softmax, randomly zeroing some attention connections during
    # training (§5.4).  This forces the model to learn redundant
    # representations rather than relying on a single attention path.
    #
    # In eval mode (dropout.eval()), nn.Dropout is a no-op — all
    # weights pass through unchanged.  This matches test A8's
    # expectation that training produces stochastic outputs while
    # eval produces deterministic ones.
    # ---------------------------------------------------------------
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # ---------------------------------------------------------------
    # Step 5: Weighted sum of value vectors.
    #
    # Each query position's output is the attention-weighted average
    # of all value vectors:
    #
    #   output[q] = Σ_k attn_weights[q, k] · V[k]
    #
    # Masked-out positions (attn_weight = 0) contribute nothing.
    #
    # Shapes:
    #   attn_weights: (batch, num_heads, seq_len_q, seq_len_k)
    #   V:            (batch, num_heads, seq_len_v, d_k)   [seq_len_k == seq_len_v]
    #   output:       (batch, num_heads, seq_len_q, d_k)
    # ---------------------------------------------------------------
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ---------------------------------------------------------------------------
# Multi-head attention module
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.

    Projects input to Q, K, V (each of shape d_model) via linear layers,
    splits into num_heads heads of size d_k, applies scaled dot-product attention,
    concatenates heads back, and projects with W_o.

    The "multi-head" trick: instead of one attention mechanism over the full
    d_model=128 space, run num_heads=4 independent attention mechanisms over
    d_k=32-dimensional subspaces.  Each head learns to attend to different
    patterns (local syntax, long-range dependencies, positional cues, etc.).
    The heads are concatenated and mixed by W_o, which is the only place
    information crosses between heads.

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
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_k  # d_model // num_heads (validated by config property)

        # ---------------------------------------------------------------
        # Projection matrices (paper §3.2.2)
        #
        # W_q, W_k, W_v project the input from d_model to d_model.
        # The paper uses bias=True; we follow this for faithfulness.
        #
        # Weight shape: (d_model, d_model) — each token's hidden vector
        # is linearly transformed into a query, key, or value vector.
        # ---------------------------------------------------------------
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---------------------------------------------------------------
        # Output projection (paper §3.2.2)
        #
        # After concatenating the head outputs, W_o mixes information
        # across heads.  Without it, each head's output would live in
        # an isolated 32-dimensional subspace and the heads could not
        # interact.  W_o is a fully-connected layer that combines all
        # head outputs into a coherent d_model-dimensional representation.
        # ---------------------------------------------------------------
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---------------------------------------------------------------
        # Dropout on attention weights (paper §5.4)
        #
        # Applied after softmax, before the weighted sum of values.
        # Randomly zeros some attention connections during training,
        # forcing the model to learn redundant attention patterns.
        # Passed through to scaled_dot_product_attention which handles
        # train/eval mode correctly.
        # ---------------------------------------------------------------
        self.dropout = nn.Dropout(config.dropout)

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
        batch_size = query.size(0)

        # ---------------------------------------------------------------
        # Step 1: Linear projections from d_model to d_model.
        #
        # Each token's hidden vector is mapped to three different roles:
        #   query  — "what am I looking for?"
        #   key    — "here's what I contain"
        #   value  — "here's the information to pass along"
        #
        # Self-attention: query, key, value all come from the same input.
        # Cross-attention: query from decoder, key & value from encoder.
        #
        # Shapes:
        #   query:  (B, S_q, d_model)  ->  Q: (B, S_q, d_model)
        #   key:    (B, S_k, d_model)  ->  K: (B, S_k, d_model)
        #   value:  (B, S_v, d_model)  ->  V: (B, S_v, d_model)
        # ---------------------------------------------------------------
        Q = self.W_q(query)  # (B, S_q, d_model)
        K = self.W_k(key)    # (B, S_k, d_model)
        V = self.W_v(value)  # (B, S_v, d_model)

        # ---------------------------------------------------------------
        # Step 2: Split each projection into multiple heads.
        #
        # The d_model-dimensional vector is carved into num_heads slices
        # of size d_k.  Think of it like this (for d_model=128, H=4, d_k=32):
        #
        #   [dim0..dim31 | dim32..dim63 | dim64..dim95 | dim96..dim127]
        #      head 0        head 1         head 2         head 3
        #
        # Each head gets its own 32-dimensional subspace and will learn
        # its own attention pattern independently of the other heads.
        #
        # The .view() + .transpose() dance:
        #   1. view(B, S, H, d_k)  — split the last dimension into (H, d_k)
        #   2. transpose(1, 2)     — move heads to dim 1 for batch matmul
        #
        # Shape pipeline:
        #   (B, S, d_model)  --view-->  (B, S, H, d_k)  --transpose-->  (B, H, S, d_k)
        # ---------------------------------------------------------------
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Now: Q, K, V are each (B, num_heads, S, d_k)

        # ---------------------------------------------------------------
        # Step 3: Scaled dot-product attention (the mathematical core).
        #
        # This is where the actual attention computation happens:
        #   scores = Q @ K^T / sqrt(d_k)
        #   weights = softmax(scores + mask)
        #   output = dropout(weights) @ V
        #
        # The function handles broadcasting the mask across heads,
        # applying dropout, and returning both the output and the
        # attention weights (needed for visualization).
        #
        # Shapes:
        #   Q, K, V: (B, H, S, d_k)
        #   attn_out:    (B, H, S_q, d_k)
        #   attn_weights: (B, H, S_q, S_k)
        # ---------------------------------------------------------------
        attn_out, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # ---------------------------------------------------------------
        # Step 4: Merge heads back into a single d_model vector.
        #
        # Reverse of step 2:
        #   1. transpose(1, 2)  — move heads back to dim 2
        #   2. reshape(B, S_q, d_model)  — concatenate the head outputs
        #
        # We use .reshape() rather than .view() because the tensor is
        # non-contiguous after transpose.  .reshape() handles this by
        # making a copy when needed; .view() would crash with a
        # "view size is not compatible" error.
        #
        # Shape pipeline:
        #   (B, H, S_q, d_k)  --transpose-->  (B, S_q, H, d_k)  --reshape-->  (B, S_q, d_model)
        # ---------------------------------------------------------------
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.d_model)

        # ---------------------------------------------------------------
        # Step 5: Final output projection.
        #
        # W_o mixes information across heads.  Each head's output is a
        # weighted sum of values from its own 32-dimensional subspace.
        # W_o allows head 0 to learn that it should amplify what head 2
        # found and suppress what head 1 found — the heads coordinate.
        #
        # Shape: (B, S_q, d_model) -> (B, S_q, d_model)
        # ---------------------------------------------------------------
        output = self.W_o(attn_out)

        return output, attn_weights


# ---------------------------------------------------------------------------
# Mask construction utilities
# ---------------------------------------------------------------------------

def create_padding_mask(
    seq: torch.Tensor, pad_token_id: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a mask that hides <pad> tokens in the key/value sequence.

    Returns mask of shape (batch, 1, 1, seq_len) where positions with pad_token_id
    are a large negative value, others are 0.  Added to attention scores before softmax.

    Usage:
        src_mask = create_padding_mask(src_tokens, config.pad_token_id)
        # Pass as mask to encoder self-attention and decoder cross-attention
    """
    # (batch, seq_len) -> boolean mask where True = pad token
    is_pad = (seq == pad_token_id)

    # Create mask: 0 where valid token, large negative where pad
    # finfo(dtype).min stays finite in fp16/bf16 (avoids NaN on all-pad rows)
    # Shape: (batch, seq_len), dtype matches attention scores, same device as seq
    mask = torch.zeros_like(seq, dtype=dtype)
    mask[is_pad] = torch.finfo(dtype).min

    # Reshape to (batch, 1, 1, seq_len) for broadcasting with attention scores
    # which have shape (batch, num_heads, seq_len_q, seq_len_k)
    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

    return mask

def create_causal_mask(
    seq_len: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create an upper-triangular mask so position i can only attend to positions ≤ i.

    Returns mask of shape (1, 1, seq_len, seq_len) where upper triangle = -inf,
    lower triangle (including diagonal) = 0.

    Args:
        seq_len: number of positions in the sequence.
        device:  torch device (default CPU). Pass the model's device to avoid
                 CPU/GPU mismatch when the mask is added to attention scores.
        dtype:   dtype of the mask (default float32). Should match the attention
                 score dtype in mixed-precision training (fp16, bf16).

    Usage:
        tgt_mask = create_causal_mask(tgt_len, device=x.device)
        # Pass as mask to decoder self-attention
    """
    # Start with all -inf, then zero out lower triangle + diagonal.
    # torch.triu with diagonal=1 keeps elements strictly above the main
    # diagonal (j > i) and sets everything on or below to 0.
    #
    # We use -inf rather than finfo(dtype).min because every row in a
    # causal mask has at least one unmasked position (the diagonal), so
    # all-blocked rows cannot occur. finfo.min is only needed when masks
    # are combined additively (padding + causal), which this codebase
    # does not do — they go into separate attention calls.
    mask = torch.full((seq_len, seq_len), float("-inf"),
                      device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)          # upper tri = -inf, lower+diag = 0
    mask = mask.unsqueeze(0).unsqueeze(0)        # (1, 1, seq_len, seq_len)
    return mask
