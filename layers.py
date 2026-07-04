"""
layers.py — Feed-forward network, encoder blocks (Post-LN & Pre-LN), and decoder block.

INITIALIZATION: the blocks themselves do NOT apply special weight init beyond
nn.Linear / nn.LayerNorm defaults. The Encoder (in transformer.py) should apply
a GPT-2 style 1/sqrt(2*num_layers) scaling to residual output projections (W_o
and FFN linear2) for Post-LN stability and deep Pre-LN variance control.

ATTENTION DROPOUT: the blocks only drop sublayer OUTPUTS. The paper also drops
attention PROBABILITIES; that dropout lives inside MultiHeadAttention, since the
block never sees the pre-output attention weights.

- FeedForward: two linear layers with ReLU activation (as per paper), applied
  position-wise (same weights for every position in the sequence).
- EncoderBlock:        Post-LN variant (paper faithful) —
      self-attention → add & norm → feed-forward → add & norm.
      Formula:  LayerNorm(x + Sublayer(x))
- EncoderBlockPreLN:   Pre-LN sublayers + per-block output norm (hybrid) —
      norm → self-attention → add → norm → feed-forward → add → norm.
      Formula:  norm3( x + Sublayer(LayerNorm(x)) )
- DecoderBlock: masked self-attention → add & norm → cross-attention → add & norm →
  feed-forward → add & norm.

Post-LN (paper) vs Pre-LN (modern):
    Post-LN:  LayerNorm( x + Sublayer(x) )    ← paper, needs warmup
    Pre-LN:   x + Sublayer( LayerNorm(x) )    ← GPT/LLaMA, warmup-free

Usage:
    ff = FeedForward(config)
    enc_block = EncoderBlock(config)           # Post-LN — paper faithful
    enc_block_pre = EncoderBlockPreLN(config)   # Pre-LN  — modern default
    dec_block = DecoderBlock(config)

    # Both encoder block variants share the same forward signature:
    out, self_attn_weights = enc_block(x, src_mask)
    out, self_attn_weights = enc_block_pre(x, src_mask)

    # Decoder block forward
    out, self_attn_w, cross_attn_w = dec_block(x, encoder_output, src_mask, tgt_mask)
"""

from __future__ import annotations

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
    A single encoder layer — the core building block repeated N times in the
    encoder stack (paper: N=6; our default: 3).

        x   (B, S, d_model)
        │
        ├───────┐                 split: copy x → residual (skip)
        │       │
   ┌────┴────┐  │
   │Self-Attn│  │
   └────┬────┘  │
        │       │
   ┌────┴────┐  │
   │ Dropout │  │
   └────┬────┘  │
        │       │
       (⊕)◄─────┘                 Add:  x + Dropout(Attn(x))
        │
   ┌────┴────┐
   │LayerNorm│  ← norm1   (Post-LN: norm sits AFTER the add, on the trunk)
   └────┬────┘
        │
        ├───────┐                 split again
        │       │
   ┌────┴────┐  │
   │  Feed-  │  │
   │ Forward │  │
   └────┬────┘  │
        │       │
   ┌────┴────┐  │
   │ Dropout │  │
   └────┬────┘  │
        │       │
       (⊕)◄─────┘                 Add:  x + Dropout(FFN(x))
        │
   ┌────┴────┐
   │LayerNorm│  ← norm2
   └────┬────┘
        │
        ▼
     output   (B, S, d_model)

    ----

    CONCEPT: The Add & Norm pattern (Post-LN)
    ─────────────────────────────────────────

    Every sublayer is wrapped as:

        LayerNorm( x + Dropout( Sublayer(x) ) )
                     │              │
                     │              └── the sublayer's transformation
                     │                  (attention or FFN)
                     │
                     └── the residual connection (the "Add")
                         The sublayer's input is ADDED BACK to its output.

    This pattern appears everywhere — encoder self-attention, encoder FFN,
    decoder self-attention, decoder cross-attention, decoder FFN.  Every
    single sublayer in the entire paper follows this identical recipe.

    The pattern is "Post-LN" because LayerNorm is applied AFTER (post)
    the residual add.  The alternative "Pre-LN" (used by GPT, LLaMA, etc.)
    swaps the order:  x + Sublayer( LayerNorm(x) ).  We follow the paper's
    Post-LN for faithfulness; the tradeoffs are discussed below.

    ----

    CONCEPT: Why the residual connection (the "Add")
    ─────────────────────────────────────────────────

    Without a residual connection, the block would be:

        x → Sublayer(x) → LayerNorm(...)

    During backprop, the gradient must flow through BOTH the sublayer
    AND LayerNorm.  If the sublayer produces tiny gradients (e.g., ReLU
    saturation, vanishing attention weights), the signal dies and earlier
    layers stop learning.  This is the classic vanishing gradient problem
    that plagued deep networks before residual connections (He et al. 2016).

    With a residual connection:

        y = x + Sublayer(x)     →    ∂L/∂x = ∂L/∂y · (1 + ∂Sublayer/∂x)
                                               │   └── sublayer gradient path
                                               └── IDENTITY gradient path (= 1)

    The gradient has TWO paths back to x:
      (a) Through the sublayer:  ∂L/∂y · ∂Sublayer/∂x
      (b) Through the identity:  ∂L/∂y · 1   ← always 1, never dies!

    Even if the sublayer gradient vanishes, the identity path still
    delivers a healthy gradient.  This is why Transformers (and ResNets)
    can stack 100+ layers while plain networks fail at ~20.

    A second, equally important property: "identity learning."  If a
    sublayer is harmful at initialization or during early training, the
    network can learn to make Sublayer(x) ≈ 0, turning the block into
    y ≈ x (a near-identity).  The block can effectively DISABLE itself
    without hurting the network.

    [REVIEW] Idealization caveat: this is often summarized as "depth is
    penalty-free," but that's an over-simplification. Near-identity
    learning removes the WORST failure mode (harmful layers tanking the
    net), but extra depth still carries real optimization cost (more to
    train, compounding numerical effects) and generalization cost
    (capacity you may not need). Read it as "depth is much SAFER to add,"
    not literally free.

    ----

    CONCEPT: The residual stream
    ───────────────────────────

    Because every block preserves the (batch, seq_len, d_model) shape and
    every sublayer adds its output back to the same tensor, the hidden
    state can be thought of as a SHARED COMMUNICATION CHANNEL called the
    "residual stream" (Elhage et al., 2021).

    Think of it as a whiteboard passed from block to block:

      Block 1:  [attention writes: "these tokens are related like this"]
                [FFN writes:     "based on that, here's what this position means"]
                → passes whiteboard to block 2

      Block 2:  [attention writes: "with that context, these tokens relate like this"]
                [FFN writes:     "here's a more refined meaning per position"]
                → passes whiteboard to block 3

    Each block INCREMENTS the hidden state (adds to it), it never rewrites
    it from scratch.  The positional encoding added before block 1 survives
    all the way through because it rides the identity path — the attention
    and FFN sublayers can only ADD to it, never erase it.

    [REVIEW] Loose-justification caveat: weight tying (sharing the input
    embedding with the output/LM-head projection; Press & Wolf 2017) works
    fundamentally because you REUSE the token↔vector map — the LM head is
    the embedding matrix transposed. The "residual stream stays in the same
    d_model space" framing is a nice intuition for WHY that reuse is
    coherent, but it's not the mechanism. Don't over-read it.

    ----

    CONCEPT: LayerNorm — per-token, not per-batch
    ──────────────────────────────────────────────

    nn.LayerNorm(d_model) normalizes each token's hidden vector across the
    d_model features. The EXACT operation PyTorch computes:

        LayerNorm(h) = γ ⊙ (h - μ) / √(σ² + ε)  +  β

        where  μ  = mean of h over the d_model axis
               σ² = BIASED variance of h over the d_model axis (÷N, not ÷(N-1))
               γ  = learned scale  vector of shape (d_model,)  — init to 1
               β  = learned shift  vector of shape (d_model,)  — init to 0
               ε  = tiny constant added INSIDE the sqrt for numerical
                    stability (nn.LayerNorm default: 1e-5)

    [REVIEW] Two from-scratch gotchas — the earlier draft wrote
    "(h - μ) / (σ + ε)", which does NOT match nn.LayerNorm:
        1. ε goes INSIDE the sqrt: √(σ² + ε), NOT (σ + ε). PyTorch does
           (h - μ) / √(Var + ε).
        2. Var is BIASED (divides by N). Use torch.var(..., unbiased=False),
           or your Karpathy-style reimplementation will diverge from the
           reference nn.LayerNorm outputs by a small but real amount.
    (The textbook notation (h-μ)/σ is the math idealization that drops ε
    entirely; fine for intuition, wrong for bit-matching the module.)

    Modern note: many recent Pre-LN stacks (e.g. LLaMA) use RMSNorm rather
    than LayerNorm — same "normalize per token" idea, but skips the mean
    subtraction and the β shift, dividing by RMS(h) = √(Σh²/N) instead of
    √(σ²+ε). RMSNorm's gain initializes to 1.0 exactly like LayerNorm's γ;
    what differs is dropping mean-centering (and the β shift), making it
    cheaper and slightly more stable in FP16.

    For a hidden tensor of shape (batch, seq_len, d_model):
      - BatchNorm  would compute μ, σ over (batch, seq_len) — across all tokens
        in the batch.  This breaks when batch sizes change or when generating
        one token at a time (inference).  BatchNorm also couples the statistics
        of different sequences, which is undesirable in NLP.
      - LayerNorm  computes μ, σ independently for EVERY INDIVIDUAL TOKEN.
        Token (b=3, s=7) is normalized using only its own 128 features, not
        influenced by token (b=3, s=8) or any other token in the batch.

    This per-token independence is the key property that makes LayerNorm
    the standard choice for Transformers:
      1. Works identically for seq_len=1 (autoregressive generation) or
         seq_len=100 (packed training batches).
      2. No cross-token or cross-batch coupling — each token is its own
         independent unit of normalization.
      3. γ and β are learned SEPARATELY for each LayerNorm instance.  The
         norms at block 1 learn different scale/shift values than the norms
         at block 6 because the hidden state statistics differ by depth.

    ----

    CONCEPT: Post-LN vs. Pre-LN — the grand tradeoff
    ─────────────────────────────────────────────────

    Post-LN (paper, what we implement):
        y = LayerNorm( x + Sublayer(x) )

    Pre-LN (GPT, LLaMA, most modern Transformers):
        y = x + Sublayer( LayerNorm(x) )

    Why the paper chose Post-LN:
      - It follows the ResNet pattern more closely: the residual branch
        includes the normalization at the end.
      - The output of each block is always normalized, so the next block
        always receives well-behaved activations.

    Why modern models switched to Pre-LN:
      - Post-LN requires a learning-rate WARMUP phase (the paper's Figure 2
        shows 4,000 warmup steps).  Without warmup, Post-LN gradients explode
        in early steps because the LayerNorm in the backward path amplifies
        gradients from deep layers back to shallow layers.
      - Pre-LN puts LayerNorm on the SUBLAYER BRANCH, not on the residual
        path.  The residual path (the identity) is never normalized, so
        gradients flow cleanly from the loss directly to any layer without
        passing through a LayerNorm.  No warmup needed.
      - Formula:  ∂L/∂x = ∂L/∂y + (∂L/∂y through Sublayer(LN(x)))
        The identity gradient path ∂L/∂y is UNTOUCHED by LayerNorm.

    Pre-LN does require a FINAL LayerNorm after the last block (since the
    last block's output is not normalized in Pre-LN).  Our code follows
    Post-LN for paper faithfulness; switching to Pre-LN is a one-line
    change per sublayer (move LN from after residual-add to before sublayer).

    ----

    CONCEPT: Self-attention in the encoder — bidirectional
    ──────────────────────────────────────────────────────

    The encoder self-attention is BIDIRECTIONAL: every position can attend
    to every other position in the source sequence.  We pass Q=x, K=x, V=x
    (all from the same input) with no causal mask.  The only mask that can
    be passed is src_mask — a padding mask that hides <pad> tokens in the
    KEY sequence (see create_padding_mask in attention.py).

    Why bidirectional?  The encoder reads the ENTIRE source sequence at
    once.  When encoding "The cat sat on the mat", position 6 ("mat")
    SHOULD be able to attend to position 1 ("cat") to form a coherent
    representation.  There's no notion of "future" or "past" in the
    encoder — the entire sequence is available simultaneously.

    This contrasts with the DECODER's self-attention, which MUST use a
    causal mask (upper-triangular -inf) so position i can only attend to
    positions ≤ i.  The causal mask prevents the decoder from "cheating"
    during autoregressive training by looking at future tokens it's
    supposed to predict.

    The encoder does NOT receive a causal mask.  The src_mask, if provided,
    is only for padding — it tells attention heads to ignore <pad> tokens
    that were added to make all sequences in a batch the same length.

    ----

    CONCEPT: Why two sublayers in this order (attention → FFN)
    ──────────────────────────────────────────────────────────

    The block has exactly two sublayers: attention first, then FFN.

    Attention (cross-position mixing):
      "For each query position, which other positions are relevant right
       now, and what should I extract from them?"

    FFN (per-position transformation):
      "Given the pooled information at this position (what attention just
       gathered), what features should I compute from it?"

    The order is "MIX then TRANSFORM":
      1. Attention gathers context from across the sequence.
      2. FFN processes each position's enriched context independently.

    Could you swap them?  Yes — FFN then attention is also a valid
    Transformer.  You'd attend over FFN-transformed features instead of
    raw hidden states.  Both orders work; "attention first" is the
    convention from the paper, not a theoretical requirement.

    Can you run them in PARALLEL?  Yes — "parallel attention + FFN"
    (used by some efficient Transformers) computes both sublayers
    simultaneously from the same input and sums both outputs:
        x + attention(x) + ffn(x), then LayerNorm.
    This is a future experiment listed in the SPEC roadmap.

    ----

    CONCEPT: Dropout placement — why BEFORE the residual add
    ───────────────────────────────────────────────────────

    The paper (§5.4) is explicit:

      "We apply dropout to the output of each sub-layer, BEFORE it is
       added to the sub-layer input and normalized."

    So the recipe is:
        LayerNorm( x + Dropout( Sublayer(x) ) )

    NOT:  LayerNorm( x + Sublayer( Dropout(x) ) )   — wrong: dropout on INPUT
    NOT:  Dropout( LayerNorm( x + Sublayer(x) ) )    — wrong: dropout after NORM

    Why before the residual add?  Dropout randomly zeroes connections,
    and those zeros ride the residual connection back to x.  If a particular
    attention head is dropped, its contribution to the residual stream is
    zero — the next sublayer must work with a partial update.  This forces
    the network to build REDUNDANT representations: no single attention head
    or FFN feature is allowed to be indispensable, because it might be
    dropped at any training step.

    At EVAL time (model.eval()), Dropout is a no-op — all connections are
    active, so the model uses its full ensemble of learned patterns.

    Note: this is SUBLAYER-OUTPUT dropout only. The paper also drops
    attention PROBABILITIES (applied after softmax, before the weighted sum);
    that dropout lives inside
    MultiHeadAttention, since the block never sees the pre-output attention
    weights. Both dropouts use the same p (paper §5.4).

    ----

    CONCEPT: Shape invariance — the block is a "pure transformation"
    ────────────────────────────────────────────────────────────────

    Input:  (batch, src_seq_len, d_model)
    Output: (batch, src_seq_len, d_model)

    The shape NEVER changes inside the block.  This is a deliberate design
    decision that makes the architecture modular:

      - Stack N blocks without adjusting anything — just loop over them.
      - Remove or add blocks without touching the rest of the model.
      - The hidden dimensionality d_model is the common currency that every
        component speaks.  The embedding layer maps tokens → d_model.  Every
        block operates on d_model vectors.  The LM head maps d_model → vocab.

    This is the opposite of CNNs, where pooling layers halve spatial
    resolution while doubling channels.  Transformers have no progressive
    dimension change — every layer sees the same "resolution" and "width."
    This homogeneity is both a strength (simplicity, modularity) and a
    weakness (quadratic attention cost at every layer, no hierarchical
    feature extraction).

    ----

    PARAMETER INVENTORY (d_model=128, d_ff=512, num_heads=4):

      Multi-head self-attention:   4 × (128×128 + 128)  =  66,048  (W_q,k,v,o)
      Feed-forward:                    2×128×512 + 512 + 128  = 131,712
      LayerNorm × 2:               2 × (128 + 128)      =     512  (γ + β each)
      ─────────────────────────────────────────────────────────
      Total per block:                                    ~198,272

    With 3 encoder blocks: ~595K encoder params (out of ~1.2M total for
    the full 3+3 encoder-decoder model).

    Vs. the paper's 6-block, d_model=512, d_ff=2048 config:
      ~7M per block × 6 = ~42M parameters (encoder half of a ~65M model).
    """

    # ---------------------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------------------

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # -----------------------------------------------------------
        # (1) Multi-head self-attention — the "mix" sublayer.
        #
        # This is a FULL MultiHeadAttention module.  At forward time,
        # we pass the same tensor for Q, K, and V so every position
        # attends to every other position using its own projection
        # matrices (W_q, W_k, W_v).
        #
        # Self-attention vs. cross-attention:
        #   - SELF:  Q, K, V all come from the same source (x).
        #            The module doesn't know or care — it just executes
        #            softmax(Q @ K^T / √d_k) @ V.  The "self" vs "cross"
        #            distinction is in HOW it's called, not in the module.
        #   - CROSS: Q comes from the decoder, K,V come from the encoder.
        #            Same MultiHeadAttention module, different arguments.
        #
        # The same MultiHeadAttention class is reused for:
        #   - Encoder self-attention (this line)
        #   - Decoder self-attention (in DecoderBlock)
        #   - Decoder cross-attention (in DecoderBlock)
        #
        # There's exactly ONE MultiHeadAttention class in the entire
        # codebase (attention.py).  That's good design — the attention
        # mechanism is identical regardless of what role it plays; only
        # the masks and input sources differ.
        #
        # No causal mask is passed here.  The encoder sees the full
        # source sequence bidirectionally.  If src_seq_len=10, every
        # query position i can attend to every key position j (0..9).
        # The src_mask, if provided, only masks out <pad> tokens.
        # -----------------------------------------------------------
        self.self_attn = MultiHeadAttention(config)

        # -----------------------------------------------------------
        # (2) First LayerNorm — normalization after the attention
        #     residual add (Post-LN).
        #
        # nn.LayerNorm(d_model) normalizes across the LAST dimension
        # of the input tensor.  For an input of shape (B, S, d_model):
        #   - Computes μ and σ² (biased variance) over dim=-1 (the d_model features).
        #   - Each of the B×S tokens is normalized INDEPENDENTLY.
        #   - Applies learned γ and β (each of shape (d_model,)).
        #
        # This is norm1 — it normalizes the output of the FIRST
        # sublayer (self-attention + residual).  Different from norm2,
        # which normalizes the output of the SECOND sublayer (FFN +
        # residual).  They learn different γ/β because the attention
        # output and FFN output have different statistical properties.
        #
        # Why elementwise_affine=True (the default)?
        #   γ and β let the network undo normalization when needed.
        #   Starting from γ=1, β=0, the initial normalization is
        #   plain zero-mean unit-variance.  Over training, the network
        #   learns to re-scale and re-shift features that benefit from
        #   it.  Without learned γ/β (elementwise_affine=False), the
        #   network cannot escape the zero-mean unit-variance constraint,
        #   which limits expressivity.
        # -----------------------------------------------------------
        self.norm1 = nn.LayerNorm(config.d_model)

        # -----------------------------------------------------------
        # (3) Position-wise feed-forward network — the "transform"
        #     sublayer.
        #
        # This is the FeedForward class defined above.  It expands from
        # d_model=128 to d_ff=512, applies ReLU, and projects back to
        # d_model=128.  The FeedForward class handles all the details
        # of linear layers, initialization, and activation.
        #
        # Key property: position-wise — same weights applied to every
        # position independently.  No cross-position mixing.  That's the
        # attention sublayer's job.
        #
        # The FeedForward class deliberately stores NO dropout inside
        # itself (see its docstring for the detailed justification).
        # Dropout is applied HERE, in the block, to the FFN output
        # before the residual add — consistent with the paper's recipe.
        # -----------------------------------------------------------
        self.ffn = FeedForward(config)

        # -----------------------------------------------------------
        # (4) Second LayerNorm — normalization after the FFN
        #     residual add (Post-LN).
        #
        # Same nn.LayerNorm(d_model) as norm1, but a SEPARATE instance
        # with its own γ and β.  norm2 specializes in normalizing
        # (x + FFN(x)) while norm1 specializes in normalizing
        # (x + Attention(x)).  Different sublayers, different statistics,
        # different learned normalization parameters.
        #
        # Using two separate LayerNorms (rather than sharing one) is the
        # standard Transformer implementation.  The paper's Figure 1
        # explicitly shows each sublayer with its own "Add & Norm" block.
        # -----------------------------------------------------------
        self.norm2 = nn.LayerNorm(config.d_model)

        # -----------------------------------------------------------
        # (5) Dropout — applied to BOTH sublayer outputs.
        #
        # One nn.Dropout instance is used TWICE in forward — once after
        # self-attention, once after the FFN.  Reusing the same module
        # is harmless because nn.Dropout is stateless (it only depends on
        # self.training, which doesn't change mid-forward).
        #
        # During training (model.train()): each forward call randomly
        # zeroes out a fraction p=config.dropout (default 0.1) of
        # elements in the sublayer output tensor, scaled by 1/(1-p)
        # so the expected value remains unchanged (inverted dropout).
        #
        # During eval (model.eval()): dropout is a no-op — all elements
        # pass through unchanged.  This is standard inverted dropout
        # (Srivastava et al., 2014) as implemented by nn.Dropout.
        #
        # The SAME dropout probability p=0.1 is used everywhere in the
        # paper (§5.4): embedding dropout, attention-weight dropout,
        # sublayer-output dropout.  We follow this uniform-p convention.
        # -----------------------------------------------------------
        self.dropout = nn.Dropout(config.dropout)

    def extra_repr(self) -> str:
        """Surface d_model and variant in print(model) for quick inspection."""
        return f"d_model={self.norm1.normalized_shape[0]}, variant=post_ln"

    # ---------------------------------------------------------------
    # FORWARD PASS — step-by-step with shape tracing
    # ---------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one encoder block.

        Args:
            x:        (batch, src_seq_len, d_model)
                      Hidden states from the previous encoder block
                      (or from the embedding layer for block 1).

            src_mask: (batch, 1, 1, src_seq_len) or None.
                      Padding mask applied to the KEY dimension of
                      self-attention.  Positions with <pad> tokens get
                      masked out (attention weight → 0).

                      Pass None when all sequences in the batch have
                      the same length (no padding).  For the reversal
                      task with fixed-length sequences, this is typically
                      None-the mask is unnecessary because we always
                      use the full sequence.

        Returns:
            x:              (batch, src_seq_len, d_model)
                            Refined hidden states — same shape as input
                            so blocks can be stacked arbitrarily.

            attn_weights:   (batch, num_heads, src_seq_len, src_seq_len)
                            Softmax attention weights from this block's
                            self-attention.  Collected by the Encoder
                            for visualization.
        """

        # -----------------------------------------------------------
        # SUB-LAYER 1: Multi-head self-attention
        # -----------------------------------------------------------
        #
        #   Q, K, V = x (same tensor, self-attention mode)
        #   attn_out  = weighted sum of V, where weights come from
        #               softmax(Q @ K^T / √d_k + mask) — see attention.py
        #   attn_weights = the softmax attention scores (for visualization)
        #
        # Shape trace:
        #   input x:                (B, S, d_model)     [e.g. (64, 10, 128)]
        #   Q, K, V after W_q,k,v:  (B, S, d_model)     each
        #   split into H=4 heads:   (B, 4, S, 32)       each
        #   scores = Q @ K^T:       (B, 4, S, S)        [e.g. (64, 4, 10, 10)]
        #   weights = softmax:      (B, 4, S, S)        same shape
        #   attn_out per head:      (B, 4, S, 32)       weighted V
        #   merge heads:            (B, S, d_model)     [e.g. (64, 10, 128)]
        #
        # Bidirectional: NO causal mask.  Position 5 can attend to
        # positions 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 — all of them.
        # Only the src_mask (padding) restricts attention, if provided.
        #
        # For the reversal task, encoder tokens attend to their
        # neighbours in the source sequence (not anti-diagonal — that
        # pattern is in the decoder's cross-attention).  Encoder
        # self-attention typically shows local n-gram patterns and
        # broad positional awareness.
        # -----------------------------------------------------------
        attn_out, attn_weights = self.self_attn(
            query=x, key=x, value=x, mask=src_mask
        )
        # attn_out:     (B, S, d_model) — attention output, same shape as x
        # attn_weights: (B, H, S, S)    — for visualization

        # -----------------------------------------------------------
        # ADD & NORM #1: residual connection + layer normalization
        # -----------------------------------------------------------
        #
        #   x = LayerNorm( x + Dropout(attn_out) )
        #
        # Breaking this down:
        #
        #   (a) Dropout(attn_out):
        #       Randomly zeros ~10% of elements in attn_out during
        #       training (no-op during eval).  Forces the network to
        #       not rely on any single attention feature.
        #
        #   (b) x + Dropout(attn_out):
        #       The RESIDUAL ADD.  The original input x is added to
        #       the (dropped) attention output.  Shape is unchanged:
        #       (B, S, d_model).
        #
        #       This creates the "residual stream": x carries forward
        #       all information from previous layers, and attn_out
        #       INCREMENTS it with cross-position mixing information.
        #
        #   (c) LayerNorm(...):
        #       Normalizes each token's d_model vector to zero mean
        #       and unit variance, then applies learned γ and β.
        #       After the residual add, the hidden state's statistics
        #       may have drifted — LayerNorm snaps them back to a
        #       well-behaved range for the next sublayer.
        #
        # Why not Dropout(x + attn_out)?  Because the paper (§5.4)
        # explicitly places dropout "before it is added" — only the
        # sublayer output gets dropped, not the identity path.  The
        # identity path (x) always flows through unmasked, keeping a
        # reliable signal path.
        #
        # At this point, x has been enriched with cross-position
        # information from self-attention.  It now carries both the
        # original token embeddings AND information about how each
        # token relates to every other token.
        # -----------------------------------------------------------
        x = self.norm1(x + self.dropout(attn_out))
        # x: (B, S, d_model) — same shape, now with attention context baked in

        # -----------------------------------------------------------
        # SUB-LAYER 2: Position-wise feed-forward network
        # -----------------------------------------------------------
        #
        #   ff_out = FFN(x) = ReLU(x @ W₁ + b₁) @ W₂ + b₂
        #
        # Shape trace:
        #   input x:        (B, S, d_model)     [e.g. (64, 10, 128)]
        #   linear1(x):     (B, S, d_ff)        [e.g. (64, 10, 512)]
        #   ReLU(...):      (B, S, d_ff)        same shape
        #   linear2(...):   (B, S, d_model)     [e.g. (64, 10, 128)]
        #
        # Position-wise means each position (token) is processed
        # independently with the SAME weights.  Position 0 and position 9
        # go through identical linear layers but with different input
        # vectors, so they produce different outputs.
        #
        # The expanded d_ff=512 hidden layer gives the ReLU a rich
        # high-dimensional space to carve decision boundaries in.
        # See FeedForward.__init__ for the detailed justification.
        #
        # After self-attention pooled information across positions,
        # the FFN now processes each position's enriched context
        # individually.  This separation of concerns — attention mixes
        # across positions, FFN transforms within positions — is the
        # fundamental decomposition that makes Transformers work.
        # -----------------------------------------------------------
        ff_out = self.ffn(x)
        # ff_out: (B, S, d_model) — position-wise transformed, same shape

        # -----------------------------------------------------------
        # ADD & NORM #2: residual connection + layer normalization
        # -----------------------------------------------------------
        #
        #   x = LayerNorm( x + Dropout(ff_out) )
        #
        # Identical pattern to Add & Norm #1, but with the FFN output
        # instead of the attention output.  The residual x now carries:
        #
        #   x = original token embeddings
        #     + positional encodings
        #     + self-attention contributions from block 1
        #     + FFN contributions from block 1
        #     + self-attention contributions from block 2
        #     + FFN contributions from block 2
        #     + ...
        #     + self-attention contributions from THIS block
        #     + FFN contributions from THIS block  ← just added now
        #
        # Every operation in the encoder is just INCREMENTING the
        # residual stream.  Nothing is ever removed from it.
        # -----------------------------------------------------------
        x = self.norm2(x + self.dropout(ff_out))
        # x: (B, S, d_model) — final output of this encoder block

        # -----------------------------------------------------------
        # RETURN: refined hidden states + attention weights for viz
        # -----------------------------------------------------------
        #
        # We return both the hidden state (to pass to the next block
        # or to the decoder's cross-attention) and the attention
        # weights (for plotting heatmaps in visualize.py).
        #
        # The attention weights are the softmax scores from the
        # self-attention sublayer — a (B, H, S, S) tensor where
        # element [b, h, i, j] tells us how much head h's query at
        # position i attended to key at position j.
        #
        # For the reversal task, typical encoder self-attention patterns:
        #   - Lower layers: local attention (each position attends to
        #     its immediate neighbours ±1-2 positions).
        #   - Middle layers: broader positional awareness.
        #   - Higher layers: heads specialize — some capture position
        #     identity, some capture token identity.
        #
        # The Encoder class collects these weights in a list (one per
        # layer) and returns them alongside the final encoder output.
        # -----------------------------------------------------------
        return x, attn_weights


# ===========================================================================
# EncoderBlockPreLN — Pre-LN sublayers + per-block output-norm (hybrid)
# ===========================================================================


class EncoderBlockPreLN(nn.Module):
    """
    A single encoder layer with Pre-LN sublayers AND a LayerNorm on the
    block's residual OUTPUT (norm3) — a self-contained, self-normalizing block.

    ┌────────────────────────────────────────────────────────────────────────┐
    │        PRE-LN ENCODER BLOCK WITH PER-BLOCK OUTPUT NORM                 │
    │                                                                        │
    │   (input x is already normalized by the previous block's norm3)        │
    │                                                                        │
    │        x   (B, S, d_model)                                             │
    │        │                                                               │
    │        ├──────────────┐   split: copy x → residual skip (stays raw)    │
    │        │              │                                                │
    │   ┌────┴─────┐        │                                                │
    │   │LayerNorm │ ← norm1│   Pre-LN: norm is ON THE BRANCH, before attn   │
    │   └────┬─────┘        │                                                │
    │   ┌────┴─────┐        │                                                │
    │   │Self-Attn │        │   Q = K = V = norm1(x)                         │
    │   └────┬─────┘        │                                                │
    │   ┌────┴─────┐        │                                                │
    │   │ Dropout  │        │                                                │
    │   └────┬─────┘        │                                                │
    │        │              │                                                │
    │       (+)◄────────────┘   z = x + Dropout(Attn(norm1(x)))              │
    │        │                     ↑ trunk stays clean — NO norm here        │
    │        ├──────────────┐   split again                                  │
    │        │              │                                                │
    │   ┌────┴─────┐        │                                                │
    │   │LayerNorm │ ← norm2│   norm on the branch, before the FFN           │
    │   └────┬─────┘        │                                                │
    │   ┌────┴─────┐        │                                                │
    │   │  Feed-   │        │   ffn(norm2(z))                                │
    │   │ Forward  │        │                                                │
    │   └────┬─────┘        │                                                │
    │   ┌────┴─────┐        │                                                │
    │   │ Dropout  │        │                                                │
    │   └────┬─────┘        │                                                │
    │        │              │                                                │
    │       (+)◄────────────┘   y = z + Dropout(FFN(norm2(z)))               │
    │        │                     ↑ trunk still clean                       │
    │   ┌────┴─────┐                                                         │
    │   │LayerNorm │ ← norm3     block-output norm (the hybrid's extra one,  │
    │   └────┬─────┘             sits ON the trunk → out = norm3(y))         │
    │        │                                                               │
    │        ▼                                                               │
    │     output   (B, S, d_model)   ← NORMALIZED, self-contained            │
    │                                                                        │
    │   norm1/norm2 = sublayer input norms (Pre-LN — on the branch).         │
    │   norm3 = block-output norm (the only norm ON the residual trunk).     │
    │   Output IS normalized — no final ln_f needed after the stack.         │
    └────────────────────────────────────────────────────────────────────────┘

    Sublayers (in order):
        1. LayerNorm(x) → Multi-head self-attention → Dropout → residual add
        2. LayerNorm(x) → Feed-Forward → Dropout → residual add
        3. LayerNorm(x)  ← per-block output normalization (norm3)

    The complete formula:

        z   = x + Dropout( Attn(  norm1(x), norm1(x), norm1(x)  ) )
        y   = z + Dropout( FFN (  norm2(z)                     ) )
        out = norm3(y)   ← self-contained: every block outputs Var ≈ 1.0

    ----

    CONCEPT: Why three norms? — what each one does
    ──────────────────────────────────────────────

    norm1 — "attention in-gate":
      Normalizes the residual stream BEFORE self-attention.  W_q/k/v always
      see zero-mean unit-variance vectors, regardless of stack depth.

    norm2 — "FFN in-gate":
      Normalizes the residual (now with attention output) BEFORE the FFN.
      The ReLU and linear layers always receive well-behaved inputs.

    norm3 — "block out-gate":
      Normalizes the FULL accumulated residual AFTER both sublayers have
      contributed.  This makes each block self-contained: output variance
      is reset to ≈ 1.0 at every block boundary.  The Encoder needs no
      separate final `ln_f`; blocks are swappable with Post-LN blocks.

    Each norm has its OWN learned γ (init 1) and β (init 0):
      - norm1.γ: "amplify/suppress features before attention sees them"
      - norm2.γ: "amplify/suppress features before the FFN sees them"
      - norm3.γ: "amplify/suppress features before the next block sees them"

    ----

    CONCEPT: What this block IS (and what it is NOT)
    ─────────────────────────────────────────────────
    An earlier draft called this "Sandwich-LN." That is INCORRECT, and the
    distinction matters, so here it is precisely:

      Sandwich-LN (Ding et al., 2021, CogView) normalizes the INPUT AND
      OUTPUT of each SUB-LAYER'S BRANCH, leaving the residual path clean:
          x + LN_out( Sublayer( LN_in(x) ) )
      Both norms are on the branch; the residual `x` bypasses both. Its
      whole point is to STABILIZE the branch while KEEPING the identity
      highway pristine — that's why it enabled deep (64-layer) FP16 training.

      THIS block instead puts one LayerNorm (norm3) on the BLOCK'S RESIDUAL
      OUTPUT, i.e. AFTER the add, on the highway itself:
          out = LN( x + Attn(LN(x)) + FFN(LN(...)) )
      That is the OPPOSITE placement from Sandwich-LN. It's essentially
      "Pre-LN sublayers, then re-normalize the whole block like Post-LN
      does its output."

    This exact configuration is NOT a standard named production block. Real
    Pre-LN stacks use ONE final `ln_f` after the ENTIRE stack, not a norm
    per block — precisely because a per-block norm on the residual path
    breaks the clean gradient highway (see the tradeoff below) for little
    gain. Treat this class as a pedagogical hybrid, not a canonical design.

    To see why, trace the gradient across two blocks:

        Block N:     o_N  = norm3_N( r_N )
        Block N+1:   o_{N+1} = norm3_{N+1}( o_N + Attn(norm1(o_N)) + ... )

    Gradient from o_{N+1} back to o_N:

        ∂o_{N+1}/∂o_N = ∂norm3_{N+1}/∂r_{N+1} · ( I + attn_branch + ffn_branch )

    The identity term I is clean, BUT it is PRE-MULTIPLIED by ∂norm3_{N+1}.
    Continuing back through block N multiplies in ∂norm3_N.  After K blocks,
    the cross-block identity gradient passes through K LayerNorms — one
    norm3 per block.  This DOES compound across the stack.

    Where each design lands on the cross-block identity gradient path:

        Pure Pre-LN:       0 LayerNorms  → pristine highway, one final ln_f.
                            Strongest theoretical stability guarantee.

        Sandwich-LN:       0 LayerNorms on the highway → extra norms are on
                            the BRANCH, so the highway stays clean too. (This
                            is why it's NOT the same as this block.)

        THIS block:        K LayerNorms  → intermediate.  Better than Post-LN
                            (2K), but the norm3-on-residual choice gives up
                            the clean highway that pure Pre-LN and Sandwich-LN keep.

        Post-LN:           2K LayerNorms → needs LR warmup (paper: 4,000 steps).

    Why this block USUALLY still trains without warmup:
      At initialization, residual variance ≈ 1.0, so σ ≈ 1 and
      ∂norm3/∂r ≈ I — the LayerNorm backward Jacobian is nearly identity.
      Early in training, norm3 barely distorts gradients.  This is an
      empirical "usually fine," not a theoretical guarantee.  For very
      deep stacks (24+ blocks), prefer pure Pre-LN + ln_f, or Sandwich-LN.

    The WITHIN-BLOCK gradient flow IS governed by correct Pre-LN dynamics:
      - `x + dropout(attn_out)` and `z + dropout(ff_out)` have clean
        identity gradients ∂L/∂y · 1 (no LayerNorm on the residual path
        within the block).
      - norm1/norm2 sit only on the SUBLAYER BRANCHES — their backward
        Jacobians affect only branch gradients, not the identity path.

    So the honest summary:

        GAIN:  self-contained blocks — every block outputs Var ≈ 1.0.
               No final ln_f needed.  Blocks are drop-in swappable with
               Post-LN blocks.  Sublayer inputs are always normalized.

        COST:  the clean cross-block gradient highway is partially given
               up.  One LayerNorm per block sits on the output path, so
               cross-block gradients traverse K ∂norm3 terms in a K-block
               stack.  ~256 extra params per block (negligible in params,
               but the real cost is the gradient path, not the param count).

    ----

    CONCEPT: The gradient flow — why Pre-LN sublayers are stable
    ───────────────────────────────────────────────────────────

    For ONE Pre-LN sublayer (as used within this block):

        y = x + Sublayer( LayerNorm(x) )

    By the chain rule:

        ∂L/∂x = ∂L/∂y  +  (∂L/∂y · ∂Sublayer/∂h · ∂LayerNorm/∂x)
                  ↑                      ↑
                  │                      └── sublayer branch (goes through LN backward)
                  │
                  └── IDENTITY gradient: ∂L/∂y · 1
                      NEVER touched by LayerNorm. Always intact.

    Contrast with Post-LN where the ENTIRE gradient (both paths) passes
    through LayerNorm backward, whose 1/σ factor can amplify gradients
    when σ is small — the root cause of Post-LN's warmup requirement.

    ----

    CONCEPT: Variance accumulation solved — norm3 resets per block
    ─────────────────────────────────────────────────────────────

    Pure Pre-LN accumulates variance with depth:

        Block 1 output:  Var ≈ 1 + Var(attn₁) + Var(ffn₁)
        Block N output:  Var ≈ 1 + Σᵢ(Var(attnᵢ) + Var(ffnᵢ))  ≈ 1 + N·ε

    With norm3, each block is self-normalizing:

        Block 1 output:  LN₃( x + attn₁(LN₁(x)) + ffn₁(...) )  → Var ≈ 1.0
        Block 2 output:  LN₃( o₁ + attn₂(LN₁(o₁)) + ffn₂(...) ) → Var ≈ 1.0

    No linear growth.  The output variance is depth-independent, same as
    Post-LN, but with better gradient flow (1 LN/block vs 2 LN/block on
    the cross-block path).  The Encoder needs no final ln_f.

    ----

    CONCEPT: Training dynamics — per-layer gradient access
    ──────────────────────────────────────────────────────

    Cross-block identity gradients by variant (N-block stack):

        Post-LN:         Loss → LN → LN → ... → LN → Embeddings
                         Gradient passes through 2N LayerNorms.

        THIS block:      Loss → n3 → n3 → ... → n3 → Embeddings
                         Gradient passes through N LayerNorms (one norm3/block).
                         Sublayer norms (n1, n2) sit on branches, not the path.

        Pure Pre-LN:     Loss → Embeddings  (clean, 0 LN on residual path)
                         One final ln_f after the stack.

        Sandwich-LN:     Same as Pure Pre-LN — 0 LN on the highway.
                         Extra norms are on the sublayer BRANCH only.

    All variants give early layers SOME gradient access (unlike vanilla
    feedforward nets which die at depth), but the QUALITY of that access
    differs.  Pure Pre-LN and Sandwich-LN give the cleanest highway;
    this block is intermediate; Post-LN is the most obstructed.

    ----

    CONCEPT: Historical context
    ───────────────────────────

    The paper used Post-LN following the ResNet convention of the time
    ("residual → normalize").  Pre-LN (x + Sublayer(LN(x))) was popularized
    by GPT-2 (2019) and analyzed by Xiong et al. (2020); it became the
    de facto standard by 2020 because it removes the warmup requirement.
    Later variants target deep-stack stability differently — e.g. Sandwich-LN
    (Ding et al., 2021: extra branch norms), DeepNorm (Wang et al.: upscale
    the residual before LN), B2T (Takase et al., 2022: bypass all but the
    final LN per layer).

    The per-block-output-norm used HERE is mostly of educational value: it
    makes each block self-contained, so the Encoder doesn't need to know
    whether its blocks are Post-LN or Pre-LN, and you can even mix both in
    one stack. Just don't mistake it for a mainstream production choice.

    ----

    CONCEPT: Same interface, drop-in replacement (with a caveat)
    ────────────────────────────────────────────────────────────

    EncoderBlockPreLN has the SAME:
      - Forward signature:      forward(x, src_mask=None) → (x, attn_weights)
      - Input shape:            (batch, src_seq_len, d_model)
      - Output shape:           (batch, src_seq_len, d_model)
      - Output normalization:   YES (same guarantee as Post-LN)
      - Return type:            tuple[Tensor, Tensor]

    Differences from EncoderBlock (Post-LN):
      - +1 LayerNorm (norm3) → 198,528 params vs 198,272 (+256).
      - Sublayer inputs are normalized (Pre-LN), not raw.
      - Cross-block gradient passes through 1 LN/block (vs 2 for Post-LN).
      - Attention weights are over norm1(x) not raw x — a minor caveat for
        visualization: the softmax patterns may differ slightly from Post-LN
        because the Q/K/V projections see normalized rather than raw features.

    These blocks can freely SWAP in the Encoder stack with no code changes
    to the Encoder class, training loop, or checkpointing.

    Interface caveat: this block's output is ALREADY normalized (norm3). If
    your decoder cross-attention is itself Pre-LN and re-normalizes its K/V,
    you'd normalize twice. Usually cross-attention consumes encoder output
    directly as K/V, so normalized K/V is fine — just confirm the convention
    matches on both sides.

    ----

    PARAMETER INVENTORY (d_model=128, d_ff=512, num_heads=4):

      Multi-head self-attention:   4 × (128×128 + 128)          =  66,048
      Feed-forward:                    2×128×512 + 512 + 128    = 131,712
      LayerNorm × 3:               3 × (128 + 128)              =     768
      ─────────────────────────────────────────────────────────────────
      Total per block:                                           198,528
    """

    # ---------------------------------------------------------------
    # CONSTRUCTOR — six modules (three norms instead of two)
    # ---------------------------------------------------------------
    #
    # The constructor stores the SAME five modules as EncoderBlock PLUS
    # one extra LayerNorm (norm3).  This extra norm is what makes the
    # block self-normalizing — it guarantees the output is normalized
    # without needing a separate final LayerNorm in the Encoder class.
    #
    # The semantic roles of the three norms compared to Post-LN:
    #
    #   Module    Post-LN role                    Pre-LN hybrid role
    #   ──────    ────────────                    ──────────────────
    #   norm1     Normalize x + Attn(x) output    Normalize x BEFORE Attn
    #                                            → "attention input gate"
    #   norm2     Normalize x + FFN(x) output     Normalize x BEFORE FFN
    #                                            → "FFN input gate"
    #   norm3     (doesn't exist)                 Normalize FULL block output
    #                                            → "block export gate"
    #   dropout   Dropout on Attn/FFN OUTPUT       Dropout on Attn/FFN OUTPUT
    #             (BEFORE residual add)            (BEFORE residual add)
    #   self_attn Working on raw x                Working on LN₁(x)
    #   ffn       Working on raw x                Working on LN₂(x)
    #
    # Post-LN gets normalized output from norm2 (which normalizes the
    # FFN residual).  This block gets normalized output from norm3
    # (a dedicated block-output normalizer).  Both achieve the same
    # end-to-end guarantee: every block outputs Var≈1.0 features.
    #
    # Dropout placement is UNCHANGED — still applied to the SUBLAYER
    # OUTPUT before the residual add (§5.4), regardless of LN placement.
    # ---------------------------------------------------------------

    def __init__(self, config: TransformerConfig):
        super().__init__()

        # -----------------------------------------------------------
        # (1) Multi-head self-attention — receives norm1(x) as input.
        #
        # The attention mechanism always operates on NORMALIZED vectors
        # (norm1 strips away depth-dependent statistics), ensuring
        # consistent Q/K/V distributions regardless of how many blocks
        # precede this one.  Same MultiHeadAttention class, same
        # computation — only the input changes vs. Post-LN.
        # -----------------------------------------------------------
        self.self_attn = MultiHeadAttention(config)

        # -----------------------------------------------------------
        # (2) First LayerNorm — "attention input gate"
        #
        # In Post-LN, norm1 normalizes the OUTPUT of attention+residual.
        # Here, norm1 normalizes the INPUT to attention.
        #
        # Its learned γ (scale) and β (shift) answer:
        #   "Which features should be amplified or suppressed before
        #    the attention mechanism sees them?"
        #
        # Since the network learns γ and β during training, it adapts
        # to whichever role the architecture assigns.  The only thing
        # that changes is WHAT norm1 normalizes (input vs. output).
        # -----------------------------------------------------------
        self.norm1 = nn.LayerNorm(config.d_model)

        # -----------------------------------------------------------
        # (3) Feed-forward — receives norm2(x) as input.
        #
        # The FFN always operates on normalized vectors, ensuring
        # consistent ReLU activation statistics at every depth.
        # -----------------------------------------------------------
        self.ffn = FeedForward(config)

        # -----------------------------------------------------------
        # (4) Second LayerNorm — "FFN input gate"
        #
        # norm2 normalizes the residual stream (now enriched with the
        # attention sublayer's contribution) before feeding it to the
        # feed-forward network.  Same nn.LayerNorm, separate γ/β —
        # norm2 learns what the FFN needs, norm1 learns what attention
        # needs.  Different sublayers, different input distributions,
        # different normalization parameters.
        # -----------------------------------------------------------
        self.norm2 = nn.LayerNorm(config.d_model)

        # -----------------------------------------------------------
        # (5) Third LayerNorm — "block out-gate"  ← THE EXTRA MODULE
        #
        # norm3 normalizes the FULL accumulated residual AFTER both
        # sublayers have contributed, so the block output is clean for
        # the next block or the decoder's cross-attention.
        #
        # [REVIEW] Placement note: norm3 sits on the RESIDUAL OUTPUT path
        # (after the add). This is NOT Sandwich-LN (whose extra norms are
        # on the sublayer BRANCH). See the class docstring for the full
        # distinction and the gradient tradeoff it implies.
        #
        # Without norm3 (pure Pre-LN):
        #   output = x + attn(LN₁(x)) + ffn(LN₂(x + attn(LN₁(x))))
        #   → variance grows with depth, need final ln_f after stack
        #
        # With norm3 (this block):
        #   output = LN₃( x + attn(LN₁(x)) + ffn(LN₂(x + attn(LN₁(x)))) )
        #   → normalized output, no final ln_f needed, self-contained!
        #
        # The REAL cost is NOT the 256 extra params (negligible).  It is
        # that norm3 sits on the BLOCK-OUTPUT path, so the cross-block
        # gradient passes through one LayerNorm per block.  Pure Pre-LN
        # has a pristine identity highway (0 LN); this block has K
        # ∂norm3 terms in a K-block stack.  See the class docstring
        # ("What this block IS") for the full gradient derivation.
        #
        # Its learned γ answers:
        #   "After all sublayers have contributed, which features
        #    should be amplified before passing to the next block?"
        # -----------------------------------------------------------
        self.norm3 = nn.LayerNorm(config.d_model)

        # -----------------------------------------------------------
        # (6) Dropout — same role: applied to SUBLAYER OUTPUT before
        #     the residual add.
        #
        # Dropout placement is IDENTICAL across all variants:
        #   Both:  Dropout( Sublayer(...) )  ← before the residual add
        #   NOT:   Sublayer( Dropout(...) )  ← dropout on input
        #   NOT:   Dropout( x + ... )       ← dropout after the add
        #
        # The paper (§5.4) is explicit about this.  LN placement
        # (Pre vs. Post) doesn't affect where dropout goes.
        # -----------------------------------------------------------
        self.dropout = nn.Dropout(config.dropout)

    def extra_repr(self) -> str:
        """Surface d_model and variant in print(model) for quick inspection."""
        return f"d_model={self.norm1.normalized_shape[0]}, variant=pre_ln+out_norm"

    # ---------------------------------------------------------------
    # FORWARD PASS — Pre-LN sublayers + final block-level norm (norm3)
    # ---------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one encoder block with Pre-LN sublayers + final norm.

        The computation:
            z = x + Dropout( Attn(  norm1(x), norm1(x), norm1(x)  ) )
            y = z + Dropout( FFN (  norm2(z)                     ) )
            out = norm3(y)   ← final normalization guarantees clean output

        Args:
            x:        (batch, src_seq_len, d_model)
                      Hidden states from the previous encoder block
                      (or from the embedding layer for block 1).
                      For Pre-LN, this is the normalized output of
                      the PREVIOUS block (norm3 guarantees this), so
                      each block starts from well-conditioned input.

            src_mask: (batch, 1, 1, src_seq_len) or None.
                      Padding mask for self-attention keys.

        Returns:
            x:              (batch, src_seq_len, d_model)
                            Refined hidden states — NORMALIZED by norm3.
                            Same end-to-end guarantee as EncoderBlock
                            (Post-LN): the output is clean, zero-mean,
                            unit-variance (after learned γ, β).  No
                            extra final LayerNorm needed in the Encoder
                            class — each block is self-contained.

            attn_weights:   (batch, num_heads, src_seq_len, src_seq_len)
                            Softmax attention weights for visualization.
                            NOTE: these are over norm1(x), not raw x —
                            expected for Pre-LN, just a minor caveat
                            when comparing heatmaps with Post-LN blocks.
        """

        # -----------------------------------------------------------
        # SUB-LAYER 1:  Pre-LN → Self-attention → Dropout → Add
        # -----------------------------------------------------------
        #
        #   attn_out, attn_weights = self_attn(norm1(x), norm1(x), norm1(x))
        #   x = x + Dropout(attn_out)
        #
        # Compare with Post-LN (EncoderBlock):
        #   attn_out, attn_weights = self_attn(x, x, x)
        #   x = LayerNorm(x + Dropout(attn_out))
        #
        # The critical differences:
        #
        #   (1) norm1(x) is passed as Q, K, V instead of raw x.
        #       The attention projections (W_q, W_k, W_v) operate on
        #       NORMALIZED vectors — mean≈0, std≈1 across the d_model
        #       features.  This means the projection weight matrices
        #       always "see" the same input statistics regardless of
        #       depth in the stack.
        #
        #   (2) The residual add is JUST x + dropout(attn_out) — NO
        #       LayerNorm wraps it.  The identity path (x) carries the
        #       full residual stream through unmasked.  The gradient
        #       through this identity path is ∂L/∂y · 1 — always intact,
        #       never passing through a LayerNorm backward.  This is the
        #       mathematical root of Pre-LN's stability.
        #
        #   (3) LayerNorm is called ONCE and its output is reused for
        #       Q, K, V.  No need to normalize separately — they should
        #       all operate on the same conditioned input.
        #
        # Shape trace:
        #   input x:               (B, S, d_model)      from prev block (normalized)
        #   norm1(x):              (B, S, d_model)      mean≈0, std≈1
        #   self_attn(..., mask):  (B, S, d_model)      attention output
        #   dropout(attn_out):     (B, S, d_model)      ~10% zeroed (train only)
        #   x + dropout(...):      (B, S, d_model)      residual, UN-normalized
        # -----------------------------------------------------------
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            query=normed, key=normed, value=normed, mask=src_mask
        )
        # attn_out:     (B, S, d_model) — attention output
        # attn_weights: (B, H, S, S)    — for visualization

        # ---- Residual add #1 (no LayerNorm — Pre-LN pattern) ----
        #
        #   x = x + Dropout(attn_out)
        #
        # The IDENTITY path (x) carries the entire accumulated residual
        # from all previous processing.  The attention sublayer ADDS its
        # contribution.  No LayerNorm follows the add — the variance has
        # increased but will be normalized by norm3 at the end of this
        # block (or by norm2 before the FFN, which always normalizes
        # unconditionally — no "if" about it).
        #
        # This is the signature Pre-LN pattern: normalize on the
        # SUBLAYER BRANCH, let the residual path carry raw values.
        # -----------------------------------------------------------
        x = x + self.dropout(attn_out)
        # x: (B, S, d_model) — residual accumulated, UN-normalized

        # -----------------------------------------------------------
        # SUB-LAYER 2:  Pre-LN → Feed-forward → Dropout → Add
        # -----------------------------------------------------------
        #
        #   ff_out = ffn(norm2(x))
        #   x = x + Dropout(ff_out)
        #
        # Same Pre-LN pattern, now applied to the FFN.
        #
        # Compare with Post-LN:
        #   ff_out = ffn(x)
        #   x = LayerNorm(x + Dropout(ff_out))
        #
        # The FFN receives norm2(x) — a freshly normalized version of
        # the residual (now enriched with attention output).  The ReLU
        # and linear layers always see consistent input statistics.
        #
        # After this add, x now contains the contributions of BOTH
        # sublayers added to the residual stream.  The variance has
        # accumulated further, but norm3 (next step) will normalize it.
        # -----------------------------------------------------------
        normed = self.norm2(x)
        ff_out = self.ffn(normed)
        # ff_out: (B, S, d_model) — FFN output on normalized input

        x = x + self.dropout(ff_out)
        # x: (B, S, d_model) — both sublayers have contributed,
        #    still UN-normalized at this point

        # -----------------------------------------------------------
        # FINAL NORMALIZATION — norm3, the "block out-gate"
        # -----------------------------------------------------------
        #
        #   x = norm3(x)
        #
        # After both sublayers added their contributions to the residual,
        # norm3 normalizes the accumulated result.  This makes the block
        # SELF-CONTAINED: every block outputs Var ≈ 1.0, so the Encoder
        # needs no final ln_f, and blocks are swappable with Post-LN.
        #
        # Why this differs from pure Pre-LN (GPT-2 style):
        #   - Pure Pre-LN outputs un-normalized residual → variance grows
        #     linearly with depth → needs one final ln_f after the stack.
        #   - This block normalizes per-block → depth-independent output
        #     variance → no final ln_f needed.
        #
        # [REVIEW] Gradient caveat — what norm3 costs:
        #   Because norm3 sits on the block-OUTPUT (residual) path, the
        #   cross-block identity gradient passes through ∂norm3 once per
        #   block. In a K-block stack that is K LayerNorm backward passes.
        #   Pure Pre-LN has 0 (clean highway); Sandwich-LN also keeps the
        #   highway clean (its norms are on the branch); Post-LN has 2K.
        #   So this block is intermediate — better than Post-LN, but it
        #   gives up the pristine highway.
        #
        #   Empirically it still trains without warmup at moderate depth
        #   because at initialization σ≈1 so ∂norm3≈I.  For very deep
        #   stacks (24+), benchmark against pure Pre-LN + ln_f.
        #
        #   The WITHIN-BLOCK gradient flow remains correct Pre-LN:
        #   `x + dropout(attn_out)` and `z + dropout(ff_out)` have clean
        #   identity gradient terms — norm1/norm2 are on the sublayer
        #   branches only, not on the residual path.
        #
        # Shape:
        #   input:  (B, S, d_model) — un-normalized residual after both sublayers
        #   output: (B, S, d_model) — mean≈0, std≈1 per token
        #                            (after learned γ₃, β₃)
        # -----------------------------------------------------------
        x = self.norm3(x)
        # x: (B, S, d_model) — NORMALIZED block output. Ready for:
        #   - The next encoder block (receives normalized input)
        #   - The decoder's cross-attention (expects normalized K, V)

        # -----------------------------------------------------------
        # RETURN (same interface as EncoderBlock)
        # -----------------------------------------------------------
        #
        # Both output (x) and attention weights (attn_weights) returned.
        # x is NORMALIZED (norm3 applied) — same guarantee as Post-LN.
        # attn_weights are the softmax scores for visualization.
        # -----------------------------------------------------------
        return x, attn_weights


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
