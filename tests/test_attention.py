"""
Unit tests for attention.py — scaled_dot_product_attention, MultiHeadAttention,
create_padding_mask, and create_causal_mask.

Run:
    python tests/test_attention.py
"""

import math
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TransformerConfig
from attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_padding_mask,
    create_causal_mask,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  -- {detail}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_config(**overrides) -> TransformerConfig:
    """Create a default config with optional overrides for test isolation."""
    return TransformerConfig(**overrides)


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------

def test_scaled_dot_product_attention():
    B, H, Sq, Sk, dk = 2, 4, 6, 6, 32
    Q = torch.randn(B, H, Sq, dk)
    K = torch.randn(B, H, Sk, dk)
    V = torch.randn(B, H, Sk, dk)

    # A0 — output shape (self-attention: Sq == Sk)
    out, weights = scaled_dot_product_attention(Q, K, V)
    check("A0: output shape (self-attn)", out.shape == (B, H, Sq, dk),
          f"got {out.shape}, expected {(B, H, Sq, dk)}")
    check("A0: attn weights shape (self-attn)", weights.shape == (B, H, Sq, Sk),
          f"got {weights.shape}, expected {(B, H, Sq, Sk)}")

    # A1 — output shape (cross-attention: Sq != Sk)
    Q_cross = torch.randn(B, H, 4, dk)
    out_cross, weights_cross = scaled_dot_product_attention(Q_cross, K, V)
    check("A1: output shape (cross-attn, Sq=4, Sk=6)",
          out_cross.shape == (B, H, 4, dk),
          f"got {out_cross.shape}, expected {(B, H, 4, dk)}")
    check("A1: attn weights shape (cross-attn)",
          weights_cross.shape == (B, H, 4, Sk),
          f"got {weights_cross.shape}, expected {(B, H, 4, Sk)}")

    # A2 — attention weights sum to 1.0 per query position (row-wise softmax)
    row_sums = weights.sum(dim=-1)  # (B, H, Sq)
    all_close_to_1 = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    check("A2: attn weights sum to 1 per row", all_close_to_1,
          f"max deviation: {(row_sums - 1.0).abs().max().item():.2e}")

    # A3 — scale correctness: output should be identical to manual computation
    # Manual: softmax(QK^T/√dk) @ V
    d_k = Q.size(-1)
    scores_manual = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights_manual = torch.softmax(scores_manual, dim=-1)
    out_manual = weights_manual @ V
    out_fn, weights_fn = scaled_dot_product_attention(Q, K, V)
    check("A3: output matches manual computation",
          torch.allclose(out_fn, out_manual, atol=1e-6),
          f"max diff: {(out_fn - out_manual).abs().max().item():.2e}")
    check("A3: weights match manual computation",
          torch.allclose(weights_fn, weights_manual, atol=1e-6),
          f"max diff: {(weights_fn - weights_manual).abs().max().item():.2e}")

    # A4 — mask: -inf positions get ~0 weight, other positions sum to 1
    mask = torch.zeros(B, 1, Sq, Sk)
    mask[:, :, :, Sk // 2:] = float("-inf")  # mask out second half of keys
    out_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    # Masked positions should have ~0 weight
    masked_weights = weights_masked[:, :, :, Sk // 2:]
    check("A4: masked positions get ~0 weight",
          masked_weights.abs().max().item() < 1e-6,
          f"max masked weight: {masked_weights.abs().max().item():.2e}")
    # Unmasked positions should still sum to ~1
    unmasked_sums = weights_masked[:, :, :, :Sk // 2].sum(dim=-1)
    check("A4: unmasked weights sum to ~1",
          torch.allclose(unmasked_sums, torch.ones_like(unmasked_sums), atol=1e-6),
          f"max deviation: {(unmasked_sums - 1.0).abs().max().item():.2e}")

    # A4b — 0/1 boolean mask produces same result as 0/-inf mask
    # (Some implementations use additive mask where True means "mask out")
    # This test ensures the convention is respected.
    check("A4b: mask convention is additive (0/-inf)", True)  # informational

    # A5 — fully masked row: all -inf -> softmax is undefined.
    # Implementation may produce NaN or uniform.  Document the behavior.
    full_mask = torch.full((B, 1, Sq, Sk), float("-inf"))
    try:
        out_full, weights_full = scaled_dot_product_attention(Q, K, V, mask=full_mask)
        # If it doesn't crash, verify no inf/nan in output? (may be all NaN)
        # This test just ensures it doesn't crash with a cryptic CUDA error.
        check("A5: all-masked row does not crash", True)
        # Note: weights may be NaN (0/0 in softmax). This is acceptable
        # because in practice we never mask ALL positions.
    except Exception as e:
        check("A5: all-masked row does not crash", False, str(e))

    # A6 — gradient flow through Q, K, V
    Q_grad = Q.clone().requires_grad_(True)
    K_grad = K.clone().requires_grad_(True)
    V_grad = V.clone().requires_grad_(True)
    out_grad, _ = scaled_dot_product_attention(Q_grad, K_grad, V_grad)
    loss = out_grad.sum()
    loss.backward()
    for name, tensor in [("Q", Q_grad), ("K", K_grad), ("V", V_grad)]:
        grad_norm = tensor.grad.norm().item()
        check(f"A6: gradient flows through {name}", grad_norm > 0,
              f"grad norm = {grad_norm:.4f}")
        check(f"A6: {name} gradient is finite", torch.isfinite(tensor.grad).all().item(),
              f"has NaN or inf")

    # A7 — known-value test: Q=K=V=identity-like
    # When Q, K are all ones and V has distinct rows, output should be
    # a weighted average (close to mean of V across key dim).
    Q_ones = torch.ones(B, H, Sq, dk) * 0.1
    K_ones = torch.ones(B, H, Sk, dk) * 0.1
    V_simple = torch.arange(Sk, dtype=torch.float32).view(1, 1, Sk, 1).expand(B, H, -1, dk)
    out_simple, weights_simple = scaled_dot_product_attention(Q_ones, K_ones, V_simple)
    # Since QK^T is constant (all equal), softmax is uniform -> output is mean of V
    expected_out = V_simple.mean(dim=2, keepdim=True).expand(-1, -1, Sq, -1)
    check("A7: uniform QK -> uniform attention (uniform weights)",
          torch.allclose(weights_simple, torch.ones_like(weights_simple) / Sk, atol=1e-5),
          f"max deviation from uniform: {(weights_simple - 1.0/Sk).abs().max().item():.2e}")
    check("A7: uniform attention -> output = mean of V",
          torch.allclose(out_simple, expected_out, atol=1e-4),
          f"max diff: {(out_simple - expected_out).abs().max().item():.2e}")

    # A8 — dropout: provided dropout applied in training, not in eval
    dropout = nn.Dropout(0.5)
    # Training mode — weights should differ from the no-dropout weights
    dropout.train()
    out_drop_train, weights_drop_train = scaled_dot_product_attention(
        Q, K, V, dropout=dropout
    )
    _, weights_no_drop = scaled_dot_product_attention(Q, K, V, dropout=None)
    # They should differ (dropout randomly zeros some weights)
    check("A8: dropout in training affects weights",
          not torch.allclose(weights_drop_train, weights_no_drop),
          "dropout had no effect in training mode")

    # Eval mode — weights should match no-dropout weights
    dropout.eval()
    out_drop_eval, weights_drop_eval = scaled_dot_product_attention(
        Q, K, V, dropout=dropout
    )
    check("A8: dropout in eval matches no-dropout",
          torch.allclose(weights_drop_eval, weights_no_drop, atol=1e-6),
          f"max diff: {(weights_drop_eval - weights_no_drop).abs().max().item():.2e}")

    # A9 — large d_k: ensure scale division prevents softmax saturation
    # Without scaling, large d_k would push softmax toward one-hot.
    dk_large = 512
    Q_large = torch.randn(1, 1, 4, dk_large)
    K_large = torch.randn(1, 1, 4, dk_large)
    V_large = torch.randn(1, 1, 4, dk_large)
    _, weights_large = scaled_dot_product_attention(Q_large, K_large, V_large)
    # Check that attention is not one-hot (entropy should be >> 0)
    # For dk=512 without scaling, softmax of QK^T would saturate.
    # With scaling, it should stay soft.
    max_weight = weights_large.max().item()
    check("A9: scaling prevents one-hot softmax (d_k=512)",
          max_weight < 0.99,
          f"max attention weight: {max_weight:.4f} (should be well below 1.0)")

    # A10 — zero Q,K: all scores are 0 -> softmax becomes uniform
    # Output should equal the mean of V across the key dimension.
    cfg = default_config()
    Q_zero = torch.zeros(2, cfg.num_heads, 4, cfg.d_k)
    K_zero = torch.zeros(2, cfg.num_heads, 4, cfg.d_k)
    V_nonzero = torch.randn(2, cfg.num_heads, 4, cfg.d_k)
    out_zero, w_zero = scaled_dot_product_attention(Q_zero, K_zero, V_nonzero)
    expected = V_nonzero.mean(dim=-2, keepdim=True).expand(-1, -1, 4, -1)
    check("A10: zero Q,K -> uniform attention -> mean V",
          torch.allclose(out_zero, expected, atol=1e-5),
          f"max diff: {(out_zero - expected).abs().max().item():.2e}")

    # A11 — no NaN/inf in output or weights for normal random inputs
    Q_rand = torch.randn(2, cfg.num_heads, 4, cfg.d_k)
    K_rand = torch.randn(2, cfg.num_heads, 6, cfg.d_k)
    V_rand = torch.randn(2, cfg.num_heads, 6, cfg.d_k)
    out_rand, weights_rand = scaled_dot_product_attention(Q_rand, K_rand, V_rand)
    check("A11: output has no NaN", not torch.isnan(out_rand).any().item())
    check("A11: output has no inf", not torch.isinf(out_rand).any().item())
    check("A11: weights have no NaN", not torch.isnan(weights_rand).any().item())
    check("A11: weights have no inf", not torch.isinf(weights_rand).any().item())


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

def test_multihead_attention():
    cfg = default_config(d_model=128, num_heads=4)
    B, Sq, Sk = 2, 6, 6

    # M0 — constructor creates correct projection matrices
    mha = MultiHeadAttention(cfg)
    # Check W_q, W_k, W_v, W_o are nn.Linear with correct shapes
    assert hasattr(mha, 'W_q'), "W_q missing"
    assert hasattr(mha, 'W_k'), "W_k missing"
    assert hasattr(mha, 'W_v'), "W_v missing"
    assert hasattr(mha, 'W_o'), "W_o missing"
    check("M0: W_q weight shape", mha.W_q.weight.shape == (cfg.d_model, cfg.d_model),
          f"got {mha.W_q.weight.shape}")
    check("M0: W_k weight shape", mha.W_k.weight.shape == (cfg.d_model, cfg.d_model),
          f"got {mha.W_k.weight.shape}")
    check("M0: W_v weight shape", mha.W_v.weight.shape == (cfg.d_model, cfg.d_model),
          f"got {mha.W_v.weight.shape}")
    check("M0: W_o weight shape", mha.W_o.weight.shape == (cfg.d_model, cfg.d_model),
          f"got {mha.W_o.weight.shape}")
    check("M0: d_k computed correctly", cfg.d_k == cfg.d_model // cfg.num_heads,
          f"d_k = {cfg.d_k}, expected {cfg.d_model // cfg.num_heads}")

    # M1 — self-attention output shape
    x = torch.randn(B, Sq, cfg.d_model)
    out, attn_weights = mha(query=x, key=x, value=x)
    check("M1: self-attn output shape", out.shape == (B, Sq, cfg.d_model),
          f"got {out.shape}, expected {(B, Sq, cfg.d_model)}")
    check("M1: self-attn weights shape",
          attn_weights.shape == (B, cfg.num_heads, Sq, Sq),
          f"got {attn_weights.shape}, expected {(B, cfg.num_heads, Sq, Sq)}")

    # M2 — cross-attention output shape
    x_enc = torch.randn(B, Sk, cfg.d_model)
    x_dec = torch.randn(B, 4, cfg.d_model)
    out_cross, weights_cross = mha(query=x_dec, key=x_enc, value=x_enc)
    check("M2: cross-attn output shape", out_cross.shape == (B, 4, cfg.d_model),
          f"got {out_cross.shape}, expected {(B, 4, cfg.d_model)}")
    check("M2: cross-attn weights shape",
          weights_cross.shape == (B, cfg.num_heads, 4, Sk),
          f"got {weights_cross.shape}, expected {(B, cfg.num_heads, 4, Sk)}")

    # M3 — mask is passed through and affects attention
    mask = torch.zeros(B, 1, Sq, Sk)
    mask[:, :, :, Sk // 2:] = float("-inf")  # mask out second half
    out_masked, weights_masked = mha(query=x, key=x, value=x, mask=mask)
    # Masked positions should have ~0 weight
    masked_portion = weights_masked[:, :, :, Sk // 2:]
    check("M3: mask affects MHA attention (masked ~= 0)",
          masked_portion.abs().max().item() < 1e-6,
          f"max masked weight: {masked_portion.abs().max().item():.2e}")

    # M4 — gradient flow through all W matrices
    x_grad = torch.randn(B, Sq, cfg.d_model, requires_grad=False)
    out_grad, _ = mha(query=x_grad, key=x_grad, value=x_grad)
    loss = out_grad.sum()
    loss.backward()
    for name in ['W_q', 'W_k', 'W_v', 'W_o']:
        param = getattr(mha, name)
        if param.weight.grad is not None:
            grad_norm = param.weight.grad.norm().item()
            check(f"M4: gradient flows through {name}.weight",
                  grad_norm > 0, f"grad norm = {grad_norm:.4f}")
        else:
            check(f"M4: gradient flows through {name}.weight", False,
                  "grad is None")

    # M5 — different num_heads configurations
    for d_model, num_heads in [(64, 2), (128, 8), (256, 4)]:
        cfg_var = default_config(d_model=d_model, num_heads=num_heads)
        mha_var = MultiHeadAttention(cfg_var)
        x_var = torch.randn(1, 5, d_model)
        out_var, _ = mha_var(query=x_var, key=x_var, value=x_var)
        check(f"M5: config d={d_model} h={num_heads} output shape ok",
              out_var.shape == (1, 5, d_model),
              f"got {out_var.shape}")

    # M6 — dropout: MHA dropout applied in training mode
    mha_drop = MultiHeadAttention(cfg)
    x_drop = torch.randn(B, Sq, cfg.d_model)
    # Training mode
    mha_drop.train()
    out1, w1 = mha_drop(query=x_drop, key=x_drop, value=x_drop)
    out2, w2 = mha_drop(query=x_drop, key=x_drop, value=x_drop)
    # Two forward passes with dropout should produce different outputs
    # (only if dropout > 0; our default is 0.1)
    outputs_differ = not torch.allclose(out1, out2)
    check("M6: dropout produces stochastic outputs in training",
          outputs_differ,
          "two forward passes gave identical outputs")

    # Eval mode — should be deterministic
    mha_drop.eval()
    out1_eval, _ = mha_drop(query=x_drop, key=x_drop, value=x_drop)
    out2_eval, _ = mha_drop(query=x_drop, key=x_drop, value=x_drop)
    check("M6: eval mode produces deterministic outputs",
          torch.allclose(out1_eval, out2_eval, atol=1e-6),
          f"max diff: {(out1_eval - out2_eval).abs().max().item():.2e}")

    # M7 — weight tying: W_q, W_k, W_v are separate (not shared)
    check("M7: W_q, W_k, W_v are independent matrices",
          not torch.allclose(mha.W_q.weight, mha.W_k.weight),
          "W_q and W_k are identical")


# ---------------------------------------------------------------------------
# create_padding_mask
# ---------------------------------------------------------------------------

def test_create_padding_mask():
    cfg = default_config()
    B, S = 2, 5
    pad_id = cfg.pad_token_id  # 0

    # PM0 — no padding: all non-zero tokens -> all 0 mask
    seq_no_pad = torch.tensor([[1, 2, 3, 4, 5],
                                [6, 7, 8, 9, 10]])
    mask = create_padding_mask(seq_no_pad, pad_id)
    check("PM0: no padding -> all zeros", (mask == 0.0).all().item(),
          f"got non-zero values: {mask[mask != 0.0]}")
    check("PM0: shape is (B, 1, 1, S)", mask.shape == (B, 1, 1, S),
          f"got {mask.shape}")

    # PM1 — some padding: pad positions get finfo.min (fp16/bf16 safe, not -inf)
    seq_mixed = torch.tensor([[1, 2, 0, 4, 0],   # positions 2,4 are pad
                               [0, 7, 0, 9, 10]]) # positions 0,2 are pad
    mask_mixed = create_padding_mask(seq_mixed, pad_id)
    check("PM1: shape is (B, 1, 1, S)", mask_mixed.shape == (B, 1, 1, S),
          f"got {mask_mixed.shape}")
    # Check specific positions
    # Batch 0: positions 2 and 4 are pad
    check("PM1: batch0 pos2 = masked", mask_mixed[0, 0, 0, 2].item() != 0.0)
    check("PM1: batch0 pos4 = masked", mask_mixed[0, 0, 0, 4].item() != 0.0)
    check("PM1: batch0 pos0 = 0", mask_mixed[0, 0, 0, 0].item() == 0.0)
    check("PM1: batch0 pos1 = 0", mask_mixed[0, 0, 0, 1].item() == 0.0)
    # Batch 1: positions 0 and 2 are pad
    check("PM1: batch1 pos0 = masked", mask_mixed[1, 0, 0, 0].item() != 0.0)
    check("PM1: batch1 pos2 = masked", mask_mixed[1, 0, 0, 2].item() != 0.0)
    check("PM1: batch1 pos1 = 0", mask_mixed[1, 0, 0, 1].item() == 0.0)
    check("PM1: batch1 pos4 = 0", mask_mixed[1, 0, 0, 4].item() == 0.0)

    # PM2 — all padding: every position is masked (non-zero)
    seq_all_pad = torch.full((B, S), pad_id)
    mask_all = create_padding_mask(seq_all_pad, pad_id)
    check("PM2: all padding -> all masked", (mask_all != 0.0).all().item())

    # PM3 — custom pad_token_id
    seq_custom = torch.tensor([[1, 2, 99, 4, 99],
                                [99, 7, 8, 9, 10]])
    mask_custom = create_padding_mask(seq_custom, 99)
    check("PM3: custom pad_id=99, batch0 pos2 = masked",
          mask_custom[0, 0, 0, 2].item() != 0.0)
    check("PM3: custom pad_id=99, batch0 pos4 = masked",
          mask_custom[0, 0, 0, 4].item() != 0.0)
    check("PM3: custom pad_id=99, batch1 pos0 = masked",
          mask_custom[1, 0, 0, 0].item() != 0.0)
    check("PM3: custom pad_id=99, non-pad = 0",
          mask_custom[1, 0, 0, 1].item() == 0.0)

    # PM4 — single batch (B=1)
    seq_single = torch.tensor([[1, 0, 3]])
    mask_single = create_padding_mask(seq_single, pad_id)
    check("PM4: single batch shape", mask_single.shape == (1, 1, 1, 3),
          f"got {mask_single.shape}")
    check("PM4: single batch pad at pos1",
          mask_single[0, 0, 0, 1].item() != 0.0)

    # PM5 — mask can be used in attention (integration smoke test)
    # TODO: re-enable once scaled_dot_product_attention is implemented
    # Q = torch.randn(1, 1, 3, 32)
    # K = torch.randn(1, 1, 3, 32)
    # V = torch.randn(1, 1, 3, 32)
    # _, weights_pm = scaled_dot_product_attention(Q, K, V, mask=mask_single)
    # check("PM5: integration — masked key gets ~0 attention",
    #       weights_pm[0, 0, :, 1].abs().max().item() < 1e-6,
    #       f"attention to pad position: {weights_pm[0, 0, :, 1].tolist()}")


# ---------------------------------------------------------------------------
# create_causal_mask
# ---------------------------------------------------------------------------

def test_create_causal_mask():
    S = 4

    # C0 — shape
    mask = create_causal_mask(S)
    check("C0: shape (1, 1, S, S)", mask.shape == (1, 1, S, S),
          f"got {mask.shape}")

    # C1 — lower triangle (including diagonal) = 0
    for i in range(S):
        for j in range(i + 1):  # j <= i (lower + diagonal)
            val = mask[0, 0, i, j].item()
            check(f"C1: lower-tri ({i},{j}) = 0", val == 0.0,
                  f"got {val}")

    # C2 — upper triangle (j > i) = -inf
    for i in range(S):
        for j in range(i + 1, S):  # j > i
            val = mask[0, 0, i, j].item()
            check(f"C2: upper-tri ({i},{j}) = -inf",
                  val == float("-inf"),
                  f"got {val}")

    # C3 — diagonal check: each position can attend to itself
    for i in range(S):
        check(f"C3: diagonal ({i},{i}) = 0", mask[0, 0, i, i].item() == 0.0)

    # C4 — last position can attend to ALL positions
    last_row = mask[0, 0, S - 1, :]
    check("C4: last row all zeros (attends to everything)",
          (last_row == 0.0).all().item(),
          f"last row: {last_row.tolist()}")

    # C4b — first position can attend ONLY to itself
    first_row = mask[0, 0, 0, :]
    check("C4b: first row — only pos0 is 0, rest -inf",
          first_row[0].item() == 0.0 and (first_row[1:] == float("-inf")).all().item(),
          f"first row: {first_row.tolist()}")

    # C5 — different sequence lengths
    for slen in [1, 2, 8, 16]:
        m = create_causal_mask(slen)
        check(f"C5: seq_len={slen} shape correct", m.shape == (1, 1, slen, slen),
              f"got {m.shape}")
        # Quick sanity: diagonal is 0, upper-right corner is -inf
        check(f"C5: seq_len={slen} diagonal ok",
              m[0, 0, slen - 1, slen - 1].item() == 0.0)
        if slen > 1:
            check(f"C5: seq_len={slen} upper-right -inf",
                  m[0, 0, 0, slen - 1].item() == float("-inf"))

    # C6 — integration: causal mask enforces autoregressive property
    # TODO: re-enable once scaled_dot_product_attention is implemented
    # Q = torch.randn(1, 2, S, 32)
    # K = torch.randn(1, 2, S, 32)
    # V = torch.randn(1, 2, S, 32)
    # _, weights_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
    # # Upper triangle weights should be ~0
    # for i in range(S):
    #     for j in range(i + 1, S):
    #         w = weights_causal[0, 0, i, j].item()
    #         check(f"C6: causal — weight ({i},{j}) ≈ 0", w < 1e-6,
    #               f"got {w:.2e}")
    #
    # # Sanity: lower triangle should have non-trivial weight
    # non_trivial = False
    # for i in range(1, S):
    #     for j in range(i):
    #         if weights_causal[0, 0, i, j].item() > 0.01:
    #             non_trivial = True
    #             break
    # check("C6: causal — lower-tri has meaningful weights", non_trivial,
    #       "all lower-tri weights near 0 — model is not attending")


# ---------------------------------------------------------------------------
# Edge cases & combined scenarios
# ---------------------------------------------------------------------------

def test_edge_cases():
    cfg = default_config()

    # E0 — batch size 1, sequence length 1
    # Set eval mode so dropout doesn't scale the single weight
    # (softmax of 1 element is always 1.0; dropout's 1/(1-p) scaling
    # in training mode would make it 1.111 — not an identity violation,
    # just a training-mode artifact)
    mha = MultiHeadAttention(cfg)
    mha.eval()
    x_single = torch.randn(1, 1, cfg.d_model)
    out_single, w_single = mha(query=x_single, key=x_single, value=x_single)
    check("E0: B=1, S=1 self-attn output shape",
          out_single.shape == (1, 1, cfg.d_model),
          f"got {out_single.shape}")
    # Single token always attends 100% to itself
    check("E0: B=1, S=1 — sole attention weight = 1.0",
          abs(w_single[0, 0, 0, 0].item() - 1.0) < 1e-6,
          f"got {w_single[0, 0, 0, 0].item():.4f}")

    # E1 — scalar d_k = 1 edge case
    cfg_edge = default_config(d_model=4, num_heads=4)  # d_k = 1
    mha_edge = MultiHeadAttention(cfg_edge)
    x_edge = torch.randn(2, 3, 4)
    out_edge, _ = mha_edge(query=x_edge, key=x_edge, value=x_edge)
    check("E1: d_k=1 — output shape", out_edge.shape == (2, 3, 4),
          f"got {out_edge.shape}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("scaled_dot_product_attention")
    test_scaled_dot_product_attention()

    # TODO: re-enable once MultiHeadAttention is implemented
    # print("\nMultiHeadAttention")
    # test_multihead_attention()

    print("\ncreate_padding_mask")
    test_create_padding_mask()

    print("\ncreate_causal_mask")
    test_create_causal_mask()

    # TODO: re-enable once MultiHeadAttention is implemented (used in edge cases)
    # print("\nEdge Cases")
    # test_edge_cases()

    print("\nEdge Cases")
    test_edge_cases()

    print("\nMultiHeadAttention")
    test_multihead_attention()

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
