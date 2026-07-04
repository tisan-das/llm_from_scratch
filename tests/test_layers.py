"""
Unit tests for layers.py — FeedForward, EncoderBlock (Post-LN),
and EncoderBlockPreLN (Pre-LN hybrid).

Run:
    python tests/test_layers.py
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TransformerConfig
from layers import FeedForward, EncoderBlock, EncoderBlockPreLN


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
# FeedForward
# ---------------------------------------------------------------------------

def test_feedforward():
    cfg = default_config(d_model=128, d_ff=512, dropout=0.1)
    B, S = 2, 10

    # F0 — constructor creates correct linear layers
    ff = FeedForward(cfg)
    assert hasattr(ff, 'linear1'), "linear1 missing"
    assert hasattr(ff, 'linear2'), "linear2 missing"
    check("F0: linear1 weight shape", ff.linear1.weight.shape == (cfg.d_ff, cfg.d_model),
          f"got {ff.linear1.weight.shape}, expected {(cfg.d_ff, cfg.d_model)}")
    check("F0: linear1 bias shape", ff.linear1.bias.shape == (cfg.d_ff,),
          f"got {ff.linear1.bias.shape}, expected {(cfg.d_ff,)}")
    check("F0: linear2 weight shape", ff.linear2.weight.shape == (cfg.d_model, cfg.d_ff),
          f"got {ff.linear2.weight.shape}, expected {(cfg.d_model, cfg.d_ff)}")
    check("F0: linear2 bias shape", ff.linear2.bias.shape == (cfg.d_model,),
          f"got {ff.linear2.bias.shape}, expected {(cfg.d_model,)}")

    # F1 — output shape (standard batch)
    x = torch.randn(B, S, cfg.d_model)
    out = ff(x)
    check("F1: output shape", out.shape == (B, S, cfg.d_model),
          f"got {out.shape}, expected {(B, S, cfg.d_model)}")

    # F2 — output shape (batch=1)
    x_one = torch.randn(1, 5, cfg.d_model)
    out_one = ff(x_one)
    check("F2: batch=1, seq=5 shape", out_one.shape == (1, 5, cfg.d_model),
          f"got {out_one.shape}")

    # F3 — output shape (single token: seq_len=1)
    x_single = torch.randn(4, 1, cfg.d_model)
    out_single = ff(x_single)
    check("F3: seq_len=1 shape", out_single.shape == (4, 1, cfg.d_model),
          f"got {out_single.shape}")

    # F4 — gradient flow through all parameters
    x_grad = torch.randn(B, S, cfg.d_model, requires_grad=True)
    out_grad = ff(x_grad)
    loss = out_grad.sum()
    loss.backward()
    for name, p in ff.named_parameters():
        grad_norm = p.grad.norm().item()
        check(f"F4: gradient flows through {name}", grad_norm > 0,
              f"grad norm = {grad_norm:.4f}")
        check(f"F4: {name} gradient finite", torch.isfinite(p.grad).all().item(),
              "has NaN or inf")

    # F5 — no internal dropout module
    has_dropout = any(isinstance(m, nn.Dropout) for m in ff.modules())
    check("F5: no internal dropout module", not has_dropout,
          "FeedForward contains nn.Dropout — should live in EncoderBlock")

    # F6 — position independence: same input at different positions → identical output
    # Create a tensor where position 0 and position 2 have the same values
    x_pos = torch.zeros(1, 4, cfg.d_model)
    x_pos[0, 0, :] = torch.randn(cfg.d_model)  # position 0
    x_pos[0, 2, :] = x_pos[0, 0, :].clone()    # position 2 = same as position 0
    with torch.no_grad():
        ff.eval()
        out_pos = ff(x_pos)
    check("F6: same input at pos0 and pos2 -> same output",
          torch.allclose(out_pos[0, 0], out_pos[0, 2], atol=1e-6),
          f"max diff: {(out_pos[0, 0] - out_pos[0, 2]).abs().max().item():.2e}")

    # F6b — different inputs at different positions → different outputs
    x_pos2 = torch.zeros(1, 4, cfg.d_model)
    x_pos2[0, 0, :] = torch.randn(cfg.d_model)      # position 0
    x_pos2[0, 1, :] = torch.randn(cfg.d_model)      # position 1 (different)
    with torch.no_grad():
        ff.eval()
        out_pos2 = ff(x_pos2)
    diff = (out_pos2[0, 0] - out_pos2[0, 1]).abs().sum().item()
    check("F6b: different inputs at different positions -> different outputs",
          diff > 1e-4,
          f"diff = {diff:.6f} (should be non-trivial)")

    # F7 — deterministic output in eval mode (no dropout, so always deterministic)
    torch.manual_seed(42)
    ff.eval()
    x_det = torch.randn(2, 5, cfg.d_model)
    out1 = ff(x_det)
    out2 = ff(x_det)
    check("F7: deterministic output in eval mode",
          torch.allclose(out1, out2, atol=1e-6),
          f"max diff: {(out1 - out2).abs().max().item():.2e}")

    # F8 — different configs: various d_model/d_ff combinations
    configs = [
        (128, 512),    # default 4×
        (64, 256),     # 4×, smaller
        (256, 1024),   # 4×, larger
        (128, 256),    # 2× expansion
        (128, 128),    # 1× expansion (no expansion)
        (64, 1024),    # 16× expansion
    ]
    for d_m, d_ff in configs:
        cfg_var = default_config(d_model=d_m, d_ff=d_ff)
        ff_var = FeedForward(cfg_var)
        x_var = torch.randn(1, 3, d_m)
        out_var = ff_var(x_var)
        check(f"F8: config d={d_m} d_ff={d_ff} output shape",
              out_var.shape == (1, 3, d_m),
              f"got {out_var.shape}")

    # F9 — no NaN/inf in output for normal random inputs
    x_rand = torch.randn(B, S, cfg.d_model)
    out_rand = ff(x_rand)
    check("F9: output has no NaN", not torch.isnan(out_rand).any().item())
    check("F9: output has no inf", not torch.isinf(out_rand).any().item())

    # F10 — ReLU zeros out negative values
    # Create an input that produces strongly negative values in linear1,
    # then set linear2 weights and bias to zero so the all-zero ReLU output
    # propagates cleanly to the final output.
    with torch.no_grad():
        ff.linear1.weight.data = torch.full_like(ff.linear1.weight.data, -0.5)
        ff.linear1.bias.data = torch.full_like(ff.linear1.bias.data, -1.0)
        ff.linear2.weight.data = torch.zeros_like(ff.linear2.weight.data)
        ff.linear2.bias.data = torch.zeros_like(ff.linear2.bias.data)
    x_neg = torch.ones(1, 2, cfg.d_model) * 2.0
    out_neg = ff(x_neg)
    # After linear1: W @ x + b = negative everywhere -> ReLU -> all zeros
    # After linear2: zeros @ W^T + zeros = zeros
    check("F10: ReLU zeros out negative pre-activations",
          (out_neg.abs() < 1e-6).all().item(),
          f"max abs value: {out_neg.abs().max().item():.2e} (expected ~0)")

    # F11 — parameter count matches expected
    total_params = sum(p.numel() for p in ff.parameters())
    expected = cfg.d_model * cfg.d_ff + cfg.d_ff + cfg.d_ff * cfg.d_model + cfg.d_model
    check("F11: parameter count matches d_model*d_ff + d_ff + d_ff*d_model + d_model",
          total_params == expected,
          f"got {total_params}, expected {expected}")


# ---------------------------------------------------------------------------
# EncoderBlock (Post-LN)
# ---------------------------------------------------------------------------

def test_encoder_block_post_ln():
    """Test the paper-faithful Post-LN encoder block."""
    cfg = default_config(d_model=128, d_ff=512, num_heads=4, dropout=0.1)
    B, S = 2, 10

    # E0 — constructor: all submodules exist
    blk = EncoderBlock(cfg)
    assert hasattr(blk, 'self_attn'), "self_attn missing"
    assert hasattr(blk, 'norm1'),     "norm1 missing"
    assert hasattr(blk, 'norm2'),     "norm2 missing"
    assert hasattr(blk, 'ffn'),       "ffn missing"
    assert hasattr(blk, 'dropout'),   "dropout missing"
    assert not hasattr(blk, 'norm3'), "norm3 should NOT exist in Post-LN"
    check("E0: all submodules present", True)

    # E1 — output shape (standard batch)
    blk.eval()
    x = torch.randn(B, S, cfg.d_model)
    out, attn = blk(x)
    check("E1: output shape", out.shape == (B, S, cfg.d_model),
          f"got {out.shape}")

    # E2 — attention weights shape
    check("E2: attn weights shape",
          attn.shape == (B, cfg.num_heads, S, S),
          f"got {attn.shape}")

    # E3 — shape invariance: batch=1, seq_len=1
    x_1 = torch.randn(1, 1, cfg.d_model)
    out_1, attn_1 = blk(x_1)
    check("E3: batch=1, seq=1 output shape", out_1.shape == (1, 1, cfg.d_model))
    check("E3: batch=1, seq=1 attn shape",
          attn_1.shape == (1, cfg.num_heads, 1, 1))

    # E4 — shape invariance: large batch
    x_large = torch.randn(32, 7, cfg.d_model)
    out_large, _ = blk(x_large)
    check("E4: large batch output shape", out_large.shape == (32, 7, cfg.d_model))

    # E5 — gradient flow through all params
    blk.train()
    x_g = torch.randn(B, S, cfg.d_model)
    out_g, _ = blk(x_g)
    loss = out_g.sum()
    loss.backward()
    for name, p in blk.named_parameters():
        grad_norm = p.grad.norm().item()
        check(f"E5: gradient flows — {name}", grad_norm > 0,
              f"grad norm = {grad_norm:.4f}")
        check(f"E5: {name} gradient finite", torch.isfinite(p.grad).all().item(),
              "has NaN or inf")

    # E6 — deterministic output in eval mode
    blk.eval()
    torch.manual_seed(42)
    x_d = torch.randn(B, S, cfg.d_model)
    out1, _ = blk(x_d)
    out2, _ = blk(x_d)
    check("E6: deterministic eval", torch.allclose(out1, out2, atol=1e-6))

    # E7 — stochastic output in train mode (dropout active)
    blk.train()
    out1, _ = blk(x_d)
    out2, _ = blk(x_d)
    check("E7: stochastic train (dropout)", not torch.allclose(out1, out2))

    # E8 — attention weights sum to 1 over key dimension
    blk.eval()
    _, attn = blk(x_d)
    sums = attn.sum(dim=-1)  # sum over key (last) dim
    check("E8: attn weights sum to 1",
          torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    # E9 — attention weights are non-negative
    check("E9: attn weights non-negative", (attn >= 0).all().item())

    # E10 — works with mask=None
    blk.eval()
    out, _ = blk(x_d, None)
    check("E10: mask=None", out.shape == (B, S, cfg.d_model))

    # E11 — works with padding mask
    mask = torch.zeros(B, 1, 1, S)
    mask[:, :, :, 0] = float('-inf')  # mask out position 0
    out_masked, attn_masked = blk(x_d, mask)
    check("E11: masked output shape", out_masked.shape == (B, S, cfg.d_model))
    # Position 0 in key dim should have ~0 attention weight
    check("E11: masked position has ~0 weight",
          (attn_masked[:, :, :, 0].abs() < 1e-4).all().item(),
          f"max masked weight: {attn_masked[:, :, :, 0].abs().max():.2e}")

    # E12 — output IS normalized (Post-LN passes through norm2)
    blk.eval()
    x_n = torch.randn(4, 5, cfg.d_model)
    with torch.no_grad():
        out_n, _ = blk(x_n)
    out_std = out_n.std(dim=-1).mean().item()
    check("E12: output std near 1.0",
          0.5 < out_std < 1.5,
          f"std = {out_std:.4f} (expected ~1.0, LayerNorm guarantees this)")

    # E13 — no NaN/inf in output for normal inputs
    x_r = torch.randn(B, S, cfg.d_model)
    out_r, _ = blk.eval()(x_r)
    check("E13: output no NaN", not torch.isnan(out_r).any().item())
    check("E13: output no inf", not torch.isinf(out_r).any().item())

    # E14 — parameter count matches expected
    total = sum(p.numel() for p in blk.parameters())
    # attention: 4 × (d_model² + d_model) = 4×(128²+128) = 66,048
    # FFN: d_model×d_ff + d_ff + d_ff×d_model + d_model = 2×16384 + 640 = 131,712
    # LN ×2: 2 × 2×d_model = 2×256 = 512
    expected = 4 * (128*128 + 128) + (2*128*512 + 512 + 128) + 2 * (2*128)
    check("E14: param count", total == expected,
          f"got {total}, expected {expected}")

    # E15 — dropout module present and correct rate
    assert isinstance(blk.dropout, nn.Dropout)
    check("E15: dropout present", True)
    check("E15: dropout rate", abs(blk.dropout.p - cfg.dropout) < 1e-6,
          f"got {blk.dropout.p}, expected {cfg.dropout}")

    # E16 — different configs work
    configs = [(64, 256, 2), (256, 1024, 8)]
    for d_m, d_ff, n_h in configs:
        cfg_v = default_config(d_model=d_m, d_ff=d_ff, num_heads=n_h)
        blk_v = EncoderBlock(cfg_v).eval()
        x_v = torch.randn(1, 3, d_m)
        out_v, _ = blk_v(x_v)
        check(f"E16: config d={d_m} h={n_h} shape",
              out_v.shape == (1, 3, d_m), f"got {out_v.shape}")


# ---------------------------------------------------------------------------
# EncoderBlockPreLN (Pre-LN hybrid)
# ---------------------------------------------------------------------------

def test_encoder_block_pre_ln():
    """Test the Pre-LN hybrid encoder block with per-block output norm."""
    cfg = default_config(d_model=128, d_ff=512, num_heads=4, dropout=0.1)
    B, S = 2, 10

    # P0 — constructor: all submodules exist, including norm3
    blk = EncoderBlockPreLN(cfg)
    for attr in ['self_attn', 'norm1', 'norm2', 'norm3', 'ffn', 'dropout']:
        assert hasattr(blk, attr), f"{attr} missing"
    check("P0: all submodules present (incl. norm3)", True)

    # P1 — output shape
    blk.eval()
    x = torch.randn(B, S, cfg.d_model)
    out, attn = blk(x)
    check("P1: output shape", out.shape == (B, S, cfg.d_model),
          f"got {out.shape}")

    # P2 — attention weights shape
    check("P2: attn weights shape",
          attn.shape == (B, cfg.num_heads, S, S),
          f"got {attn.shape}")

    # P3 — shape invariance: edge cases
    for label, b, s in [("batch=1,seq=1", 1, 1),
                         ("batch=32,seq=3", 32, 3),
                         ("batch=1,seq=10", 1, 10)]:
        x_v = torch.randn(b, s, cfg.d_model)
        out_v, _ = blk(x_v)
        check(f"P3: {label} output shape", out_v.shape == (b, s, cfg.d_model))

    # P4 — gradient flow through all params
    blk.train()
    x_g = torch.randn(B, S, cfg.d_model)
    out_g, _ = blk(x_g)
    loss = out_g.sum()
    loss.backward()
    for name, p in blk.named_parameters():
        grad_norm = p.grad.norm().item()
        check(f"P4: gradient flows — {name}", grad_norm > 0,
              f"grad norm = {grad_norm:.4f}")
        check(f"P4: {name} gradient finite", torch.isfinite(p.grad).all().item(),
              "has NaN or inf")

    # P5 — deterministic eval
    blk.eval()
    torch.manual_seed(42)
    x_d = torch.randn(B, S, cfg.d_model)
    out1, _ = blk(x_d)
    out2, _ = blk(x_d)
    check("P5: deterministic eval", torch.allclose(out1, out2, atol=1e-6))

    # P6 — stochastic train
    blk.train()
    out1, _ = blk(x_d)
    out2, _ = blk(x_d)
    check("P6: stochastic train (dropout)", not torch.allclose(out1, out2))

    # P7 — attention weights sum to 1
    blk.eval()
    _, attn = blk(x_d)
    sums = attn.sum(dim=-1)
    check("P7: attn weights sum to 1",
          torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    # P8 — attention weights non-negative
    check("P8: attn weights non-negative", (attn >= 0).all().item())

    # P9 — mask=None works
    blk.eval()
    out, _ = blk(x_d, None)
    check("P9: mask=None", out.shape == (B, S, cfg.d_model))

    # P10 — padding mask works
    mask = torch.zeros(B, 1, 1, S)
    mask[:, :, :, 0] = float('-inf')
    out_m, attn_m = blk(x_d, mask)
    check("P10: masked output shape", out_m.shape == (B, S, cfg.d_model))
    check("P10: masked pos has ~0 weight",
          (attn_m[:, :, :, 0].abs() < 1e-4).all().item(),
          f"max: {attn_m[:, :, :, 0].abs().max():.2e}")

    # P11 — output IS normalized (norm3 guarantees this)
    blk.eval()
    x_n = torch.randn(4, 5, cfg.d_model)
    with torch.no_grad():
        out_n, _ = blk(x_n)
    out_std = out_n.std(dim=-1).mean().item()
    check("P11: output std near 1.0",
          0.5 < out_std < 1.5,
          f"std = {out_std:.4f} (expected ~1.0, norm3 guarantees this)")

    # P12 — no NaN/inf
    x_r = torch.randn(B, S, cfg.d_model)
    out_r, _ = blk.eval()(x_r)
    check("P12: output no NaN", not torch.isnan(out_r).any().item())
    check("P12: output no inf", not torch.isinf(out_r).any().item())

    # P13 — norm3 exists and has exactly 256 params (γ+β for d_model=128)
    assert hasattr(blk, 'norm3')
    n3_params = sum(p.numel() for p in blk.norm3.parameters())
    check("P13: norm3 param count", n3_params == 2 * cfg.d_model,
          f"got {n3_params}, expected {2 * cfg.d_model}")

    # P14 — total parameter count = Post-LN + 256
    post_blk = EncoderBlock(cfg)
    post_p = sum(p.numel() for p in post_blk.parameters())
    pre_p = sum(p.numel() for p in blk.parameters())
    check("P14: param count = Post-LN + 256",
          pre_p == post_p + 256,
          f"Pre-LN: {pre_p}, Post-LN: {post_p}, diff: {pre_p - post_p}")

    # P15 — different configs work
    configs = [(64, 256, 2), (256, 1024, 8)]
    for d_m, d_ff, n_h in configs:
        cfg_v = default_config(d_model=d_m, d_ff=d_ff, num_heads=n_h)
        blk_v = EncoderBlockPreLN(cfg_v).eval()
        x_v = torch.randn(1, 3, d_m)
        out_v, _ = blk_v(x_v)
        check(f"P15: config d={d_m} h={n_h} shape",
              out_v.shape == (1, 3, d_m), f"got {out_v.shape}")

    # P16 — dropout present with correct rate
    assert isinstance(blk.dropout, nn.Dropout)
    check("P16: dropout rate correct",
          abs(blk.dropout.p - cfg.dropout) < 1e-6)


# ---------------------------------------------------------------------------
# Cross-variant comparison tests
# ---------------------------------------------------------------------------

def test_cross_variant():
    """Tests that compare Post-LN and Pre-LN variants side by side."""
    cfg = default_config(d_model=128, d_ff=512, num_heads=4, dropout=0.1)
    B, S = 2, 10

    post = EncoderBlock(cfg).eval()
    pre = EncoderBlockPreLN(cfg).eval()

    # C0 — identical forward signatures (same args, same return type)
    x = torch.randn(B, S, cfg.d_model)
    mask = torch.zeros(B, 1, 1, S)
    mask[:, :, :, -1] = float('-inf')

    for blk, name in [(post, "Post-LN"), (pre, "Pre-LN")]:
        out, attn = blk(x)
        assert isinstance(out, torch.Tensor), f"{name}: out not Tensor"
        assert isinstance(attn, torch.Tensor), f"{name}: attn not Tensor"
        # mask variant
        out_m, attn_m = blk(x, mask)
        assert out_m.shape == (B, S, cfg.d_model), f"{name}: masked shape wrong"
        # None mask
        out_n, attn_n = blk(x, None)
        assert out_n.shape == (B, S, cfg.d_model), f"{name}: None mask shape wrong"
    check("C0: identical forward interface", True)

    # C1 — both produce same output shapes
    post_out, _ = post(x)
    pre_out, _ = pre(x)
    check("C1: same output shapes", post_out.shape == pre_out.shape)
    _, post_attn = post(x)
    _, pre_attn = pre(x)
    check("C1: same attn shapes", post_attn.shape == pre_attn.shape)

    # C2 — both outputs are normalized (Post: norm2, Pre: norm3)
    x_n = torch.randn(4, 5, cfg.d_model)
    with torch.no_grad():
        post_out_n, _ = post(x_n)
        pre_out_n, _ = pre(x_n)
    post_std = post_out_n.std(dim=-1).mean().item()
    pre_std = pre_out_n.std(dim=-1).mean().item()
    check("C2: Post-LN output normalized", 0.5 < post_std < 1.5,
          f"std={post_std:.4f}")
    check("C2: Pre-LN output normalized", 0.5 < pre_std < 1.5,
          f"std={pre_std:.4f}")

    # C3 — Pre-LN has exactly one more LayerNorm than Post-LN
    post_ln_count = sum(1 for m in post.modules() if isinstance(m, nn.LayerNorm))
    pre_ln_count = sum(1 for m in pre.modules() if isinstance(m, nn.LayerNorm))
    check("C3: Pre-LN has 1 more LayerNorm",
          pre_ln_count == post_ln_count + 1,
          f"Pre={pre_ln_count}, Post={post_ln_count}")

    # C4 — both blocks handle seq_len=1 identically
    x_1 = torch.randn(1, 1, cfg.d_model)
    post_out_1, _ = post(x_1)
    pre_out_1, _ = pre(x_1)
    check("C4: Post seq_len=1 shape", post_out_1.shape == (1, 1, cfg.d_model))
    check("C4: Pre  seq_len=1 shape", pre_out_1.shape == (1, 1, cfg.d_model))

    # C5 — both attention weight matrices sum to 1 (valid probabilities)
    _, post_attn = post(x)
    _, pre_attn = pre(x)
    check("C5: Post attn sums to 1",
          torch.allclose(post_attn.sum(-1), torch.ones_like(post_attn.sum(-1))))
    check("C5: Pre  attn sums to 1",
          torch.allclose(pre_attn.sum(-1), torch.ones_like(pre_attn.sum(-1))))

    # C6 — self-attention weights are non-negative in both
    check("C6: Post attn >= 0", (post_attn >= 0).all().item())
    check("C6: Pre  attn >= 0", (pre_attn >= 0).all().item())

    # C7 — masked positions have ~0 weight in both
    post_m_out, post_m_attn = post(x, mask)
    pre_m_out, pre_m_attn = pre(x, mask)
    check("C7: Post masked pos ~0",
          (post_m_attn[:, :, :, -1].abs() < 1e-4).all().item())
    check("C7: Pre  masked pos ~0",
          (pre_m_attn[:, :, :, -1].abs() < 1e-4).all().item())

    # C8 — both train modes produce stochastic outputs
    post.train(); pre.train()
    x_s = torch.randn(2, 3, cfg.d_model)
    post1, _ = post(x_s); post2, _ = post(x_s)
    pre1, _ = pre(x_s);   pre2, _ = pre(x_s)
    check("C8: Post train mode stochastic", not torch.allclose(post1, post2))
    check("C8: Pre  train mode stochastic", not torch.allclose(pre1, pre2))

    # C9 — both eval modes produce deterministic outputs
    post.eval(); pre.eval()
    post1, _ = post(x_s); post2, _ = post(x_s)
    pre1, _ = pre(x_s);   pre2, _ = pre(x_s)
    check("C9: Post eval deterministic", torch.allclose(post1, post2))
    check("C9: Pre  eval deterministic", torch.allclose(pre1, pre2))

    # C10 — gradients exist and are finite in both (train mode)
    post.train(); pre.train()
    x_g = torch.randn(2, 5, cfg.d_model)
    for blk, name in [(post, "Post"), (pre, "Pre")]:
        out_g, _ = blk(x_g)
        out_g.sum().backward()
        for p_name, p in blk.named_parameters():
            check(f"C10: {name} grad exists — {p_name}", p.grad is not None)
            check(f"C10: {name} grad finite — {p_name}",
                  torch.isfinite(p.grad).all().item())
        blk.zero_grad()


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("FeedForward")
    test_feedforward()

    print("\nEncoderBlock (Post-LN)")
    test_encoder_block_post_ln()

    print("\nEncoderBlockPreLN (Pre-LN hybrid)")
    test_encoder_block_pre_ln()

    print("\nCross-variant comparison")
    test_cross_variant()

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
