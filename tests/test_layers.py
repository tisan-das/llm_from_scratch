"""
Unit tests for layers.py — FeedForward.

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
from layers import FeedForward


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
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("FeedForward")
    test_feedforward()

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
