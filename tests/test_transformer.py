"""
Unit tests for transformer.py — Encoder (Post-LN and Pre-LN hybrid).

Covers:
    - Constructor & submodule presence (default Post-LN, Pre-LN hybrid, final_norm)
    - Forward pass shapes (standard, edge cases: seq_len=1, batch=1)
    - Attention map collection (per-layer count, shapes, need_weights=False)
    - Output normalization (both Post-LN and Pre-LN variants self-normalize)
    - Masking (mask=None, padding mask, all-zero mask)
    - Gradient flow through all parameters (both variants)
    - GPT-2 scaled init (correct factor, opt-out, seeded verification)
    - Determinism (eval mode) and stochasticity (train mode with dropout)
    - extra_repr() metadata
    - Cross-variant consistency (shapes, attention maps, parameter deltas)
    - Edge cases: num_layers=1, large d_model, large batch

Run:
    python tests/test_transformer.py
"""

import math
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TransformerConfig
from layers import EncoderBlock, EncoderBlockPreLN
from transformer import Encoder


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


# ===================================================================
# TESTS — Constructor & Initialization
# ===================================================================

def test_constructor():
    """Verify submodule presence, layer counts, and config wiring."""
    cfg = default_config(num_encoder_layers=3, d_model=128, dropout=0.1)

    # ---- E0: Default (Post-LN) submodules ----
    enc = Encoder(cfg)
    assert hasattr(enc, 'token_embedding'), "token_embedding missing"
    assert hasattr(enc, 'position_encoding'), "position_encoding missing"
    assert hasattr(enc, 'embed_dropout'), "embed_dropout missing"
    assert hasattr(enc, 'layers'), "layers (ModuleList) missing"
    check("E0: token_embedding", True)
    check("E0: position_encoding", True)
    check("E0: embed_dropout", True)
    check("E0: layers", True)

    # ---- E1: Layer count matches config ----
    check("E1: num_layers matches config",
          len(enc.layers) == cfg.num_encoder_layers,
          f"got {len(enc.layers)}, expected {cfg.num_encoder_layers}")

    # ---- E2: Each layer is the correct block class (default: EncoderBlock) ----
    assert all(isinstance(blk, EncoderBlock) for blk in enc.layers), \
        "default block_cls should be EncoderBlock"
    check("E2: default blocks are EncoderBlock (Post-LN)", True)

    # ---- E3: final_norm=False (default) → None ----
    check("E3: final_norm=None by default", enc.final_norm is None)

    # ---- E4: final_norm=True → LayerNorm present ----
    enc_fn = Encoder(cfg, final_norm=True)
    assert enc_fn.final_norm is not None, "final_norm should exist"
    assert isinstance(enc_fn.final_norm, nn.LayerNorm), "should be LayerNorm"
    check("E4: final_norm=True adds LayerNorm", True)
    check("E4: final_norm is nn.LayerNorm",
          isinstance(enc_fn.final_norm, nn.LayerNorm))

    # ---- E5: block_cls=EncoderBlockPreLN uses Pre-LN blocks ----
    enc_pre = Encoder(cfg, block_cls=EncoderBlockPreLN)
    assert all(isinstance(blk, EncoderBlockPreLN) for blk in enc_pre.layers), \
        "should all be EncoderBlockPreLN"
    check("E5: block_cls=EncoderBlockPreLN", True)

    # ---- E6: embed_dropout has correct probability ----
    check("E6: embed_dropout rate",
          abs(enc.embed_dropout.p - cfg.dropout) < 1e-6,
          f"got {enc.embed_dropout.p}, expected {cfg.dropout}")


# ===================================================================
# TESTS — Forward Pass: Shapes
# ===================================================================

def test_forward_shapes():
    """Verify output shapes for standard and edge-case inputs."""
    cfg = default_config(num_encoder_layers=3, d_model=128)
    B, S = 4, 10

    # ---- E10: Standard Post-LN forward shapes ----
    enc = Encoder(cfg).eval()
    src = torch.randint(0, cfg.vocab_size, (B, S))
    out, maps = enc(src)
    check("E10: output shape", out.shape == (B, S, cfg.d_model),
          f"got {out.shape}")
    check("E10: attn maps count", len(maps) == cfg.num_encoder_layers,
          f"got {len(maps)}")
    check("E10: attn map shape",
          maps[0].shape == (B, cfg.num_heads, S, S),
          f"got {maps[0].shape}")

    # ---- E11: Pre-LN hybrid forward shapes ----
    enc_pre = Encoder(cfg, block_cls=EncoderBlockPreLN).eval()
    out_pre, maps_pre = enc_pre(src)
    check("E11: Pre-LN output shape", out_pre.shape == (B, S, cfg.d_model))
    check("E11: Pre-LN attn maps count",
          len(maps_pre) == cfg.num_encoder_layers)

    # ---- E12: seq_len=1 ----
    src_1 = torch.randint(0, cfg.vocab_size, (2, 1))
    out_1, maps_1 = enc(src_1)
    check("E12: seq_len=1 output", out_1.shape == (2, 1, cfg.d_model))
    check("E12: seq_len=1 attn shape",
          maps_1[0].shape == (2, cfg.num_heads, 1, 1))

    # ---- E13: batch=1 ----
    src_b1 = torch.randint(0, cfg.vocab_size, (1, 7))
    out_b1, maps_b1 = enc(src_b1)
    check("E13: batch=1 output", out_b1.shape == (1, 7, cfg.d_model))
    check("E13: batch=1 attn shape",
          maps_b1[0].shape == (1, cfg.num_heads, 7, 7))

    # ---- E14: Large batch ----
    src_large = torch.randint(0, cfg.vocab_size, (32, 5))
    out_large, _ = enc(src_large)
    check("E14: large batch output", out_large.shape == (32, 5, cfg.d_model))

    # ---- E15: need_weights=False returns None ----
    out_nw, maps_nw = enc(src, need_weights=False)
    check("E15: output shape (no weights)", out_nw.shape == (B, S, cfg.d_model))
    check("E15: maps is None when need_weights=False", maps_nw is None)

    # ---- E16: need_weights=False also works for Pre-LN ----
    out_nw2, maps_nw2 = enc_pre(src, need_weights=False)
    check("E16: Pre-LN need_weights=False maps", maps_nw2 is None)


# ===================================================================
# TESTS — Forward Pass: Normalization
# ===================================================================

def test_forward_normalization():
    """Verify that encoder outputs are normalized (std ~1.0 per token)."""
    cfg = default_config(num_encoder_layers=3, d_model=128)
    B, S = 4, 10

    src = torch.randint(0, cfg.vocab_size, (B, S))

    # ---- E20: Post-LN output is normalized (norm2 in each block) ----
    enc = Encoder(cfg).eval()
    with torch.no_grad():
        out, _ = enc(src)
    out_std = out.std(dim=-1).mean().item()
    check("E20: Post-LN output std ~1.0",
          0.5 < out_std < 1.5,
          f"std = {out_std:.4f} (expected ~1.0)")

    # ---- E21: Pre-LN hybrid output is normalized (norm3 in each block) ----
    enc_pre = Encoder(cfg, block_cls=EncoderBlockPreLN).eval()
    with torch.no_grad():
        out_pre, _ = enc_pre(src)
    out_pre_std = out_pre.std(dim=-1).mean().item()
    check("E21: Pre-LN output std ~1.0",
          0.5 < out_pre_std < 1.5,
          f"std = {out_pre_std:.4f} (expected ~1.0)")

    # ---- E22: final_norm=True output is normalized ----
    enc_fn = Encoder(cfg, final_norm=True).eval()
    with torch.no_grad():
        out_fn, _ = enc_fn(src)
    out_fn_std = out_fn.std(dim=-1).mean().item()
    check("E22: final_norm=True output std ~1.0",
          0.5 < out_fn_std < 1.5,
          f"std = {out_fn_std:.4f} (expected ~1.0)")

    # ---- E23: Output has no NaN or inf ----
    for name, e in [("Post-LN", enc), ("Pre-LN", enc_pre)]:
        with torch.no_grad():
            o, _ = e(src)
        check(f"E23: {name} output no NaN", not torch.isnan(o).any().item())
        check(f"E23: {name} output no inf", not torch.isinf(o).any().item())


# ===================================================================
# TESTS — Forward Pass: Masking
# ===================================================================

def test_forward_masking():
    """Verify mask=None, padding mask, and all-zero mask behaviors."""
    cfg = default_config(num_encoder_layers=3, d_model=128)
    B, S = 4, 10

    # ---- E30: mask=None works (both variants) ----
    for name, enc in [
        ("Post-LN", Encoder(cfg).eval()),
        ("Pre-LN", Encoder(cfg, block_cls=EncoderBlockPreLN).eval()),
    ]:
        src = torch.randint(0, cfg.vocab_size, (B, S))
        out, maps = enc(src, None)
        check(f"E30: {name} mask=None output", out.shape == (B, S, cfg.d_model))
        check(f"E30: {name} mask=None attn", len(maps) == cfg.num_encoder_layers)

    # ---- E31: Padding mask works — masked position gets ~0 attention ----
    enc = Encoder(cfg).eval()
    src = torch.randint(1, cfg.vocab_size, (B, S))  # no <pad> tokens to start
    # Create a padding mask that masks position 0
    src_mask = torch.zeros(B, 1, 1, S)
    src_mask[:, :, :, 0] = float('-inf')
    _, maps = enc(src, src_mask)
    # Position 0 in the KEY dimension should have ~0 attention weight
    # (softmax of -inf = 0, so it shouldn't contribute)
    check("E31: masked position has ~0 weight",
          (maps[-1][:, :, :, 0].abs() < 1e-4).all().item(),
          f"max masked weight: {maps[-1][:, :, :, 0].abs().max():.2e}")

    # ---- E32: All-zero mask (attends to everything) = same as mask=None ----
    enc2 = Encoder(cfg).eval()
    zero_mask = torch.zeros(B, 1, 1, S)
    src2 = torch.randint(0, cfg.vocab_size, (B, S))
    torch.manual_seed(42)
    out_zero, _ = enc2(src2, zero_mask)
    torch.manual_seed(42)
    out_none, _ = enc2(src2, None)
    check("E32: zero_mask == no_mask",
          torch.allclose(out_zero, out_none, atol=1e-5),
          f"max diff: {(out_zero - out_none).abs().max():.2e}")


# ===================================================================
# TESTS — Gradient Flow
# ===================================================================

def test_gradient_flow():
    """Verify gradients propagate through every parameter."""
    cfg = default_config(num_encoder_layers=3, d_model=128)
    B, S = 4, 10

    # ---- E40: Post-LN gradients ----
    enc = Encoder(cfg).train()
    src = torch.randint(0, cfg.vocab_size, (B, S))
    out, _ = enc(src)
    loss = out.sum()
    loss.backward()
    for name, p in enc.named_parameters():
        check(f"E40: grad exists — {name}", p.grad is not None)
        check(f"E40: grad finite — {name}",
              torch.isfinite(p.grad).all().item(),
              "NaN or inf")
    enc.zero_grad()

    # ---- E41: Pre-LN hybrid gradients ----
    enc_pre = Encoder(cfg, block_cls=EncoderBlockPreLN).train()
    out_pre, _ = enc_pre(src)
    out_pre.sum().backward()
    for name, p in enc_pre.named_parameters():
        check(f"E41: grad exists — {name}", p.grad is not None)
        check(f"E41: grad finite — {name}",
              torch.isfinite(p.grad).all().item(),
              "NaN or inf")
    enc_pre.zero_grad()

    # ---- E42: Gradients also flow with final_norm=True ----
    enc_fn = Encoder(cfg, final_norm=True).train()
    out_fn, _ = enc_fn(src)
    out_fn.sum().backward()
    for name, p in enc_fn.named_parameters():
        check(f"E42: grad exists — {name}", p.grad is not None)
        check(f"E42: grad finite — {name}",
              torch.isfinite(p.grad).all().item(),
              "NaN or inf")


# ===================================================================
# TESTS — Scaled Initialization
# ===================================================================

def test_scaled_init():
    """Verify GPT-2 style 1/sqrt(2*num_layers) residual scaling."""
    cfg = default_config(num_encoder_layers=3, d_model=128)
    N = cfg.num_encoder_layers
    scale = 1.0 / math.sqrt(2 * N)

    # ---- E50: Scaled and unscaled weights differ ----
    # Seed both the same so init is identical apart from the scaling
    torch.manual_seed(42)
    enc_s = Encoder(cfg, scaled_init=True)
    torch.manual_seed(42)
    enc_ns = Encoder(cfg, scaled_init=False)

    # Attention output projection (W_o) — should differ by exactly `scale`
    w_s = enc_s.layers[0].self_attn.W_o.weight
    w_ns = enc_ns.layers[0].self_attn.W_o.weight
    # Element-wise ratio should be exactly `scale` for every element
    ratio = (w_s / w_ns).mean().item()
    check("E50: W_o ratio matches 1/sqrt(2N)",
          abs(ratio - scale) < 1e-6,
          f"ratio={ratio:.6f}, expected={scale:.6f}")
    check("E50: W_o element-wise ratio is uniform",
          (w_s / w_ns).std().item() < 1e-6,
          f"std of ratio: {(w_s/w_ns).std():.2e}")

    # ---- E51: FFN linear2 also scaled ----
    ff_s = enc_s.layers[0].ffn.linear2.weight
    ff_ns = enc_ns.layers[0].ffn.linear2.weight
    ratio_ff = (ff_s / ff_ns).mean().item()
    check("E51: FFN linear2 ratio matches 1/sqrt(2N)",
          abs(ratio_ff - scale) < 1e-6,
          f"ratio={ratio_ff:.6f}, expected={scale:.6f}")

    # ---- E52: Scaling is applied to ALL layers ----
    for i in range(N):
        w_s_i = enc_s.layers[i].self_attn.W_o.weight
        w_ns_i = enc_ns.layers[i].self_attn.W_o.weight
        ratio_i = (w_s_i / w_ns_i).mean().item()
        check(f"E52: layer {i} W_o scaled",
              abs(ratio_i - scale) < 1e-6,
              f"ratio={ratio_i:.6f}")

    # ---- E53: Scaled init is opt-out-able ----
    enc_default = Encoder(cfg)  # scaled_init=True by default
    enc_optout = Encoder(cfg, scaled_init=False)
    w_default = enc_default.layers[0].self_attn.W_o.weight.norm()
    w_optout = enc_optout.layers[0].self_attn.W_o.weight.norm()
    # These should differ significantly (factor ~scale)
    ratio_default = w_default / (w_optout + 1e-8)
    # Note: different random seeds, so just check that they're not identical
    check("E53: scaled_init=False produces different weights",
          not torch.allclose(
              enc_default.layers[0].self_attn.W_o.weight,
              enc_optout.layers[0].self_attn.W_o.weight
          ))


# ===================================================================
# TESTS — Determinism & Stochasticity
# ===================================================================

def test_determinism():
    """Verify eval is deterministic; train produces stochastic outputs."""
    cfg = default_config(num_encoder_layers=3, d_model=128, dropout=0.1)
    B, S = 4, 10
    src = torch.randint(0, cfg.vocab_size, (B, S))

    # ---- E60: Deterministic in eval mode (both variants) ----
    for name, enc_cls in [
        ("Post-LN", EncoderBlock),
        ("Pre-LN", EncoderBlockPreLN),
    ]:
        enc = Encoder(cfg, block_cls=enc_cls).eval()
        torch.manual_seed(42)
        out1, _ = enc(src)
        torch.manual_seed(42)
        out2, _ = enc(src)
        check(f"E60: {name} deterministic eval",
              torch.allclose(out1, out2, atol=1e-6),
              f"max diff: {(out1-out2).abs().max():.2e}")

    # ---- E61: Stochastic in train mode (embed_dropout active) ----
    for name, enc_cls in [
        ("Post-LN", EncoderBlock),
        ("Pre-LN", EncoderBlockPreLN),
    ]:
        enc = Encoder(cfg, block_cls=enc_cls).train()
        out1, _ = enc(src)
        out2, _ = enc(src)
        check(f"E61: {name} stochastic train",
              not torch.allclose(out1, out2),
              "outputs identical — dropout may not be active")


# ===================================================================
# TESTS — extra_repr
# ===================================================================

def test_extra_repr():
    """Verify extra_repr() surfaces key metadata."""
    cfg = default_config(num_encoder_layers=3, d_model=128)

    # ---- E70: Contains num_layers and d_model ----
    enc = Encoder(cfg)
    rep = enc.extra_repr()
    check("E70: contains num_layers", "num_layers=3" in rep)
    check("E70: contains d_model", "d_model=128" in rep)

    # ---- E71: Reflects final_norm state ----
    enc_fn = Encoder(cfg, final_norm=True)
    rep_fn = enc_fn.extra_repr()
    check("E71: reflects final_norm=True", "final_norm=True" in rep_fn)

    enc_nofn = Encoder(cfg, final_norm=False)
    rep_nofn = enc_nofn.extra_repr()
    check("E71: reflects final_norm=False", "final_norm=False" in rep_nofn)


# ===================================================================
# TESTS — Cross-Variant Consistency
# ===================================================================

def test_cross_variant():
    """Compare Post-LN and Pre-LN encoders side-by-side."""
    cfg = default_config(num_encoder_layers=3, d_model=128, dropout=0.1)
    B, S = 4, 10

    post = Encoder(cfg).eval()
    pre = Encoder(cfg, block_cls=EncoderBlockPreLN).eval()
    src = torch.randint(0, cfg.vocab_size, (B, S))

    # ---- E80: Same output shapes ----
    out_p, maps_p = post(src)
    out_r, maps_r = pre(src)
    check("E80: same output shape", out_p.shape == out_r.shape)
    check("E80: same attn maps count", len(maps_p) == len(maps_r))

    # ---- E81: Same attention map shapes ----
    for i in range(cfg.num_encoder_layers):
        check(f"E81: layer {i} attn shape match",
              maps_p[i].shape == maps_r[i].shape)

    # ---- E82: Pre-LN has more params (norm3 per block) ----
    post_p = sum(p.numel() for p in post.parameters())
    pre_p = sum(p.numel() for p in pre.parameters())
    expected_diff = cfg.num_encoder_layers * 2 * cfg.d_model  # 3 × 256 = 768
    check("E82: Pre-LN has more params than Post-LN",
          pre_p == post_p + expected_diff,
          f"Post={post_p}, Pre={pre_p}, diff={pre_p-post_p}, expected {expected_diff}")

    # ---- E83: Both produce valid attention probabilities ----
    for name, maps in [("Post-LN", maps_p), ("Pre-LN", maps_r)]:
        for i, attn in enumerate(maps):
            sums = attn.sum(dim=-1)
            check(f"E83: {name} layer {i} attn sums to 1",
                  torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
            check(f"E83: {name} layer {i} attn non-negative",
                  (attn >= 0).all().item())


# ===================================================================
# TESTS — Edge Cases
# ===================================================================

def test_edge_cases():
    """Boundary and unusual config tests."""
    cfg = default_config(d_model=128)

    # ---- E90: num_layers=1 works ----
    cfg1 = default_config(num_encoder_layers=1)
    enc1 = Encoder(cfg1).eval()
    src = torch.randint(0, cfg1.vocab_size, (2, 5))
    out1, maps1 = enc1(src)
    check("E90: num_layers=1 output", out1.shape == (2, 5, cfg1.d_model))
    check("E90: num_layers=1 maps count", len(maps1) == 1)

    # ---- E91: Large d_model (256) works ----
    cfg_large = default_config(num_encoder_layers=2, d_model=256,
                                num_heads=4, d_ff=1024)
    enc_large = Encoder(cfg_large).eval()
    src_large = torch.randint(0, cfg_large.vocab_size, (2, 6))
    out_l, maps_l = enc_large(src_large)
    check("E91: d_model=256 output shape",
          out_l.shape == (2, 6, 256))
    check("E91: d_model=256 attn shape",
          maps_l[0].shape == (2, 4, 6, 6))

    # ---- E92: Pre-LN hybrid with num_layers=1 ----
    enc_p1 = Encoder(cfg1, block_cls=EncoderBlockPreLN).eval()
    out_p1, maps_p1 = enc_p1(src)
    check("E92: Pre-LN num_layers=1 output", out_p1.shape == (2, 5, cfg1.d_model))
    check("E92: Pre-LN num_layers=1 maps", len(maps_p1) == 1)
    # Output should still be normalized (norm3 per block)
    out_std = out_p1.std(dim=-1).mean().item()
    check("E92: Pre-LN num_layers=1 output normalized",
          0.5 < out_std < 1.5, f"std={out_std:.4f}")

    # ---- E93: Very long sequence (relative to max_seq_len) ----
    cfg_long = default_config(max_seq_len=20)
    enc_long = Encoder(cfg_long).eval()
    src_long = torch.randint(0, cfg_long.vocab_size, (1, 15))
    out_long, _ = enc_long(src_long)
    check("E93: long sequence output", out_long.shape == (1, 15, cfg_long.d_model))


# ===================================================================
# Run all
# ===================================================================

if __name__ == "__main__":
    print("Constructor & Initialization")
    test_constructor()

    print("\nForward Pass — Shapes")
    test_forward_shapes()

    print("\nForward Pass — Normalization")
    test_forward_normalization()

    print("\nForward Pass — Masking")
    test_forward_masking()

    print("\nGradient Flow")
    test_gradient_flow()

    print("\nScaled Init")
    test_scaled_init()

    print("\nDeterminism & Stochasticity")
    test_determinism()

    print("\nextra_repr")
    test_extra_repr()

    print("\nCross-Variant Consistency")
    test_cross_variant()

    print("\nEdge Cases")
    test_edge_cases()

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
