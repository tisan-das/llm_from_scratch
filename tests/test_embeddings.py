"""
Unit tests for embeddings.py — TokenEmbedding and LearnedPositionalEncoding.

Run:
    python tests/test_embeddings.py
"""

import torch
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TransformerConfig
from embeddings import TokenEmbedding, LearnedPositionalEncoding


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
# TokenEmbedding
# ---------------------------------------------------------------------------

def test_token_embedding():
    cfg = TransformerConfig()
    tok = TokenEmbedding(cfg)

    # T0 — weight init matches N(0, 1/√d)
    actual_std = tok.embedding.weight.std().item()
    expected_std = 1.0 / (cfg.d_model ** 0.5)
    check("T0: weight init std = 1/sqrt(d)",
          abs(actual_std - expected_std) < 0.02,
          f"expected {expected_std:.4f}, got {actual_std:.4f}")

    # T1 — output shape
    x = torch.randint(0, cfg.vocab_size, (4, 10))
    out = tok(x)
    expected = (4, 10, cfg.d_model)
    check("T1: output shape", out.shape == expected,
          f"got {out.shape}, expected {expected}")

    # T2 — scaling: forward output = raw_embedding × √d
    single = torch.tensor([[5]])
    unscaled = tok.embedding(single).detach()       # (1, 1, d)  — pre-scale
    scaled = tok(single).detach()                    # (1, 1, d)  — post-scale
    scale_factor = cfg.d_model ** 0.5
    actual_ratio = (scaled.abs().mean() / unscaled.abs().mean()).item()
    check("T2: output = raw x sqrt(d)",
          abs(actual_ratio - scale_factor) < 1e-4,
          f"expected {scale_factor:.4f}, got {actual_ratio:.4f}")

    # T2b — after scaling, a token row has L2 norm ≈ √d (~11.3 for d=128)
    tok_row = tok.embedding.weight[5].detach() * tok.scale
    tok_norm = tok_row.norm().item()
    expected_norm = cfg.d_model ** 0.5
    check("T2b: scaled row L2 norm ~ sqrt(d)",
          abs(tok_norm - expected_norm) < 0.5 * expected_norm,
          f"expected ~{expected_norm:.1f}, got {tok_norm:.1f}")

    # T2c — token/pos parity: ratio ≈ 1.0 (both should be ~√d)
    # Learned positions use default N(0,1) init → row norm ≈ √d.
    # If we later switch to sinusoidal, this ratio shifts to ~√2 ≈ 1.4.
    # Either way it stays under 2.5 (the old broken init gave ratio ~12).
    pos = LearnedPositionalEncoding(cfg)
    pos_row = pos.position_embedding.weight[3].detach()
    tok_norm_scaled = tok.embedding.weight[5].detach().norm().item() * tok.scale
    pos_norm = pos_row.norm().item()
    ratio = tok_norm_scaled / (pos_norm + 1e-8)
    check("T2c: token/pos norm parity (ratio ~ 1.0)",
          0.5 < ratio < 2.5,
          f"ratio = {ratio:.2f} (target ~1.0, old broken init gave ~12)")

    # T3 — full vocab range (no crash)
    try:
        _ = tok(torch.tensor([[0, 29, 4, 15]]))
        check("T3: full vocab range [0,29]", True)
    except Exception as e:
        check("T3: full vocab range [0,29]", False, str(e))

    # T4 — gradient flow through embedding weight
    out = tok(torch.randint(0, cfg.vocab_size, (2, 3)))
    loss = out.sum()
    loss.backward()
    grad_norm = tok.embedding.weight.grad.norm().item()
    check("T4: gradients flow", grad_norm > 0,
          f"grad norm = {grad_norm:.4f}")


# ---------------------------------------------------------------------------
# LearnedPositionalEncoding
# ---------------------------------------------------------------------------

def test_positional_encoding():
    cfg = TransformerConfig()
    pos = LearnedPositionalEncoding(cfg)

    # P0 — default N(0,1) init → row L2 norm ≈ √d
    pos_row = pos.position_embedding.weight[3].detach()
    pos_norm = pos_row.norm().item()
    expected_norm = cfg.d_model ** 0.5
    check("P0: position row L2 norm ~ sqrt(d)",
          abs(pos_norm - expected_norm) < 0.5 * expected_norm,
          f"expected ~{expected_norm:.1f}, got {pos_norm:.1f}")

    # P1 — output shape (broadcast-ready: 1, S, d)
    x = torch.randint(0, 10, (4, 10))
    out = pos(x)
    expected = (1, 10, cfg.d_model)
    check("P1: output shape (1, S, d) for broadcast", out.shape == expected,
          f"got {out.shape}, expected {expected}")

    # P2 — same positions = same vectors regardless of batch content
    out1 = pos(torch.randint(0, 10, (4, 10)))
    out2 = pos(torch.randint(0, 10, (4, 10)))
    check("P2: batch invariance",
          torch.allclose(out1, out2),
          "positions differ when they should be identical")

    # P3 — different positions learn different vectors
    out_all = pos(torch.randn(1, 10))
    diff = (out_all[0, 0] - out_all[0, 5]).abs().sum().item()
    check("P3: pos 0 != pos 5", diff > 1e-6,
          f"diff = {diff:.6f}")

    # P4 — respects max_seq_len bound (clear error, not raw IndexError)
    try:
        _ = pos(torch.randn(1, cfg.max_seq_len))
        check(f"P4: at max_seq_len={cfg.max_seq_len}", True)
    except Exception as e:
        check(f"P4: at max_seq_len={cfg.max_seq_len}", False, str(e))

    try:
        _ = pos(torch.randn(1, cfg.max_seq_len + 1))
        check("P4: beyond max_seq_len raises ValueError", False,
              "should have raised ValueError")
    except ValueError as e:
        msg = str(e)
        ok = ("exceeds" in msg and "max_seq_len" in msg)
        check("P4: beyond max_seq_len raises ValueError", ok,
              f"message: {msg[:60]}...")

    # P5 — gradient flow through position embedding weight
    out = pos(torch.randn(2, 3))
    loss = out.sum()
    loss.backward()
    grad_norm = pos.position_embedding.weight.grad.norm().item()
    check("P5: gradients flow", grad_norm > 0,
          f"grad norm = {grad_norm:.4f}")


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_integration():
    cfg = TransformerConfig()
    tok = TokenEmbedding(cfg)
    pos = LearnedPositionalEncoding(cfg)

    # I1 — token + position broadcast-adds correctly
    tok_emb = tok(torch.tensor([[4, 5, 6, 7]]))       # (1, 4, d)
    pos_emb = pos(torch.randn(1, 4))                   # (1, 4, d)
    combined = tok_emb + pos_emb                        # (1, 4, d)
    expected = (1, 4, cfg.d_model)
    check("I1: token + pos broadcast shape", combined.shape == expected,
          f"got {combined.shape}, expected {expected}")

    # I2 — same positions, different tokens produce different combined vectors.
    # The position contribution is identical, so the difference is purely
    # from the token embeddings — confirming neither component zeroes out.
    base = pos(torch.randn(1, 2))                      # (1, 2, d)
    combined_a = tok(torch.tensor([[4, 29]])) + base    # 'a' at pos0, 'z' at pos1
    diff = (combined_a[0, 0] - combined_a[0, 1]).abs().sum().item()
    check("I2: different tokens -> different output",
          diff > 1e-2,
          f"diff = {diff:.6f} (should be large — different chars)")

    # I3 — same token at different positions produces different output
    # (because positional encoding differs per position).
    combined_b = tok(torch.tensor([[4, 4]])) + base     # 'a' at pos0 AND pos1
    diff_pos = (combined_b[0, 0] - combined_b[0, 1]).abs().sum().item()
    check("I3: same token, different pos -> different output",
          diff_pos > 1e-2,
          f"diff = {diff_pos:.6f} (should be large — different positions)")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("TokenEmbedding")
    test_token_embedding()

    print("\nLearnedPositionalEncoding")
    test_positional_encoding()

    print("\nIntegration")
    test_integration()

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
