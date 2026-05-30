"""
generate.py — Autoregressive inference: greedy decoding (and optionally beam search).

Loads a trained checkpoint and generates reversed sequences character by character.
At each step, the model predicts the next token; the new token is appended to the
decoder input for the next step until <eos> is generated or max_len is reached.

Usage:
    python generate.py --checkpoint checkpoints/best.pt
    # Then enters interactive mode: type a string, get its reversal

    python generate.py --checkpoint checkpoints/best.pt --input "hello"
    # One-shot: reverse "hello" and exit
"""

import argparse
import torch
import torch.nn.functional as F

from config import TransformerConfig
from transformer import Transformer
from attention import create_causal_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns namespace with:
        --checkpoint PATH   path to a .pt checkpoint (required)
        --input STR         optional string to reverse directly (instead of interactive)
        --max_len INT       max generation length (default from config or 50)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def greedy_decode(
    model: Transformer,
    src_tokens: torch.Tensor,
    sos_token_id: int,
    eos_token_id: int,
    max_len: int,
    device: torch.device,
) -> list[int]:
    """
    Greedy autoregressive decoding.

    1. Encode source sequence once
    2. Start decoder with <sos>
    3. Loop: predict next token → append to decoder input → repeat
    4. Stop on <eos> or max_len

    Args:
        model:        trained Transformer
        src_tokens:   (1, src_seq_len)  single example
        sos_token_id: start-of-sequence token
        eos_token_id: end-of-sequence token
        max_len:      maximum number of tokens to generate
        device:       cpu or cuda

    Returns:
        list of generated token IDs (including <sos> and <eos> if generated)
    """
    raise NotImplementedError


def decode_tokens(token_ids: list[int], id_to_char: dict[int, str]) -> str:
    """
    Convert a list of token IDs back to a string, skipping special tokens.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    1. Parse args
    2. Load checkpoint
    3. Build char↔id mapping
    4. If --input provided: reverse it and exit
    5. Else: interactive loop
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
