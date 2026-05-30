"""
visualize.py — Visualization utilities for training curves and attention heatmaps.

- plot_training_curves(): reads training_log.csv, plots loss & accuracy.
- plot_attention_heatmaps(): runs a single example through the model and plots
  encoder self-attention, decoder self-attention, and decoder cross-attention
  as heatmaps for every head in every layer.

For the reversal task, cross-attention should show an anti-diagonal pattern:
decoder position i attends to encoder position (seq_len - i - 1).

Usage:
    # After training
    python visualize.py --checkpoint checkpoints/best.pt --log logs/training_log.csv

    # Plot attention for a specific input
    python visualize.py --checkpoint checkpoints/best.pt --input "abcde"
"""

import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np

from config import TransformerConfig
from transformer import Transformer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns namespace with:
        --checkpoint PATH   path to a .pt checkpoint (required)
        --log PATH          path to training_log.csv (optional; skip curves if absent)
        --input STR         example string to visualize attention for (default: random)
        --output_dir DIR    directory to save plots (default: 'plots')
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(log_path: str, output_dir: str) -> None:
    """
    Read training_log.csv and produce a 3-panel figure:
        1. Training loss (per step)
        2. Validation loss (per epoch)
        3. Token-level accuracy (per epoch)

    Saves to output_dir/training_curves.png.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Attention heatmaps
# ---------------------------------------------------------------------------

def plot_attention_heatmaps(
    model: Transformer,
    src_tokens: torch.Tensor,
    id_to_char: dict[int, str],
    output_path: str,
) -> None:
    """
    Run src_tokens through the model and plot attention heatmaps.

    Generates:
        - Encoder self-attention: one row per layer, one column per head
        - Decoder self-attention: same layout (should show causal triangular pattern)
        - Decoder cross-attention: rows=decoder positions, cols=encoder positions

    For the reversal task, cross-attention should reveal an anti-diagonal pattern.

    Args:
        model:       trained Transformer (eval mode)
        src_tokens:  (1, src_seq_len) single source sequence
        id_to_char:  mapping from token ID → character for axis labels
        output_path: where to save the figure (e.g., 'plots/attention.png')
    """
    raise NotImplementedError


def _plot_attention_grid(
    attn_weights_list: list[torch.Tensor],
    num_heads: int,
    title_prefix: str,
    xticklabels: list[str],
    yticklabels: list[str],
) -> plt.Figure:
    """
    Helper: plot a grid of attention heatmaps.

    attn_weights_list: list of tensors, each (1, num_heads, seq_len_q, seq_len_k)
                       one per layer.
    Creates (num_layers × num_heads) subplots with labeled axes.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    1. Parse args
    2. Load checkpoint
    3. If --log provided: plot training curves
    4. Run attention visualization on the provided or random input
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
