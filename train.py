"""
train.py — Training loop for the vanilla Transformer on the reversal task.

Handles:
- Fresh start vs resume from checkpoint (explicit --resume flag)
- Teacher-forcing forward pass with cross-entropy loss
- Gradient clipping, optimizer stepping
- Periodic checkpoint saving (every epoch + best model)
- Metrics logging to CSV
- Sanity checks: shape test, overfit-on-single-batch test

Usage:
    # Fresh start (cleans checkpoints/ and logs/ directories)
    python train.py

    # Resume from a specific checkpoint
    python train.py --resume checkpoints/ckpt_epoch_005.pt
"""

import argparse
import csv
import os
import shutil
import dataclasses
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TransformerConfig
from transformer import Transformer
from data import ReversalDataset
from attention import create_causal_mask, create_padding_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns namespace with:
        --resume PATH    path to a .pt checkpoint to resume from (optional)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    config: TransformerConfig,
    epoch: int,
    step: int,
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: float,
    path: str | Path,
) -> None:
    """
    Save a complete training checkpoint to a single .pt file.

    The checkpoint dict contains everything needed to fully resume training:

        {
            'epoch':               int,                          # which epoch
            'step':                int,                          # global step count
            'model_state_dict':    model.state_dict(),           # all weights & biases
            'optimizer_state_dict': optimizer.state_dict(),     # Adam momentum & velocity
            'config':              dataclasses.asdict(config),   # all hyperparameters
            'train_losses':        list[float],                  # per-batch history
            'val_losses':          list[float],                  # per-epoch history
            'best_val_loss':       float,                        # best val loss so far
        }

    model_state_dict is needed for both inference and training resume.
    optimizer_state_dict is needed ONLY for training resume (preserves Adam momentum).
    config ensures the checkpoint is self-describing — no external config file needed.
    """
    raise NotImplementedError


def load_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Load a checkpoint and restore model (+ optionally optimizer) state.

    Restores into the provided model (in-place via load_state_dict).
    If optimizer is provided, its state is also restored (for training resume).
    If optimizer is None, only model weights are restored (for inference).

    Returns the full checkpoint dict for accessing epoch, step, losses, etc.

    Keys in the returned dict:
        epoch, step, model_state_dict, optimizer_state_dict,
        config (as plain dict), train_losses, val_losses, best_val_loss
    """
    raise NotImplementedError


def cleanup_for_fresh_start(
    checkpoint_dir: str = "checkpoints", log_dir: str = "logs"
) -> None:
    """Delete and recreate checkpoints/ and logs/ directories for a clean run."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TransformerConfig,
    device: torch.device,
) -> list[float]:
    """
    Run one training epoch. Returns list of per-batch losses.
    """
    raise NotImplementedError


def validate(
    model: Transformer,
    dataloader: DataLoader,
    config: TransformerConfig,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on validation set.
    Returns (average loss, token-level accuracy).
    """
    raise NotImplementedError


def run_sanity_checks(config: TransformerConfig):
    """
    Before full training, verify:
    1. All tensor shapes are correct (shape test)
    2. Causal mask is lower-triangular (mask test)
    3. Model can overfit a single batch of 4 examples (overfit test)
    These catch 90% of implementation bugs early.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Full training pipeline:

    1. Parse args (--resume optional)
    2. If fresh start: cleanup checkpoints/ and logs/
    3. Initialize or load model, optimizer, config
    4. Run sanity checks (if fresh start)
    5. Training loop: for each epoch → train, validate, save checkpoint
    6. Log metrics to CSV
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
