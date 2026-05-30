"""
data.py — Toy dataset for character-level sequence reversal.

Generates random sequences of lowercase letters, reverses them for targets,
and prepares teacher-forcing inputs (shifted right with <sos>/<eos>).

Special tokens:  <pad>=0  <sos>=1  <eos>=2  <unk>=3  a-z=4..29

Usage:
    from config import TransformerConfig
    config = TransformerConfig()

    dataset = ReversalDataset(num_samples=5000, seq_len=10, config=config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        collate_fn=dataset.collate_fn)

    for batch in loader:
        src_tokens  = batch['src_tokens']     # (B, src_len)
        tgt_input   = batch['tgt_input']      # (B, tgt_len) — teacher forcing input
        tgt_output  = batch['tgt_output']     # (B, tgt_len) — expected output (for loss)
        src_mask    = batch['src_mask']       # (B, 1, 1, src_len) or None
        tgt_mask    = batch['tgt_mask']       # (1, 1, tgt_len, tgt_len)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from config import TransformerConfig


class ReversalDataset(Dataset):
    """
    Generates random character sequences and their reversals.

    Each sample: source "a b c d" → target "d c b a"

    Teacher forcing shift:
        src:        [a,  b,  c,  d]
        tgt_input:  [<sos>,  d,  c,  b,  a]      # target shifted right
        tgt_output: [d,  c,  b,  a,  <eos>]      # target with <eos> appended
    """

    def __init__(self, num_samples: int, seq_len: int, config: TransformerConfig):
        """
        Args:
            num_samples: how many (source, target) pairs to generate
            seq_len:     number of characters per source/target sequence
            config:      TransformerConfig (for vocab, special tokens)
        """
        super().__init__()
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a dict with:
            'src_tokens':  (src_seq_len,)         source token IDs
            'tgt_input':   (tgt_seq_len,)         decoder input (teacher forcing)
            'tgt_output':  (tgt_seq_len,)         decoder target (for loss)
        where tgt_seq_len = src_seq_len + 1 (extra token for <sos>/<eos>)
        """
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch, pad_token_id: int) -> dict[str, torch.Tensor]:
        """
        Pads sequences in a batch to the same length and builds masks.

        Returns a dict with:
            'src_tokens':  (batch, src_seq_len)
            'tgt_input':   (batch, tgt_seq_len)
            'tgt_output':  (batch, tgt_seq_len)
            'src_mask':    (batch, 1, 1, src_seq_len) — padding mask for encoder
            'tgt_mask':    (1, 1, tgt_seq_len, tgt_seq_len) — causal mask for decoder
        """
        raise NotImplementedError


def generate_batch(
    batch_size: int, seq_len: int, config: TransformerConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one batch of (source, target) pairs directly — useful for quick testing
    without a full DataLoader.

    Returns:
        src: (batch_size, seq_len)
        tgt: (batch_size, seq_len) — reversed src
    """
    raise NotImplementedError
