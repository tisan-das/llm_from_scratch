"""
config.py — All hyperparameters for the vanilla Transformer in a single dataclass.

Usage:
    from config import TransformerConfig
    cfg = TransformerConfig()                          # defaults
    cfg = TransformerConfig(d_model=256, num_layers=4) # override
    cfg_dict = dataclasses.asdict(cfg)                  # serialize for checkpoint
    cfg = TransformerConfig(**cfg_dict)                 # restore from checkpoint
"""

from dataclasses import dataclass, asdict


@dataclass
class TransformerConfig:
    # Model dimensions
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dropout: float = 0.1

    # Vocabulary & sequences
    vocab_size: int = 30
    max_seq_len: int = 12
    pad_token_id: int = 0
    sos_token_id: int = 1
    eos_token_id: int = 2

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 20
    grad_clip: float = 1.0

    # Derived
    @property
    def d_k(self) -> int:
        """Dimension per attention head. d_model must be divisible by num_heads."""
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        return self.d_model // self.num_heads
