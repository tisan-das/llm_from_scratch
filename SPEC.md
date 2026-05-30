# SPEC: Vanilla Transformer from Scratch

## 1. Goal

Implement the original Transformer architecture as described in **"Attention Is All You Need"** (Vaswani et al., 2017) — a full encoder-decoder stack — from scratch using PyTorch. The focus is on understanding the architecture deeply, not on raw matrix multiplications. Every component is implemented explicitly so that each equation from the paper maps to a visible block of code.

A tiny toy task (character-level sequence reversal) is used to verify correctness end-to-end. The model trains on CPU in minutes.

**Why this exists:**
- Grok the full encoder-decoder transformer (not just decoder-only like GPT)
- Establish a clean, hackable codebase for experimenting with architectural variants from recent research
- Build visualization tools (attention heatmaps) to develop intuition

---

## 2. Toy Task: Character-Level Sequence Reversal

| Property | Value |
|----------|-------|
| Task | Reverse a character sequence |
| Example input | `"a b c d"` |
| Example target | `"d c b a"` |
| Vocabulary | Lowercase a-z + `<pad>`, `<sos>`, `<eos>`, `<unk>` |
| Vocab size | 30 |
| Sequence length | Configurable (default 10) |
| Data | All possible sequences of length `seq_len` over `vocab` (or random sampling if combinatorially large) |
| Source | Randomly generated, no real dataset needed |
| Metrics | Accuracy (exact match), token-level accuracy, cross-entropy loss |

### Why Reversal?

- Non-trivial — the decoder MUST use cross-attention to look at the right encoder positions (cannot solve without attending to source)
- Tiny vocabulary, tiny sequences → trains fast on CPU
- Attention heatmaps are interpretable (decoder should point at mirrored encoder positions)
- Proves the full encoder-decoder stack works correctly

### Special Tokens

| Token | ID | Meaning |
|-------|----|---------|
| `<pad>` | 0 | Padding (not used in fixed-length setup initially, but slots reserved) |
| `<sos>` | 1 | Start of sequence |
| `<eos>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown character |
| `a-z` | 4-29 | Actual characters |

---

## 3. Architecture Overview

```
                  SOURCE: "a b c d"                     TARGET: "<sos> d c b a"
                       |                                       |
              [Input Embedding]                        [Input Embedding]
              [Positional Encoding]                    [Positional Encoding]
                       |                                       |
              ╔═════════════════╗                      ╔═════════════════╗
              ║   ENCODER x N   ║                      ║   DECODER x N   ║
              ║                 ║                      ║                 ║
              ║  ┌───────────────────┐                ║  ┌───────────────────┐
              ║  │ Multi-Head        │                ║  │ Masked Multi-Head │
              ║  │ Self-Attention   │                ║  │ Self-Attention    │
              ║  └───────────────────┘                ║  └───────────────────┘
              ║  ┌───────────────────┐                ║  ┌───────────────────┐
              ║  │ Add & Norm       │                ║  │ Add & Norm        │
              ║  └───────────────────┘                ║  └───────────────────┘
              ║  ┌───────────────────┐                ║  ┌───────────────────┐
              ║  │ Feed Forward     │                ║  │ Cross-Attention   │ ← attends to encoder output
              ║  └───────────────────┘                ║  └───────────────────┘
              ║  ┌───────────────────┐                ║  ┌───────────────────┐
              ║  │ Add & Norm       │                ║  │ Add & Norm        │
              ║  └───────────────────┘                ║  └───────────────────┘
              ║                 │                     ║  ┌───────────────────┐
              ╚═════════════════╝                     ║  │ Feed Forward     │
                       |                              ║  └───────────────────┘
                encoder_output                        ║  ┌───────────────────┐
                       |                              ║  │ Add & Norm        │
                       └──────────────┐               ║  └───────────────────┘
                                      │               ║                 │
                                      │               ╚═════════════════╝
                                      │                        |
                                      └────→ [Cross-Attention] |
                                                            |
                                                   [Linear + Softmax]
                                                            |
                                                    predicted tokens
```

### Hyperparameters (Default — Tiny)

| Parameter | Value | Paper value | Rationale |
|-----------|-------|-------------|-----------|
| `d_model` | 128 | 512 | Small enough for CPU; big enough to have multiple heads |
| `num_heads` | 4 | 8 | `d_model` must be divisible by `num_heads` (128/4=32 per head) |
| `d_ff` | 512 | 2048 | 4× multiplier as per paper |
| `num_layers` | 3 | 6 | 3 encoder + 3 decoder layers is plenty for reversal |
| `dropout` | 0.1 | 0.1 | Same as paper |
| `vocab_size` | 30 | varies | 26 chars + 4 special tokens |
| `max_seq_len` | 12 | varies | Enough for 10 chars + special tokens |
| `batch_size` | 64 | varies | Small enough for CPU memory |
| `learning_rate` | 1e-4 | varies | Adam optimizer, will tune |
| `epochs` | 20 | varies | Reversal is an easy task |

---

## 4. File Structure

```
llm_from_scratch/
├── SPEC.md                  # This document
├── README.md                # Quick start, architecture diagram (Mermaid), usage
├── config.py                # TransformerConfig dataclass — all hyperparameters
├── embeddings.py            # TokenEmbedding + LearnedPositionalEncoding
├── attention.py             # scaled_dot_product_attention + MultiHeadAttention
├── layers.py                # FeedForward, EncoderBlock, DecoderBlock
├── transformer.py           # Encoder, Decoder, Transformer (top-level)
├── data.py                  # ReversalDataset, generate_data, tokenization, collate_fn
├── train.py                 # Training loop, checkpointing, logging
├── generate.py              # Greedy (and optionally beam) decoding
├── visualize.py             # Attention heatmaps, training curves
├── checkpoints/             # Saved model checkpoints (.pt files)
├── logs/                    # Training logs (CSV)
└── plots/                   # Saved figures (loss curves, attention heatmaps)
```

### Why This Split?

- **`attention.py` is isolated**: If you want to try Grouped-Query Attention or Flash Attention later, you only touch this file.
- **`embeddings.py` is isolated**: Swap to RoPE, ALiBi, or sinusoidal by changing this file.
- **`layers.py` is isolated**: Experiment with Pre-LN vs Post-LN, SwiGLU, parallel attention/FFN here.
- **`transformer.py` is thin**: Just stacks blocks. Changing depth/width is `config.py`.
- **`data.py` is isolated**: Swap the toy task (reverse → sort → tiny translation) without touching model code.
- **`visualize.py` is standalone**: Use on any checkpoint to inspect attention patterns.

---

## 5. Component Specifications

### 5.1 `config.py` — `TransformerConfig`

A `@dataclass` holding every hyperparameter. Saved alongside checkpoints for reproducibility.

```python
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

    # Derived (computed in __post_init__)
    d_k: int = d_model // num_heads   # 32 — dimension per head
```

### 5.2 `embeddings.py`

#### TokenEmbedding
- Wraps `nn.Embedding(vocab_size, d_model)` with proper scaling.
- The paper multiplies embeddings by `√d_model`. We follow this.

#### LearnedPositionalEncoding
- `nn.Embedding(max_seq_len, d_model)` — learns a vector per position (0, 1, 2, ...).
- At forward time, adds positional embedding to token embedding element-wise.
- Shape: `(batch, seq_len, d_model)` token embeddings + `(1, seq_len, d_model)` position embeddings → broadcast add.

**Note:** The paper uses fixed sinusoidal positional encoding. We use learned for simplicity. The interface is identical — easy to swap later.

### 5.3 `attention.py`

#### Function: `scaled_dot_product_attention(Q, K, V, mask=None)`

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k + mask) @ V
```

| Argument | Shape | Meaning |
|----------|-------|---------|
| `Q` | `(batch, num_heads, seq_len_q, d_k)` | Queries |
| `K` | `(batch, num_heads, seq_len_k, d_k)` | Keys |
| `V` | `(batch, num_heads, seq_len_v, d_k)` | Values (`seq_len_k == seq_len_v`) |
| `mask` | `(batch, 1, seq_len_q, seq_len_k)` or `(1, 1, seq_len_q, seq_len_k)` | `0` = attend, `-inf` = mask out |
| **Returns** | `(batch, num_heads, seq_len_q, d_k)` | Weighted sum of values + attention weights |

**Implementation details:**
- `scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)` — shape `(batch, heads, seq_q, seq_k)`
- If mask provided: `scores = scores + mask` — `-inf` positions become `-inf` in scores
- `attn_weights = F.softmax(scores, dim=-1)` — exp(-inf) = 0, so masked positions get zero weight
- Optionally apply dropout to attention weights (paper does)
- `output = attn_weights @ V`
- Return both `output` and `attn_weights` (the weights are needed for visualization)

#### Class: `MultiHeadAttention`

| Component | Shape transformation |
|-----------|---------------------|
| Input | `(batch, seq_len, d_model)` |
| `W_q, W_k, W_v` | Each `nn.Linear(d_model, d_model)`, no bias (paper uses bias, many implementations don't — we'll use bias for faithfulness) |
| Project to Q, K, V | `(batch, seq_len, d_model)` → `(batch, seq_len, d_model)` |
| Split into heads | `(batch, seq_len, d_model)` → `(batch, num_heads, seq_len, d_k)` via `.view()` and `.transpose()` |
| Scaled dot-product attention | `(batch, num_heads, seq_len, d_k)` → `(batch, num_heads, seq_len, d_k)` |
| Concatenate heads | Reverse the split: `(batch, num_heads, seq_len, d_k)` → `(batch, seq_len, d_model)` |
| `W_o` projection | `nn.Linear(d_model, d_model)` |
| Output | `(batch, seq_len, d_model)` + attention weights |

**Constructor signature:**
```python
MultiHeadAttention(d_model: int, num_heads: int, dropout: float)
```

**Forward signature:**
```python
forward(query, key, value, mask=None) -> (output, attn_weights)
```
- For **self-attention**: `query == key == value` (same tensor passed three times)
- For **cross-attention**: `query` from decoder, `key` and `value` from encoder output

#### Mask Construction Utilities

Two helper functions (could live in `attention.py` or `data.py`):

1. **`create_padding_mask(seq, pad_token_id)`** → `(batch, 1, 1, seq_len)` — `True` where token is pad, used to mask keys in attention
2. **`create_causal_mask(seq_len)`** → `(1, 1, seq_len, seq_len)` — upper triangular `-inf`, lower triangular `0`

### 5.4 `layers.py`

#### `FeedForward`

```
FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
```

| Component | Shape |
|-----------|-------|
| Input | `(batch, seq_len, d_model)` |
| `linear1` | `nn.Linear(d_model, d_ff)` |
| Activation | `F.relu` (paper uses ReLU; modern LLMs use GELU/SwiGLU — easy swap point) |
| `linear2` | `nn.Linear(d_ff, d_model)` |
| Dropout | After second linear, before residual add |
| Output | `(batch, seq_len, d_model)` |

#### `EncoderBlock`

```
def forward(self, x, src_mask=None):
    # 1. Multi-head self-attention
    attn_out, attn_weights = self.self_attn(x, x, x, src_mask)
    x = self.norm1(x + self.dropout(attn_out))   # Add & Norm

    # 2. Feed-forward
    ff_out = self.ffn(x)
    x = self.norm2(x + self.dropout(ff_out))     # Add & Norm

    return x, attn_weights
```

**Contains:**
- `self_attn`: `MultiHeadAttention`
- `norm1`, `norm2`: `nn.LayerNorm(d_model)` — **Post-LN** as per paper (norm AFTER residual add)
- `ffn`: `FeedForward`
- `dropout`: `nn.Dropout`

**Note on Post-LN vs Pre-LN:**
- Paper (Post-LN): `LayerNorm(x + Sublayer(x))` ← we implement this
- Modern (Pre-LN): `x + Sublayer(LayerNorm(x))` ← GPT/LLaMA use this
- We follow the paper for faithfulness; easy to experiment later

#### `DecoderBlock`

```
def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
    # 1. Masked multi-head self-attention
    attn_out_1, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
    x = self.norm1(x + self.dropout(attn_out_1))

    # 2. Cross-attention to encoder
    attn_out_2, cross_attn_weights = self.cross_attn(x, encoder_output, encoder_output, src_mask)
    x = self.norm2(x + self.dropout(attn_out_2))

    # 3. Feed-forward
    ff_out = self.ffn(x)
    x = self.norm3(x + self.dropout(ff_out))

    return x, self_attn_weights, cross_attn_weights
```

**Contains:**
- `self_attn`: `MultiHeadAttention` (masked)
- `cross_attn`: `MultiHeadAttention` (Q from decoder, K,V from encoder)
- `norm1`, `norm2`, `norm3`: `nn.LayerNorm(d_model)`
- `ffn`: `FeedForward`
- Returns both attention weight sets for visualization

### 5.5 `transformer.py`

#### `Encoder`

```python
class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = LearnedPositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.num_encoder_layers)
        ])

    def forward(self, src_tokens, src_mask=None):
        # Embed + position
        x = self.token_embedding(src_tokens)
        x = x + self.position_encoding(src_tokens)  # broadcast add
        x = self.dropout(x)

        # Pass through each encoder block
        all_self_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            all_self_attn_weights.append(attn_weights)

        return x, all_self_attn_weights  # (batch, seq_len, d_model), list of attention weights
```

Shape: `src_tokens (batch, src_seq_len)` → `output (batch, src_seq_len, d_model)`

#### `Decoder`

```python
class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = LearnedPositionalEncoding(config)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.num_decoder_layers)
        ])

    def forward(self, tgt_tokens, encoder_output, src_mask=None, tgt_mask=None):
        x = self.token_embedding(tgt_tokens)
        x = x + self.position_encoding(tgt_tokens)
        x = self.dropout(x)

        all_self_attn_weights = []
        all_cross_attn_weights = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            all_self_attn_weights.append(self_attn)
            all_cross_attn_weights.append(cross_attn)

        return x, all_self_attn_weights, all_cross_attn_weights
```

Shape: `tgt_tokens (batch, tgt_seq_len)` → `output (batch, tgt_seq_len, d_model)`

#### `Transformer` (Top-Level)

```python
class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Weight tying: share embedding weights between encoder, decoder, and lm_head
        # (Optional — GPT does this; the original paper doesn't. We'll make it configurable.)

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        encoder_output, enc_self_attn = self.encoder(src_tokens, src_mask)
        decoder_output, dec_self_attn, cross_attn = self.decoder(
            tgt_tokens, encoder_output, src_mask, tgt_mask
        )
        logits = self.lm_head(decoder_output)  # (batch, tgt_seq_len, vocab_size)
        return logits, enc_self_attn, dec_self_attn, cross_attn
```

Shapes:
| Variable | Shape |
|----------|-------|
| `src_tokens` | `(batch, src_seq_len)` |
| `tgt_tokens` | `(batch, tgt_seq_len)` |
| `logits` | `(batch, tgt_seq_len, vocab_size)` |

### 5.6 `data.py`

#### `ReversalDataset`

- Generates all possible sequences or random samples
- For seq_len=10 and vocab_size=26, there are 26^10 possible sequences (too many). We randomly sample N examples per epoch.
- Stores source sequences and target sequences (reversed)

#### Data Flow for Training (Teacher Forcing)

```
Source:       ['a', 'b', 'c', 'd']
Target:       ['d', 'c', 'b', 'a']

Encoder input:    ['a', 'b', 'c', 'd']
Decoder input:    ['<sos>', 'd', 'c', 'b', 'a']     # target shifted right, <sos> prepended
Decoder target:   ['d', 'c', 'b', 'a', '<eos>']     # target with <eos> appended

Loss: CrossEntropy(predicted, decoder_target) over the tgt_seq_len positions
```

#### `generate_batch(batch_size, seq_len, vocab_chars)`

Returns:
- `src`: `(batch, seq_len)` — random character sequences
- `tgt`: `(batch, seq_len)` — reversed `src`

#### `collate_fn` or inside `__getitem__`

Converts raw characters to token IDs, shifts for teacher forcing, returns dict of tensors.

### 5.7 `train.py`

#### Training Loop (per epoch)

```
for batch in dataloader:
    src_tokens = batch.src                  # (batch, src_len)
    tgt_input = batch.tgt_input            # (batch, tgt_len) — with <sos>
    tgt_output = batch.tgt_output          # (batch, tgt_len) — with <eos>

    # Create masks
    src_mask = create_padding_mask(src_tokens, pad_id)    # None if no padding
    tgt_mask = create_causal_mask(tgt_len)                 # upper triangular -inf

    # Forward
    logits, _, _, _ = model(src_tokens, tgt_input, src_mask, tgt_mask)

    # Loss: CrossEntropy over vocab at each target position
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),    # (batch * tgt_len, vocab_size)
        tgt_output.view(-1),            # (batch * tgt_len)
        ignore_index=pad_token_id       # Don't compute loss on <pad>
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
```

#### Checkpointing

**Save format** (single `.pt` file):
```python
checkpoint = {
    'epoch': epoch,
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': dataclasses.asdict(config),    # Full config for reproducibility
    'train_losses': train_losses,            # List of per-batch losses
    'val_losses': val_losses,                # List of per-epoch val losses
    'best_val_loss': best_val_loss,
}
torch.save(checkpoint, f'checkpoints/ckpt_epoch_{epoch:03d}.pt')
```

**Save strategy:**
- Save every N steps (configurable, default: every epoch)
- Always save best model (by validation loss): `checkpoints/best.pt`

**Checkpoint loading (explicit only):**
- Checkpoints are loaded **only** when an explicit CLI argument is provided. There is no auto-detection or auto-resume.
- Usage: `python train.py --resume checkpoints/ckpt_epoch_005.pt`
- If `--resume` is provided, the script loads the checkpoint and continues training from that epoch/step. All metrics (train_losses, val_losses, best_val_loss) are restored. The optimizer state (Adam moments) is also restored so momentum is not lost.
- If `--resume` is **not** provided, it is treated as a fresh start.

**Fresh start cleanup:**
- On a fresh start (no `--resume`), the entire `checkpoints/` directory is deleted and recreated empty.
- This guarantees no stale checkpoints from previous runs accumulate and avoids confusion.
- The `logs/` directory is also cleaned on fresh start (old training CSV is removed).

**CLI summary:**
```bash
# Fresh start (wipes checkpoints/ and logs/)
python train.py

# Resume from a specific checkpoint
python train.py --resume checkpoints/ckpt_epoch_005.pt
```

#### Logging

- Print: epoch, batch, loss, perplexity, accuracy
- Write to CSV: `logs/training_log.csv` with columns `[epoch, step, loss, val_loss, accuracy, timestamp]`
- `visualize.py` reads this CSV to plot curves

### 5.8 `generate.py`

#### Greedy Decoding

```
def greedy_decode(model, src_tokens, max_len, sos_id, eos_id):
    encoder_output, _ = model.encoder(src_tokens, src_mask=None)

    # Start with <sos> token
    generated = [sos_id]

    for step in range(max_len):
        tgt_tokens = torch.tensor([generated]).to(device)   # (1, current_len)

        # Causal mask for current sequence length
        tgt_mask = create_causal_mask(len(generated))

        decoder_output, _, _ = model.decoder(tgt_tokens, encoder_output, tgt_mask=tgt_mask)

        # Get logits for last position only (more efficient)
        logits = model.lm_head(decoder_output[:, -1, :])    # (1, vocab_size)
        next_token = torch.argmax(logits, dim=-1).item()

        generated.append(next_token)
        if next_token == eos_id:
            break

    return generated
```

#### (Optional) Beam Search

Add later — greedy is sufficient to verify correctness for the reverse task.

### 5.9 `visualize.py`

#### Training Curves
- Reads `logs/training_log.csv`
- Subplots: training loss (per step), validation loss (per epoch), accuracy (per epoch)
- Saves to `plots/training_curves.png`

#### Attention Heatmaps
- Takes a single example through the model with `model.eval()` and `torch.no_grad()`
- Captures attention weights from all layers
- Plots:
  - **Encoder self-attention**: Grid of `(num_layers × num_heads)` heatmaps for source sequence
  - **Decoder self-attention**: Same for decoder sequence (should show causal triangular pattern)
  - **Decoder cross-attention**: Grid of `(num_layers × num_heads)` heatmaps — rows=decoder positions, cols=encoder positions
- For the reverse task, cross-attention should show an **anti-diagonal** pattern: decoder position i attends to encoder position `seq_len - i - 1`
- Saves to `plots/attention_heatmaps.png`

#### Architecture Diagram
- A Mermaid diagram in `README.md` showing the full data flow
- Can generate via `torchview` for a shape-annotated graph (optional)

---

## 6. Tensor Shape Reference

Here's the complete shape flow for a single batch. Assume:

- `B = batch_size = 64`
- `S = src_seq_len = 10`
- `T = tgt_seq_len = 11` (10 chars + `<sos>`)
- `V = vocab_size = 30`
- `H = num_heads = 4`
- `D = d_model = 128`
- `K = d_k = 32`

| Step | Input Shape | Output Shape | Operation |
|------|-------------|--------------|-----------|
| Token embed (encoder) | `(B, S)` | `(B, S, D)` | `nn.Embedding(V, D)` |
| Position encode (encoder) | `(B, S)` | `(B, S, D)` | `nn.Embedding(max_len, D)` |
| Add | `(B, S, D)` + `(B, S, D)` | `(B, S, D)` | Element-wise |
| Encoder block × N | `(B, S, D)` | `(B, S, D)` | — |
| └─ Self-attn: Q, K, V projections | `(B, S, D)` each | `(B, S, D)` each | `nn.Linear(D, D)` |
| └─ Reshape to heads | `(B, S, D)` | `(B, H, S, K)` | `.view(B, S, H, K).transpose(1, 2)` |
| └─ Attention scores | `(B, H, S, K)` | `(B, H, S, S)` | `Q @ K^T / √K` |
| └─ Mask (none for encoder) | — | — | — |
| └─ Softmax | `(B, H, S, S)` | `(B, H, S, S)` | Row-wise along last dim |
| └─ Weighted values | `(B, H, S, S)` @ `(B, H, S, K)` | `(B, H, S, K)` | Matrix multiply |
| └─ Merge heads | `(B, H, S, K)` | `(B, S, D)` | `.transpose(1,2).reshape(B, S, D)` |
| └─ W_o projection | `(B, S, D)` | `(B, S, D)` | `nn.Linear(D, D)` |
| └─ FFN | `(B, S, D)` | `(B, S, D)` | `ReLU(Linear(D, D_ff)) → Linear(D_ff, D)` |
| Token embed (decoder) | `(B, T)` | `(B, T, D)` | `nn.Embedding(V, D)` |
| Position encode (decoder) | `(B, T)` | `(B, T, D)` | `nn.Embedding(max_len, D)` |
| Decoder self-attn: scores | `(B, H, T, T)` | `(B, H, T, T)` | QK^T with causal mask |
| Decoder cross-attn: Q from decoder, K/V from encoder | Q: `(B, H, T, K)` vs K: `(B, H, S, K)` | scores: `(B, H, T, S)` | Q @ K^T |
| Decoder cross-attn: values | weights @ V where V: `(B, H, S, K)` | `(B, H, T, K)` | — |
| LM head | `(B, T, D)` | `(B, T, V)` | `nn.Linear(D, V)` |
| Loss: cross-entropy | `(B*T, V)` vs `(B*T,)` | scalar | `F.cross_entropy` |

---

## 7. Verification & Sanity Checks

Before full training, these checks catch 90% of bugs:

1. **Shape test**: Pass random tensors through each component, verify output shapes against the reference table above.
2. **Causal mask test**: Verify that for a 3-token sequence, the attention weights matrix is lower-triangular (upper positions are ~0).
3. **Overfitting test**: Train on a single batch of 4 examples. The model should memorize them (loss → 0, accuracy → 100%) within a few hundred steps. If it doesn't, something is broken.
4. **Gradient flow test**: Check that gradients are non-zero and don't explode/nan-out on the first backward pass.
5. **Identity test**: If source = target (copy task, not reverse), the model should learn faster. Try this as a warmup.

---

## 8. Expected Outcomes

After training on the reversal task:
- **Train accuracy** (exact sequence match): >95% within a few epochs
- **Attention patterns**: Cross-attention shows anti-diagonal (decoder position `i` → encoder position `S - i - 1`)
- **Training time**: <5 minutes on modern CPU for the tiny config

---

## 9. Future Experimentation Roadmap

Once the vanilla implementation is working, here are paper-aligned experiments to try:

| Experiment | What to change | File to touch |
|------------|---------------|---------------|
| Pre-LN | Move LayerNorm before sublayer | `layers.py` |
| Sinusoidal Positional Encoding | Use fixed sine/cosine instead of learned | `embeddings.py` |
| GELU / SwiGLU activation | Replace ReLU in FFN | `layers.py` |
| Weight tying | Share embedding weights across encoder/decoder/lm_head | `transformer.py` |
| Grouped-Query Attention (GQA) | Fewer KV heads than Q heads | `attention.py` |
| Rotary Position Embedding (RoPE) | Apply rotation to Q,K before attention | `attention.py` + `embeddings.py` |
| Flash Attention (triton) | Fused attention kernel | `attention.py` |
| Parallel attention + FFN | No sequential dependency in block | `layers.py` |
| Encoder-only (BERT-like) | Remove decoder, add masked LM head | `transformer.py` (new file) |
| Decoder-only (GPT-like) | Remove encoder, use causal self-attn only | `transformer.py` (new file) |
| Different toy tasks | Sort, copy, tiny translation | `data.py` |
| Real data | Replace character dataset with real text | `data.py` (new file) |

---

## 10. Dependencies

```
torch >= 1.12
numpy
matplotlib
tqdm
```

No heavy dependencies. Everything runs on CPU.

---

## 11. Git Workflow

- `main` branch: stable, working implementation
- Feature branches for each experiment (e.g., `exp/pre-ln`, `exp/rope`)
- Commit often, with meaningful messages
- `.gitignore`: `checkpoints/`, `logs/`, `plots/`, `__pycache__/`, `.venv/`, `*.pyc`

---

*Last updated: 2026-05-30*
