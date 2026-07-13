# Qwen2.5-Coder-3B-Instruct — From Scratch

> A full from-scratch PyTorch implementation of **Qwen2.5-Coder-3B-Instruct**, featuring a custom GPU-accelerated Flash Attention kernel written in [Triton](https://github.com/triton-lang/triton).

---

## Overview

This project re-implements the [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) model architecture entirely from scratch in PyTorch, without relying on the HuggingFace `transformers` modelling code for the core forward pass. The highlight of this implementation is a **custom Triton Flash Attention kernel** that replaces PyTorch's native attention backend for the prefill phase.

### Key Highlights

- 🧠 **From-scratch model implementation** — `Qwen2ForCausalLM`, `Qwen2Model`, `Qwen2Attention`, `Qwen2MLP`, `Qwen2RMSNorm`, and `Qwen2RotaryEmbedding` all implemented manually.
- ⚡ **Custom Triton Flash Attention** — A hand-written forward and backward Triton kernel implementing the Flash Attention 2 algorithm with:
  - Tiled SRAM-based computation for O(1) memory w.r.t. sequence length
  - Support for **causal (masked) attention** and non-causal attention
  - **Grouped Query Attention (GQA)** support (16 Q heads, 2 KV heads)
  - `bfloat16` / `float16` / `float32` dtype awareness (no hardcoded casts)
  - Boundary-checked tiled loads/stores for arbitrary sequence lengths
  - Auto-tuning via `@triton.autotune`
- 🔁 **SDPA fallback for decoding** — During token generation (where Q length = 1), the implementation automatically falls back to PyTorch's `F.scaled_dot_product_attention` with GQA broadcasting, since the Triton kernel is optimised for the prefill phase.
- ✅ **Drop-in compatible** with HuggingFace's `AutoTokenizer`, `GenerationMixin`, and `.generate()`.

---

## Architecture

The Qwen2.5-Coder-3B-Instruct model has the following configuration:

| Parameter | Value |
|---|---|
| Hidden size | 2048 |
| Layers | 36 |
| Attention heads (Q) | 16 |
| Attention heads (KV) | 2 (GQA) |
| Head dimension | 128 |
| Intermediate size | 11008 |
| Vocabulary size | 151,936 |
| Max position embeddings | 32,768 |
| Activation | SiLU |
| Normalization | RMSNorm (ε = 1e-6) |
| Position encoding | RoPE |
| Attention | Flash Attention 2 (custom Triton) |

---

## Project Structure

```
Qwen2.5/
├── src/
│   ├── __init__.py
│   ├── configuration_qwen2.py          # Qwen2Config — model hyperparameters
│   ├── modeling_qwen2.py               # Full model: Qwen2ForCausalLM, Qwen2Model, Qwen2Attention, etc.
│   ├── flash_attention.py              # Attention router: Triton kernel (prefill) + SDPA (decode)
│   └── modeling_flash_attention_utils.py  # HF-compatible flash attention utilities
│
├── flash_attention_triton.py           # Custom Triton Flash Attention forward + backward kernel
│
├── examples/
│   └── inference.py                    # End-to-end inference script
│
├── model/
│   ├── config.json                     # Model config (loaded via AutoConfig)
│   ├── tokenizer_config.json
│   ├── *.safetensors                   # Pre-trained weights (not committed)
│   └── model.safetensors.index.json
│
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## Custom Triton Flash Attention

The kernel in [`flash_attention_triton.py`](flash_attention_triton.py) implements Flash Attention 2 end-to-end in Triton.

### Forward Pass (`_attn_fwd`)

- Tiles the Q, K, V matrices into SRAM blocks of shape `(BLOCK_SIZE_Q, HEAD_DIM)` and `(HEAD_DIM, BLOCK_SIZE_KV)`.
- Maintains running softmax statistics `m_i` (max) and `l_i` (sum) per query row — no materialising the full `N×N` attention matrix.
- Supports **causal masking** via a two-stage loop (Stage 1: non-causal blocks, Stage 2: diagonal/causal block).
- Handles **GQA** by computing separate byte offsets for Q vs K/V using their own strides and `index_kv_head = index_head // (NUM_HEADS // NUM_KV_HEADS)`.
- Boundary-safe: all `tl.load` / `tl.store` calls use `boundary_check` + `padding_option="zero"` for sequences that don't align to block sizes.

### Backward Pass (`_attn_bwd_dq`, `_attn_bwd_dk_dv`)

- Implements the memory-efficient Flash Attention backward pass using the saved logsumexp `M` from the forward pass.
- Separate kernels for computing `dK/dV` (fix Q-block, iterate KV) and `dQ` (fix KV-block, iterate Q).
- Supports causal masking in the backward pass.

### Integration (`TritonAttention`)

```python
# flash_attention_triton.py
class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale): ...
    @staticmethod
    def backward(ctx, dO): ...
```

```python
# src/flash_attention.py — the router
def flash_attention_forward(module, query, key, value, ...):
    # Prefill (Q_len == KV_len): use Triton kernel
    # Decode  (Q_len == 1):      fallback to SDPA with GQA head broadcasting
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (Triton requires CUDA)
- CUDA toolkit ≥ 11.8

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Qwen2.5-from-scratch.git
cd Qwen2.5-from-scratch

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

Place the pre-trained Qwen2.5-Coder-3B-Instruct weights in the `model/` directory. You can download them from HuggingFace:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Coder-3B-Instruct', local_dir='./model')
"
```

### Run Inference

```bash
python -m examples.inference
```

With optional arguments:

```bash
python -m examples.inference \
  --model_path ./model \
  --input_text "Write a Python function to compute the Fibonacci sequence." \
  --dtype bfloat16 \
  --max_new_tokens 512 \
  --temperature 0.7
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `./model` | Path to model weights |
| `--input_text` | *(essay prompt)* | Prompt to send to the model |
| `--dtype` | `bfloat16` | Model dtype (`bfloat16`, `float32`) |
| `--max_new_tokens` | `1024` | Max tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--device_map` | `cuda` | Device (`cuda`, `cpu`, `auto`) |

---

## How It Works — Attention Routing

```
model.generate(prompt)
        │
        ├── Prefill phase (process full prompt)
        │       │
        │       └── flash_attention_forward()
        │               │  Q_len == KV_len  →  TritonAttention.apply()
        │               │                       (custom Triton kernel)
        │
        └── Decode phase (generate token-by-token)
                │
                └── flash_attention_forward()
                        │  Q_len == 1, KV_len > 1  →  F.scaled_dot_product_attention()
                        │                              (with KV head repeat for GQA)
```

---

## Development

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Run tests:

```bash
pytest
```

The standalone Triton kernel test (no model required):

```bash
python flash_attention_triton.py
```

This runs a correctness check comparing the custom kernel's output and gradients against PyTorch's reference implementation.

---

## Known Limitations

- The custom Triton backward pass kernels do not yet support GQA (same-stride assumption on Q/K/V). Training with the custom kernel is therefore not recommended without further modification.
- `tl.make_block_ptr` usage generates a deprecation warning in newer Triton versions — a future update will migrate to `tl.make_tensor_descriptor`.
- Sliding window attention (used in some Qwen2 variants) is not implemented in the custom kernel.

---

## References

- [Flash Attention 2 paper](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [Triton documentation](https://triton-lang.org/)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
- [HuggingFace Transformers — Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)
- [Umar Jamil's Triton Flash Attention tutorial](https://github.com/umarbutool/triton-flash-attention) — referenced for kernel structure

---

## License

This project is for educational and research purposes.
