# Qwen2.5-Coder-3B-Instruct — From Scratch

> A full from-scratch PyTorch implementation of **Qwen2.5-Coder-3B-Instruct**, featuring a highly optimized suite of custom GPU-accelerated kernels written in [Triton](https://github.com/triton-lang/triton).

---

## Overview

This project re-implements the [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) model architecture entirely from scratch in PyTorch, without relying on the HuggingFace `transformers` modelling code for the core forward pass. The highlight of this implementation is a **highly optimized, bandwidth-bound inference and training stack** driven by custom Triton kernels.

### ⚡ Performance Highlights & Optimizations

- 🧠 **From-scratch implementation** — `Qwen2ForCausalLM`, `Qwen2Attention`, `Qwen2MLP`, `Qwen2RMSNorm`, and `Qwen2RotaryEmbedding` implemented manually.
- 🚀 **Custom Triton Flash Attention (Prefill)** 
  - Tiled SRAM-based computation for O(1) memory w.r.t. sequence length.
  - Auto-tuning via `@triton.autotune` for both **forward and backward** passes.
  - **L2 Cache Swizzling** for maximized memory locality during grid execution.
  - Causal masking and native Grouped Query Attention (GQA) support.
- ⚡ **Dynamic Triton Flash Decoding (Generation)**
  - Replaces PyTorch's SDPA for the decoding phase.
  - Dynamically calculates the optimal `SPLIT_K` factor based on the GPU's hardware topology to guarantee **100% Streaming Multiprocessor (SM) occupancy** regardless of sequence length.
- 🗜️ **Int8 KV Cache Quantization**
  - Uses `QuantizedStaticCache` to dynamically compress the Key and Value states to `int8` before storing them in VRAM.
  - The Triton Flash Decoding kernel natively performs **on-the-fly dequantization** directly in SRAM registers, doubling memory bandwidth during generation!
- 🏎️ **Fused Custom Kernels**
  - **Fused SwiGLU**: `_swiglu_kernel` eliminates intermediate tensor allocations during the MLP pass.
  - **Fused RMSNorm**: `_rmsnorm_kernel` performs root-mean-square normalization in a single pass.
  - **Fused RoPE**: `_rope_kernel` applies Rotary Position Embeddings without slicing or copying.
- 📉 **Zero-Copy GQA**
  - Replaced native `repeat_interleave` with a zero-allocation `.expand()` projection for Key and Value heads.
- 📊 **Static KV Caching**
  - Statically pre-allocates VRAM to eliminate dynamic shape changes and `realloc` bottlenecks during generation.
  - Fully unblocks **CUDA Graph** capture for Python-overhead-free inference.
- ✅ **Drop-in Compatible** with HuggingFace's `AutoTokenizer` and `GenerationMixin`.

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
| Attention | Flash Attention 2 (Prefill) & Flash Decoding (Generation) |

---

## Project Structure

```
Qwen2.5/
├── src/
│   ├── __init__.py
│   ├── configuration_qwen2.py          # Qwen2Config — model hyperparameters
│   ├── modeling_qwen2.py               # Full model: Qwen2ForCausalLM, Qwen2Model, Qwen2Attention, etc.
│   ├── flash_attention.py              # Attention router: Triton kernel (prefill) + Flash Decoding (decode)
│   └── modeling_flash_attention_utils.py  
│
├── flash_attention_triton.py           # Custom Triton Flash Attention forward + backward kernel
├── flash_decoding.py                   # Custom Triton Flash Decoding kernel with Int8 dequantization
│
├── examples/
│   ├── inference.py                    # End-to-end inference script
│   ├── interactive.py                  # Interactive chat script
│   └── benchmark.py                    # Benchmark tokens-per-second throughput
│
├── model/
│   ├── config.json                     # Model config (loaded via AutoConfig)
│   ├── tokenizer_config.json
│   ├── *.safetensors                   # Pre-trained weights (not committed)
│   └── model.safetensors.index.json
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

To run the interactive chat:
```bash
python examples/interactive.py
```

To benchmark the inference optimizations (Flash Decoding + Int8 KV Cache):
```bash
python examples/benchmark.py --batch_size 4 --seq_len 1024 --max_new_tokens 512
```

---

## The Ultimate Optimization: `torch.compile`

Because this implementation utilizes a **Static KV Cache** and has fused away dynamic Python control flow, it is perfectly primed for PyTorch 2's JIT compiler.

If you are writing a custom inference script, simply add:
```python
model = torch.compile(model, mode="reduce-overhead")
```
This will allow PyTorch to natively trace the `Qwen2Attention` and `Qwen2MLP` blocks and fuse the parallel Linear projections (`q_proj`, `k_proj`, `v_proj`) together natively without breaking HuggingFace checkpoint loading!

---

## References

- [Flash Attention 2 paper](https://arxiv.org/abs/2307.08691) — Dao, 2023
- [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [Triton documentation](https://triton-lang.org/)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
- [HuggingFace Transformers — Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)

---

## License

This project is for educational and research purposes.
