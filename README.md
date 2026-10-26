# Qwen2.5 From Scratch, with Custom Triton Kernels

A from-scratch PyTorch implementation of the Qwen2 architecture (tested with **Qwen2.5-Coder-3B-Instruct**), with hand-written [Triton](https://github.com/triton-lang/triton) kernels for attention, decoding, normalization and the MLP. It loads standard Hugging Face weights and works with `AutoTokenizer` and `model.generate()`.

On an RTX 4060 Laptop GPU it generates text **about 1.18× faster** than the stock Hugging Face model, with the same weights and settings.

## Results

Stock Hugging Face (`AutoModelForCausalLM`, default settings) vs this implementation. Same weights, bf16, batch size 1, 256 generated tokens, median of 3 runs.

| Prompt length | Decode speed (HF → this repo) | Speedup | Time to first token (HF → this repo) | Speedup |
|---|---|---|---|---|
| ~40 tokens | 30.9 → 36.3 tok/s | 1.17× | 37.1 → 37.3 ms | 1.00× |
| ~500 tokens | 30.5 → 36.2 tok/s | 1.19× | 138.0 → 136.8 ms | 1.01× |
| ~1.5k tokens | 30.1 → 35.8 tok/s | 1.19× | 340.2 → 325.2 ms | 1.05× |
| ~4k tokens | 29.3 → 34.9 tok/s | 1.19× | 1034.5 → 948.0 ms | 1.09× |

- **Multi-turn chat** (sampling, growing history): 30.3 → 33.8 tok/s (**1.12×**).
- **KV cache size:** the INT8 cache needs about half the memory of a bf16 cache (int8 values plus one fp32 scale per token and head).

Measured on an RTX 4060 Laptop GPU (8 GB) under WSL2, with PyTorch 2.14, Triton 3.8 and transformers 5.20. Run `examples/compare_with_hf.py` to reproduce this on your own GPU.

At batch size 1, every generated token has to read all 6.2 GB of weights, which puts the ceiling at roughly 38 tok/s on this GPU. Most of the gain comes from removing CPU overhead so the GPU never waits.

Outputs are not bit-identical to Hugging Face: the INT8 cache and different kernel arithmetic can flip near-tied tokens, so greedy outputs sometimes diverge after a few tokens.

## What's custom

| Component | What it does |
|---|---|
| **Flash Attention (prefill)** `triton_custom/flash_attention_triton.py` | Tiled causal attention with online softmax and native grouped-query attention. One compiled kernel serves every prompt length, and autotuning results are cached on disk. It also has a backward pass for standalone use, which needs sequence lengths that are a multiple of 128 (the model trains with PyTorch SDPA instead). |
| **Flash Decoding (generation)** `triton_custom/flash_decoding_triton.py` | Splits the KV cache across GPU blocks, then merges the partial softmax results. Reads the INT8 cache directly and dequantizes in registers. |
| **INT8 KV cache** `QuantizedStaticCache` | Stores keys and values as int8 with one scale per token and head, which halves cache memory. Used automatically by `generate()`. |
| **CUDA-graph decoding** | Each decode step is captured once and replayed with a single launch, instead of ~1,200 small kernel launches from Python. |
| **Fused kernels** | RMSNorm, SwiGLU and rotary embeddings (RoPE), each in a single Triton kernel. |

All of this is used automatically when the model is loaded with `attn_implementation="triton_attention"` and run on a CUDA GPU. Some cases fall back to standard PyTorch code:

- **CPU, or when gradients are needed (training):** plain PyTorch everywhere.
- **Padded batches:** PyTorch SDPA with an attention mask, and no CUDA graph.
- **To switch features off:** set `model.config.use_int8_kv_cache = False` or `model.config.use_cuda_graph = False`.

## Project structure

```
├── src/
│   ├── modeling_qwen2.py        # The model: attention, MLP, RMSNorm, RoPE, INT8 cache, CUDA-graph decoding
│   ├── attention.py             # Picks the attention kernel: Triton prefill, flash decoding, or SDPA fallback
│   └── configuration_qwen2.py   # Qwen2Config
├── triton_custom/
│   ├── flash_attention_triton.py   # Flash Attention kernel (prefill)
│   └── flash_decoding_triton.py    # Flash Decoding kernel (generation, INT8 support)
├── examples/
│   ├── interactive.py           # Chat in the terminal
│   └── compare_with_hf.py       # Benchmark against stock Hugging Face
├── tests/                       # Kernels and a tiny random model vs Hugging Face, plus config/tokenizer smoke tests
└── model/                       # Model weights and tokenizer (download separately)
```

## Quickstart

**Requirements:** an NVIDIA GPU and Linux or WSL2. Triton does not run natively on Windows.

```bash
# 1. Install PyTorch with CUDA (includes Triton): https://pytorch.org/get-started/locally/
#    then the remaining dependencies
pip install -r requirements.txt

# 2. Download the weights into ./model
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-3B-Instruct', local_dir='./model')"

# 3. Chat
python examples/interactive.py
```

Under WSL2, keep the project and its virtual environment on the Linux filesystem (e.g. `~/`) rather than on `/mnt/c` or `/mnt/d`. Reading from the Windows drive is much slower: importing `transformers` alone can take over a minute.

### Use it in your own code

```python
from transformers import AutoTokenizer
from src.modeling_qwen2 import Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./model")
model = Qwen2ForCausalLM.from_pretrained("./model", dtype="auto", device_map="cuda", attn_implementation="triton_attention")

messages = [{"role": "user", "content": "Write a python function that checks if a number is prime."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Benchmark against Hugging Face

```bash
python examples/compare_with_hf.py                                       # full run, ~6 minutes
python examples/compare_with_hf.py --reps 1 --new_tokens 64 --skip_chat  # quick run
```

The script loads each model in its own process and reports time to first token, decode speed, total time, extra memory, and whether the greedy outputs match.

### Tests

```bash
pip install pytest
pytest tests
```

`tests/test_kernels.py` checks the Triton kernels against PyTorch. `tests/test_model.py` builds a tiny random Qwen2 and checks it against the Hugging Face implementation with the same weights: generation with and without the INT8 cache and CUDA graph, cache reuse across `generate()` calls, padded batches, the CPU fallback, and training gradients. They take a few minutes the first time (kernel compilation and autotuning) and skip on machines without a CUDA GPU, except the CPU test. The smoke tests need the weights in `./model`.

## Limitations

- **Batch size:** tuned for batch size 1 (interactive use). Batches work, but padded batches skip the CUDA graph.
- **Supported models:** head dimension must be a power of two (all Qwen2 sizes use 64 or 128). Sliding-window attention is not supported.
- **First run:** each new prompt-length bucket costs a one-time autotuning step of under a second, and the very first run compiles the kernels (~10 s). Both are cached on disk afterwards.
- **Flash decoding speed:** each query head reads its KV head separately, so the kernel is slower than PyTorch's SDPA at long contexts. It is a small part of total time at chat lengths.

## References

- [FlashAttention-2](https://arxiv.org/abs/2307.08691), Dao, 2023
- [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [Triton documentation](https://triton-lang.org/)
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)
