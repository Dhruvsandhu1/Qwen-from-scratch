"""Compare the custom Triton implementation against stock Hugging Face Qwen2.

Both models load the same local weights and run the same `generate()` calls:
  - "hf":     AutoModelForCausalLM.from_pretrained(..., dtype="auto"), i.e. what you get by just
              downloading the model (bf16, PyTorch SDPA attention, default cache).
  - "custom": src.modeling_qwen2.Qwen2ForCausalLM with attn_implementation="triton_attention"
              (Triton FlashAttention prefill, flash decoding, INT8 KV cache, CUDA-graph decode,
              fused RMSNorm/SwiGLU/RoPE kernels).

Each model runs in its own subprocess, so memory numbers and compiled kernels do not mix.

Usage:
    python examples/compare_with_hf.py
    python examples/compare_with_hf.py --new_tokens 128 --reps 1 --skip_chat   # quicker run
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

RESULT_PREFIX = "RESULT_JSON:"
FILLER = "The quick brown fox jumps over the lazy dog. "  # 9 words, about 10 tokens
CHAT_QUESTIONS = ["What is a hash map?", "How does it handle collisions?", "Write a tiny one in Python.", "What is its time complexity?"]


def build_prompts(tokenizer):
    def chat(text):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    return {
        "short (~40 tok)": chat("Write a python function that checks if a number is prime."),
        "medium (~500 tok)": chat("Summarize this: " + FILLER * 50),
        "long (~1.5k tok)": chat("Summarize this: " + FILLER * 150),
        "very long (~4k tok)": chat("Summarize this: " + FILLER * 411),
    }


def run_worker(args):
    """Benchmark one implementation and print its results as a JSON line."""
    import torch
    from transformers import AutoTokenizer

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = build_prompts(tokenizer)

    t0 = time.perf_counter()
    if args.worker == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype="auto", device_map=device)
    else:
        from src.modeling_qwen2 import Qwen2ForCausalLM

        model = Qwen2ForCausalLM.from_pretrained(args.model_path, dtype="auto", device_map=device, attn_implementation="triton_attention")
    load_s = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / 1e9

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - start

    result = {"load_s": load_s, "weights_gb": weights_gb, "cases": {}}
    with torch.no_grad():
        for name, text in prompts.items():
            inputs = tokenizer([text], return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            # Warm-up: lets the custom model compile/autotune its kernels for this prompt length.
            model.generate(**inputs, max_new_tokens=4, do_sample=False)
            torch.cuda.reset_peak_memory_stats()

            prefill, decode, total = [], [], []
            for _ in range(args.reps):
                # Time to first token = prefill (plus one sampling step).
                _, t_first = timed(lambda: model.generate(**inputs, max_new_tokens=1, do_sample=False))
                # Full answer; min_new_tokens forces the same length for both models.
                out, t_all = timed(lambda: model.generate(**inputs, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False))
                prefill.append(t_first)
                total.append(t_all)
                decode.append((args.new_tokens - 1) / (t_all - t_first))

            result["cases"][name] = {
                "prompt_tokens": prompt_len,
                "prefill_ms": statistics.median(prefill) * 1000,
                "decode_tok_s": statistics.median(decode),
                "total_s": statistics.median(total),
                # Memory used beyond the weights: KV cache, activations, logits.
                "peak_extra_mem_mb": (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / 1e6,
                "tokens": out[0, prompt_len:].tolist(),
            }
            c = result["cases"][name]
            print(f"  [{args.worker}] {name:20s} prefill {c['prefill_ms']:7.1f} ms | decode {c['decode_tok_s']:5.1f} tok/s", file=sys.stderr, flush=True)

        if not args.skip_chat:
            # A chat session with the model's default sampling settings; the history grows every turn.
            torch.manual_seed(0)
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            chat_tokens, chat_seconds = 0, 0.0
            for question in CHAT_QUESTIONS:
                messages.append({"role": "user", "content": question})
                inputs = tokenizer([tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)], return_tensors="pt").to(device)
                out, dt = timed(lambda: model.generate(**inputs, max_new_tokens=200, min_new_tokens=200))
                new_ids = out[0, inputs.input_ids.shape[1] :]
                chat_tokens += len(new_ids)
                chat_seconds += dt
                messages.append({"role": "assistant", "content": tokenizer.decode(new_ids, skip_special_tokens=True)})
            result["chat"] = {"tokens": chat_tokens, "seconds": chat_seconds}

    print(RESULT_PREFIX + json.dumps(result), flush=True)


def run_in_subprocess(which, args):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", which, "--model_path", args.model_path, "--new_tokens", str(args.new_tokens), "--reps", str(args.reps)]
    if args.skip_chat:
        cmd.append("--skip_chat")
    print(f"Benchmarking {'stock Hugging Face' if which == 'hf' else 'custom Triton'} model...", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, cwd=repo_root)
    if proc.returncode != 0:
        sys.exit(f"The {which} benchmark failed (exit code {proc.returncode}).")
    line = next(l for l in proc.stdout.splitlines() if l.startswith(RESULT_PREFIX))
    return json.loads(line[len(RESULT_PREFIX) :])


def first_difference(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


def print_report(hf, custom, args):
    print(f"\nResults (median of {args.reps} run(s), {args.new_tokens} new tokens, batch 1)\n")
    header = f"{'Prompt':<20} {'Tokens':>6} | {'Prefill HF':>10} {'Custom':>9} {'Speedup':>7} | {'Decode HF':>9} {'Custom':>8} {'Speedup':>7} | {'Total speedup':>13} | {'Extra mem HF':>12} {'Custom':>8}"
    print(header)
    print("-" * len(header))
    for name, h in hf["cases"].items():
        c = custom["cases"][name]
        print(
            f"{name:<20} {h['prompt_tokens']:>6} | "
            f"{h['prefill_ms']:>8.1f}ms {c['prefill_ms']:>7.1f}ms {h['prefill_ms'] / c['prefill_ms']:>6.2f}x | "
            f"{h['decode_tok_s']:>5.1f}t/s {c['decode_tok_s']:>4.1f}t/s {c['decode_tok_s'] / h['decode_tok_s']:>6.2f}x | "
            f"{h['total_s'] / c['total_s']:>12.2f}x | "
            f"{h['peak_extra_mem_mb']:>10.0f}MB {c['peak_extra_mem_mb']:>6.0f}MB"
        )

    if "chat" in hf:
        h_rate = hf["chat"]["tokens"] / hf["chat"]["seconds"]
        c_rate = custom["chat"]["tokens"] / custom["chat"]["seconds"]
        print(f"\nChat (4 sampled turns, growing history): HF {h_rate:.1f} tok/s, custom {c_rate:.1f} tok/s -> {c_rate / h_rate:.2f}x")

    print(f"\nLoad time: HF {hf['load_s']:.1f}s, custom {custom['load_s']:.1f}s | Weights in GPU memory: HF {hf['weights_gb']:.2f} GB, custom {custom['weights_gb']:.2f} GB")

    print("\nGreedy output agreement (INT8 KV cache and different kernel arithmetic can flip near-tied tokens):")
    for name, h in hf["cases"].items():
        diff = first_difference(h["tokens"], custom["cases"][name]["tokens"])
        print(f"  {name:<20} " + ("identical" if diff is None else f"first differs at generated token {diff}"))


def main():
    parser = argparse.ArgumentParser(description="Compare the custom Triton Qwen2 against stock Hugging Face Qwen2.")
    parser.add_argument("--model_path", default=os.path.join(repo_root, "model"), help="Local model directory (weights, config, tokenizer)")
    parser.add_argument("--new_tokens", type=int, default=256, help="Tokens to generate per measurement")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per measurement (the median is reported)")
    parser.add_argument("--skip_chat", action="store_true", help="Skip the multi-turn chat measurement")
    parser.add_argument("--worker", choices=["hf", "custom"], help=argparse.SUPPRESS)  # internal: run one side only
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    hf = run_in_subprocess("hf", args)
    custom = run_in_subprocess("custom", args)
    print_report(hf, custom, args)


if __name__ == "__main__":
    main()
