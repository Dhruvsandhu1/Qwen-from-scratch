"""Lightweight benchmark for Qwen2 models (latency, throughput, GPU memory).

Usage: python examples/benchmark.py --model_path ./model --batch_size 1 --seq_len 32
"""

import time
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from src.modeling_qwen2 import Qwen2ForCausalLM
from loguru import logger


def measure(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    torch_dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else None
    model = Qwen2ForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    # prepare synthetic input tokens of length seq_len
    sample_text = "Hello world " * (args.seq_len // 2 + 1)
    inputs = tokenizer([sample_text.strip()] * args.batch_size, return_tensors="pt")
    input_ids = inputs.input_ids[:, : args.seq_len].to(device)

    # warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.generate(input_ids, max_new_tokens=args.max_new_tokens)

    # benchmark
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    tokens_generated = 0
    with torch.no_grad():
        for i in range(args.iters):
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
            # count newly generated tokens per batch (approx)
            tokens_generated += (out.shape[1] - input_ids.shape[1]) * input_ids.shape[0]
    elapsed = time.time() - start

    results = {
        "device": device.type,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "max_new_tokens": args.max_new_tokens,
        "iters": args.iters,
        "total_time_s": elapsed,
        "samples_per_sec": args.iters * args.batch_size / elapsed if elapsed > 0 else float("inf"),
        "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else float("inf"),
    }

    if device.type == "cuda":
        results["gpu_peak_mem_bytes"] = torch.cuda.max_memory_allocated()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen2 model")
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations to measure")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--dtype", type=str, default="float16", help="torch dtype name, e.g. float16, bfloat16, float32")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    logger.info("Loading and running benchmark, this may take a while...")
    results = measure(args)

    logger.info("\nBenchmark results:")
    for k, v in results.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
