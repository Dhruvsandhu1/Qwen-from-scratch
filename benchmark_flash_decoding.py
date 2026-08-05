import torch
import triton
import torch.nn.functional as F
from flash_decoding import flash_decode


def benchmark():
    torch.manual_seed(42)
    # Define parameters for the benchmark
    BATCH = 1
    N_HEADS = 2
    HEAD_DIM = 128

    # We will vary the context length (KV cache length) to see the scaling
    seq_lengths = [2**i for i in range(10, 17)]  # From 1024 to 32768

    print(f"Benchmarking Decoding Performance")
    print(f"Batch Size: {BATCH}, Heads: {N_HEADS}, Head Dim: {HEAD_DIM}\n")
    print(f"{'Seq Len (KV)':<15} | {'SDPA (ms)':<15} | {'Flash Decoding (ms)':<20} | {'Speedup':<10}")
    print("-" * 70)

    for N_CTX in seq_lengths:
        q = torch.randn((BATCH, N_HEADS, 1, HEAD_DIM), dtype=torch.float16, device="cuda")
        k = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")
        v = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")

        # Benchmark SDPA
        def run_sdpa():
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Benchmark custom Flash Decoding
        def run_flash_decoding():
            return flash_decode(q, k, v)

        # Warmup and benchmarking using Triton's do_bench which accurately measures GPU time
        ms_sdpa = triton.testing.do_bench(run_sdpa, quantiles=None)
        ms_flash = triton.testing.do_bench(run_flash_decoding, quantiles=None)

        speedup = ms_sdpa / ms_flash

        print(f"{N_CTX:<15} | {ms_sdpa:<15.4f} | {ms_flash:<20.4f} | {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()
