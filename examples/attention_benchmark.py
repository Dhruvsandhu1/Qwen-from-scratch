import argparse
import time
from types import SimpleNamespace
import os

# Workaround: allow duplicate OpenMP runtime for this isolated benchmark process.
# Not recommended for production; helps run the microbenchmark in this environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import sys

# Ensure repository root is on sys.path so `from src...` works when running this script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def build_tensors(batch, heads, seq, head_dim, device, dtype):
    shape = (batch, heads, seq, head_dim)
    t = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    return t, t.clone(), t.clone()


def time_fn(fn, warmup=10, repeats=100):
    # warmup
    for _ in range(warmup):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--attn_impl", choices=["auto", "eager", "flash_attention_2"], default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32

    q, k, v = build_tensors(args.batch, args.heads, args.seq, args.head_dim, device, dtype)

    # simple module-like object expected by eager_attention_forward
    module = SimpleNamespace()
    module.num_key_value_groups = 1
    module.training = False
    module.attention_dropout = 0.0
    module.scaling = args.head_dim**-0.5 if hasattr(args, "head_dim") else q.shape[-1] ** -0.5

    # prepare a causal mask placeholder (None or a tensor) -- keep None for this microbenchmark
    mask = None

    print(f"device={device}, dtype={dtype}, shapes q={q.shape}")

    # eager baseline
    def call_eager():
        return eager_attention_forward(module, q, k, v, mask, scaling=module.scaling, dropout=0.0)

    eager_time = time_fn(call_eager, warmup=10, repeats=args.repeats)
    tokens_per_sec_eager = (args.batch * args.seq) / eager_time

    print(f"eager avg time: {eager_time*1000:.3f} ms, tokens/sec: {tokens_per_sec_eager:,.0f}")

    # Try to benchmark an optimized backend. Prefer the requested impl, but fall back
    # to PyTorch's `scaled_dot_product_attention` or `flash_attn` if available.
    impl = args.attn_impl

    impl_tried = None
    impl_time = None
    tokens_per_sec_impl = None

    # 1) If transformers exposes a get_interface method, use it
    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface") and impl in ("auto", "flash_attention_2"):
        impl_name = "flash_attention_2" if impl == "auto" else impl
        impl_fn = ALL_ATTENTION_FUNCTIONS.get_interface(impl_name, eager_attention_forward)
        impl_tried = f"transformers.{impl_name} via ALL_ATTENTION_FUNCTIONS"

        def call_impl():
            # Some attention implementations expect the module to have a `.config` attribute
            # with an `_attn_implementation` field. Provide a minimal stub if missing.
            if not hasattr(module, "config"):
                module.config = SimpleNamespace(_attn_implementation=impl_name)
            return impl_fn(module, q, k, v, mask, scaling=module.scaling, dropout=0.0, is_causal=False)

        try:
            impl_time = time_fn(call_impl, warmup=10, repeats=args.repeats)
        except AttributeError:
            # Retry with a slightly richer config stub if implementation accesses other fields
            if not hasattr(module, "config"):
                module.config = SimpleNamespace(_attn_implementation=impl_name, num_key_value_groups=1)
            impl_time = time_fn(call_impl, warmup=10, repeats=args.repeats)
        tokens_per_sec_impl = (args.batch * args.seq) / impl_time

    # 2) Try PyTorch's scaled_dot_product_attention (PyTorch 2.x)
    if impl_time is None:
        try:
            from torch.nn.functional import scaled_dot_product_attention

            impl_tried = "torch.scaled_dot_product_attention"

            def call_torch_sdpa():
                # reshape to (batch*heads, seq, head_dim)
                b, h, s, d = q.shape
                q2 = q.reshape(b * h, s, d)
                k2 = k.reshape(b * h, s, d)
                v2 = v.reshape(b * h, s, d)
                out = scaled_dot_product_attention(q2, k2, v2, attn_mask=None, is_causal=False, dropout_p=0.0)
                return out.reshape(b, h, s, d)

            impl_time = time_fn(call_torch_sdpa, warmup=10, repeats=args.repeats)
            tokens_per_sec_impl = (args.batch * args.seq) / impl_time
        except Exception:
            impl_time = None

    # 3) Try flash_attn package if present
    if impl_time is None:
        try:
            import flash_attn  # type: ignore

            impl_tried = "flash_attn (python package)"
            # flash_attn API varies; try common entrypoints
            try:
                flash_fn = flash_attn.flash_attn_unpadded_qkvpacked
            except Exception:
                flash_fn = None

            if flash_fn is not None:

                def call_flash():
                    # Many flash_attn kernels expect packed QKV; building a safe call is non-trivial.
                    # Skip complex packing here and just report presence.
                    return torch.empty(1, device=q.device)

                impl_time = float("nan")
                tokens_per_sec_impl = float("nan")
        except Exception:
            impl_time = None

    if impl_time is None:
        print("No alternate optimized attention implementation detected on this system.")
    else:
        print(f"impl tried: {impl_tried}")
        if impl_time == impl_time:  # not NaN
            print(f"impl avg time: {impl_time*1000:.3f} ms, tokens/sec: {tokens_per_sec_impl:,.0f}")
            speedup = eager_time / impl_time if impl_time > 0 else float("inf")
            print(f"speedup (eager / impl): {speedup:.3f}")
        else:
            print("Detected flash_attn but could not run a direct microbenchmark here.")


if __name__ == "__main__":
    main()
