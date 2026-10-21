import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from triton_custom.flash_attention_triton import TritonAttention
from triton_custom.flash_decoding_triton import flash_decode
from loguru import logger


def get_target_dtype(query: torch.Tensor, module: torch.nn.Module) -> torch.dtype:
    """If the query is in float32, return a target dtype compatible with flash attention. Return None otherwise."""
    if query.dtype == torch.float32:
        device_type = query.device.type
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_is_quantized"):
            return module.config.dtype
        else:
            return next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
    return None


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning("Flash Attention does not support `output_attentions=True`." " Please set your attention to `eager` if you want any of these features.")

    if any(dim == 0 for dim in query.shape):
        raise ValueError("Tensor query has shape  with a zero dimension.\n" "FlashAttention does not support inputs with dim=0.\n" "Please check your input shapes or use SDPA instead.")

    # INT8 keys/values (from QuantizedStaticCache) come with per-token scales.
    key_scales = kwargs.get("key_scales", None)
    value_scales = kwargs.get("value_scales", None)
    is_int8 = key_scales is not None and value_scales is not None

    target_dtype = get_target_dtype(query, module)
    if target_dtype is not None:
        query = query.to(target_dtype)
        if not is_int8:
            key = key.to(target_dtype)
            value = value.to(target_dtype)

    is_causal = is_causal if is_causal is not None else module.is_causal
    if scaling is None:
        scaling = 1.0 / (query.shape[-1] ** 0.5)

    # Shapes: [batch, num_heads, seq_len, head_dim]. `attention_mask` is an SDPA-style boolean mask
    # (True = attend), or None when plain causal attention with no padding is enough.
    q_len, kv_len = query.shape[2], key.shape[2]

    if q_len == 1 and attention_mask is None:
        # Decode step: Triton flash decoding, dequantizing INT8 keys/values in registers if given.
        # `kv_len` (a device tensor) is set by the CUDA-graph decode step, where key/value are the
        # whole fixed-size cache buffers; otherwise every given key/value token is valid.
        attn_output = flash_decode(query, key, value, sm_scale=scaling, key_scales=key_scales, value_scales=value_scales, kv_len=kwargs.get("kv_len"))
    elif is_int8:
        raise ValueError("INT8 keys/values are only supported for single-token decoding without a padding mask.")
    elif attention_mask is None and q_len == kv_len:
        # Prefill with no earlier tokens and no padding: Triton FlashAttention-2.
        attn_output = TritonAttention.apply(query, key, value, is_causal, scaling)
    else:
        # Padding, or several new tokens on top of a cache: PyTorch SDPA with an explicit mask.
        if query.shape[1] != key.shape[1]:  # GQA: SDPA needs one KV head per query head
            num_queries_per_kv = query.shape[1] // key.shape[1]
            # expand is a zero-copy view, much cheaper than repeat_interleave
            batch, h_kv, seq, dim = key.shape
            key = key[:, :, None, :, :].expand(batch, h_kv, num_queries_per_kv, seq, dim).reshape(batch, -1, seq, dim)
            value = value[:, :, None, :, :].expand(batch, h_kv, num_queries_per_kv, seq, dim).reshape(batch, -1, seq, dim)

        if attention_mask is None and is_causal:
            # The new queries are the last q_len positions, so the causal mask is aligned bottom-right.
            # (SDPA's `is_causal=True` aligns it top-left, which is wrong when q_len < kv_len.)
            attention_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).tril(diagonal=kv_len - q_len)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout if module.training else 0.0, scale=scaling)

    return attn_output.transpose(1, 2).contiguous(), None
