import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flash_attention_triton import TritonAttention

try:
    from loguru import logger

    logger = logger.bind(name=__name__)
except Exception:
    import logging as _logging

    logger = _logging.getLogger(__name__)


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
    s_aux: torch.Tensor | None = None,  # alias: learnable attention sink
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning("Flash Attention does not support `output_attentions=True`." " Please set your attention to `eager` if you want any of these features.")

    if any(dim == 0 for dim in query.shape):
        raise ValueError("Tensor query has shape  with a zero dimension.\n" "FlashAttention does not support inputs with dim=0.\n" "Please check your input shapes or use SDPA instead.")

    target_dtype = get_target_dtype(query, module)
    if target_dtype is not None:
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    is_causal = is_causal if is_causal is not None else module.is_causal

    if query.shape[2] != key.shape[2]:
        import torch.nn.functional as F
        import torch

        # Triton kernel doesn't support cross-attention or decoding (where Q_LEN != KV_LEN).
        # Fallback to standard PyTorch SDPA for decoding.

        # SDPA requires matching head counts, so we repeat KV heads for GQA
        if query.shape[1] != key.shape[1]:
            num_queries_per_kv = query.shape[1] // key.shape[1]
            key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

        # SDPA requires (batch, heads, seq, dim) layout which we already have.
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout if module.training else 0.0, is_causal=is_causal and query.shape[2] > 1, scale=scaling)
        return attn_output.transpose(1, 2).contiguous(), None

    if scaling is None:
        scaling = 1.0 / (query.shape[-1] ** 0.5)

    attn_output = TritonAttention.apply(query, key, value, is_causal, scaling)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
