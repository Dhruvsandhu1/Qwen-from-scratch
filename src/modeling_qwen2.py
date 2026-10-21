"""
PyTorch Implementation of Qwen2 with Custom GPU Triton Kernels.

This module provides a from-scratch, highly optimized implementation of the
Qwen2 architecture, fully compatible with the
Hugging Face Transformers ecosystem (e.g., PreTrainedModel, GenerationMixin).

Key Performance Features & Custom Kernels:
    - Fused Triton Kernels:
        * `_swiglu_kernel`: Fused SwiGLU activation for MLP forward passes
          eliminating intermediate allocation overhead.
        * `_rmsnorm_kernel`: Fused Root Mean Square Normalization.
        * `_rope_kernel`: Fused Rotary Position Embeddings applied in-place
          across query and key head dimensions.
    - Quantized Static KV Cache (`QuantizedStaticCache`):
        * Dynamic INT8 KV quantization with per-token scale factors to halve
          cache memory footprint and enable register-level dequantization.
        * Used by `generate()` by default with the Triton backend
          (set `config.use_int8_kv_cache = False` to turn it off).
    - Attention Dispatch:
        * Custom Triton Flash Attention-2 (prefill) and Flash Decoding (generation),
          registered as the "triton_attention" attention backend, with an SDPA fallback
          for padded batches.
    - The Triton kernels are inference-only and CUDA-only; on CPU, or when gradients
      are needed, the equivalent PyTorch code runs instead.
    - Grouped Query Attention (GQA):
        * Memory-efficient projection and key-value replication.

Primary Classes:
    - `Qwen2PreTrainedModel`: Base model class for initialization and configuration.
    - `Qwen2Model`: Core transformer backbone without heads.
    - `Qwen2ForCausalLM`: Causal language model with LM head and generation support.
    - `Qwen2Attention`: Multi-head/GQA self-attention module.
    - `Qwen2MLP`: SwiGLU feed-forward network.
    - `Qwen2RMSNorm`: Root Mean Square Layer Normalization.
    - `QuantizedStaticCache`: Static INT8 key-value cache buffer.

Example:
    >>> from transformers import AutoTokenizer
    >>> from src.modeling_qwen2 import Qwen2ForCausalLM
    >>> from src.configuration_qwen2 import Qwen2Config
    >>>
    >>> config = Qwen2Config.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    >>> model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", attn_implementation="triton_attention")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    >>>
    >>> inputs = tokenizer("def quicksort(arr):", return_tensors="pt")
    >>> outputs = model.generate(**inputs, max_new_tokens=64)
    >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""

from collections.abc import Callable
from typing import Optional
import torch
from torch import nn
import triton
import triton.language as tl

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernelized_func,
)
from transformers.masking_utils import (
    AttentionMaskInterface,
    create_causal_mask,
    create_sliding_window_causal_mask,
    sdpa_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_qwen2 import Qwen2Config
from .attention import flash_attention_forward

# The Triton attention kernels are registered as their own backend, so loading the model does not
# require the `flash-attn` package. Use `from_pretrained(..., attn_implementation="triton_attention")`.
# Masks are built like SDPA masks: `None` when plain causal attention is enough, a boolean mask otherwise.
TRITON_ATTN_IMPLEMENTATION = "triton_attention"
AttentionInterface.register(TRITON_ATTN_IMPLEMENTATION, flash_attention_forward)
AttentionMaskInterface.register(TRITON_ATTN_IMPLEMENTATION, sdpa_mask)


def _can_use_triton(*tensors: torch.Tensor) -> bool:
    """The Triton kernels in this file run on CUDA only and have no backward pass.

    Fall back to PyTorch on CPU, or when autograd needs to track the result (training).
    """
    if not all(t.is_cuda for t in tensors):
        return False
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))


def _use_int8_kv_cache(config) -> bool:
    """`generate()` uses the INT8 cache with the Triton backend, unless `config.use_int8_kv_cache = False`."""
    return config._attn_implementation == TRITON_ATTN_IMPLEMENTATION and getattr(config, "use_int8_kv_cache", True)


@triton.jit
def _swiglu_kernel(GATE, UP, OUT, stride_g_row, stride_u_row, stride_o_row, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)

    GATE += row_idx * stride_g_row
    UP += row_idx * stride_u_row
    OUT += row_idx * stride_o_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    gate = tl.load(GATE + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + cols, mask=mask, other=0.0).to(tl.float32)

    gate_silu = gate * tl.sigmoid(gate)
    out = gate_silu * up

    tl.store(OUT + cols, out.to(GATE.dtype.element_ty), mask=mask)


class QuantizedStaticLayer(CacheLayerMixin):
    """One layer of the INT8 KV cache.

    Keys and values are stored as int8 with one float32 scale per (batch, head, token), using symmetric
    absmax quantization over head_dim. Buffers are allocated once, for `max_cache_len` tokens, on the
    first update, so memory does not grow while decoding.
    """

    is_compileable = False
    is_sliding = False
    is_croppable = True

    def __init__(self, max_cache_len: int, **kwargs):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.cumulative_length = 0
        self.key_scales: torch.Tensor | None = None
        self.value_scales: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        shape = (self.batch_size, self.num_heads, self.max_cache_len)
        self.keys = torch.zeros((*shape, key_states.shape[-1]), dtype=torch.int8, device=self.device)
        self.values = torch.zeros((*shape, value_states.shape[-1]), dtype=torch.int8, device=self.device)
        self.key_scales = torch.zeros((*shape, 1), dtype=torch.float32, device=self.device)
        self.value_scales = torch.zeros((*shape, 1), dtype=torch.float32, device=self.device)
        self.is_initialized = True

    @staticmethod
    def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        return torch.round(x / scale).clamp_(-127, 127).to(torch.int8), scale

    def _dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (q.float() * scale).to(self.dtype)

    def _append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> int:
        """Quantize and store the new tokens. Returns how many tokens were cached before them."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        start = self.cumulative_length
        end = start + key_states.shape[-2]
        if end > self.max_cache_len:
            raise ValueError(f"QuantizedStaticCache is full: {end} tokens needed, max_cache_len is {self.max_cache_len}.")

        k_q, k_s = self._quantize(key_states)
        v_q, v_s = self._quantize(value_states)
        self.keys[:, :, start:end] = k_q
        self.values[:, :, start:end] = v_q
        self.key_scales[:, :, start:end] = k_s
        self.value_scales[:, :, start:end] = v_s
        self.cumulative_length = end
        return start

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Store the new tokens and return full-precision keys/values for every cached token.

        The new tokens are returned exactly as given; only earlier tokens go through the int8 round trip.
        """
        start = self._append(key_states, value_states)
        if start == 0:
            return key_states, value_states
        past_keys = self._dequantize(self.keys[:, :, :start], self.key_scales[:, :, :start])
        past_values = self._dequantize(self.values[:, :, :start], self.value_scales[:, :, :start])
        return torch.cat([past_keys, key_states], dim=-2), torch.cat([past_values, value_states], dim=-2)

    def update_quantized(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Store the new tokens and return int8 keys, int8 values, key scales and value scales for every cached token."""
        self._append(key_states, value_states)
        n = self.cumulative_length
        return self.keys[:, :, :n], self.values[:, :, :n], self.key_scales[:, :, :n], self.value_scales[:, :, :n]

    def update_quantized_at(self, key_states: torch.Tensor, value_states: torch.Tensor, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CUDA-graph version of `update_quantized` for one new token.

        The write slot comes from `position`, a device tensor of shape (1,), and the whole fixed-size
        buffers are returned, so shapes and addresses never change between steps. The caller passes the
        valid length to the attention kernel and advances `cumulative_length` after the step.
        """
        k_q, k_s = self._quantize(key_states)
        v_q, v_s = self._quantize(value_states)
        self.keys.index_copy_(2, position, k_q)
        self.values.index_copy_(2, position, v_q)
        self.key_scales.index_copy_(2, position, k_s)
        self.value_scales.index_copy_(2, position, v_s)
        return self.keys, self.values, self.key_scales, self.value_scales

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cumulative_length + query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        # Stale data past `cumulative_length` is never read, so the buffers do not need zeroing.
        self.cumulative_length = 0

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:
            raise ValueError("Pass a negative number of tokens to `crop`, like `DynamicLayer.crop`.")
        self.cumulative_length = max(0, self.cumulative_length + tokens_to_remove)

    def _map_batch(self, fn) -> None:
        if self.is_initialized:
            self.keys, self.values, self.key_scales, self.value_scales = (fn(t) for t in (self.keys, self.values, self.key_scales, self.value_scales))
            self.batch_size = self.keys.shape[0]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._map_batch(lambda t: t.index_select(0, beam_idx.to(t.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._map_batch(lambda t: t.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._map_batch(lambda t: t[indices])


class QuantizedStaticCache(Cache):
    """INT8 KV cache (one `QuantizedStaticLayer` per decoder layer), holding up to `max_cache_len` tokens.

    With the Triton backend, single-token decode steps read the int8 data directly (`update_quantized`);
    flash decoding dequantizes it in registers. Any other path calls `update`, which returns full-precision tensors.
    """

    def __init__(self, config, max_cache_len: int, **kwargs):
        text_config = config.get_text_config(decoder=True)
        if "sliding_attention" in (getattr(text_config, "layer_types", None) or []):
            raise NotImplementedError("QuantizedStaticCache does not support sliding-window attention layers.")
        super().__init__(layers=[QuantizedStaticLayer(max_cache_len) for _ in range(text_config.num_hidden_layers)])
        self.max_cache_len = max_cache_len
        # False when the prompt had padding: decode steps then need a mask, so they cannot use the CUDA graph.
        self.graph_safe = True
        # Captured decode step (see `Qwen2ForCausalLM._graph_decode`). It records the addresses of this
        # cache's buffers, so it lives on the cache and is reused as long as the buffers are.
        self.decode_graph = None

    def update_quantized(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        return self.layers[layer_idx].update_quantized(key_states, value_states)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # The fused kernel hard-codes SiLU.
        if self.config.hidden_act != "silu" or not _can_use_triton(gate, up):
            return self.down_proj(self.act_fn(gate) * up)

        gate_2d = gate.view(-1, gate.shape[-1])
        up_2d = up.view(-1, up.shape[-1])
        N_ROWS, N_COLS = gate_2d.shape

        out_2d = torch.empty_like(gate_2d)
        BLOCK_SIZE = triton.next_power_of_2(N_COLS)

        _swiglu_kernel[(N_ROWS,)](gate_2d, up_2d, out_2d, gate_2d.stride(0), up_2d.stride(0), out_2d.stride(0), N_COLS, BLOCK_SIZE=BLOCK_SIZE)

        out = out_2d.view_as(gate)
        down_proj = self.down_proj(out)
        return down_proj


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen2Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@triton.jit
def _rope_kernel(
    X,
    COS,
    SIN,
    stride_x_b,
    stride_x_h,
    stride_x_s,
    stride_x_d,
    stride_c_b,
    stride_c_s,
    stride_c_d,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    c_offset = batch_idx * stride_c_b + seq_idx * stride_c_s
    x_offset = batch_idx * stride_x_b + head_idx * stride_x_h + seq_idx * stride_x_s

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HALF_DIM

    x1_ptrs = X + x_offset + cols * stride_x_d
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)

    x2_ptrs = X + x_offset + (cols + HALF_DIM) * stride_x_d
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

    c1_ptrs = COS + c_offset + cols * stride_c_d
    c1 = tl.load(c1_ptrs, mask=mask, other=0.0).to(tl.float32)
    c2_ptrs = COS + c_offset + (cols + HALF_DIM) * stride_c_d
    c2 = tl.load(c2_ptrs, mask=mask, other=0.0).to(tl.float32)

    s1_ptrs = SIN + c_offset + cols * stride_c_d
    s1 = tl.load(s1_ptrs, mask=mask, other=0.0).to(tl.float32)
    s2_ptrs = SIN + c_offset + (cols + HALF_DIM) * stride_c_d
    s2 = tl.load(s2_ptrs, mask=mask, other=0.0).to(tl.float32)

    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2

    tl.store(x1_ptrs, out1.to(X.dtype.element_ty), mask=mask)
    tl.store(x2_ptrs, out2.to(X.dtype.element_ty), mask=mask)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if not _can_use_triton(q, k, cos, sin):
        cos_b, sin_b = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        return q * cos_b + _rotate_half(q) * sin_b, k * cos_b + _rotate_half(k) * sin_b

    # The Triton path rotates q and k in place. It reads them through their strides, so the
    # layout can be [batch, heads, seq, head_dim] or [batch, seq, heads, head_dim].

    # cos and sin have shape [batch, seq, head_dim]. Their batch can be 1 (positions shared by
    # all rows): a batch stride of 0 makes every row read that single entry.
    assert cos.stride() == sin.stride(), "cos and sin must have the same layout"
    stride_c_b = cos.stride(0) if cos.shape[0] > 1 else 0
    BATCH = q.shape[0]
    HEADS_Q = q.shape[1] if unsqueeze_dim == 1 else q.shape[2]
    HEADS_K = k.shape[1] if unsqueeze_dim == 1 else k.shape[2]
    SEQ_LEN = q.shape[2] if unsqueeze_dim == 1 else q.shape[1]
    HEAD_DIM = q.shape[-1]
    HALF_DIM = HEAD_DIM // 2
    BLOCK_SIZE = triton.next_power_of_2(HALF_DIM)

    # Q Launch
    _rope_kernel[(BATCH, HEADS_Q, SEQ_LEN)](q, cos, sin, q.stride(0), q.stride(1 if unsqueeze_dim == 1 else 2), q.stride(2 if unsqueeze_dim == 1 else 1), q.stride(3), stride_c_b, cos.stride(1), cos.stride(2), HALF_DIM, BLOCK_SIZE=BLOCK_SIZE)

    # K Launch
    _rope_kernel[(BATCH, HEADS_K, SEQ_LEN)](k, cos, sin, k.stride(0), k.stride(1 if unsqueeze_dim == 1 else 2), k.stride(2 if unsqueeze_dim == 1 else 1), k.stride(3), stride_c_b, cos.stride(1), cos.stride(2), HALF_DIM, BLOCK_SIZE=BLOCK_SIZE)

    return q, k


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Single-token decode steps on the Triton backend read the INT8 cache directly. Everything
        # else (prefill, padded batches, other backends) gets full-precision keys/values from `update`.
        decode_position = kwargs.pop("decode_position", None)
        read_int8 = isinstance(past_key_values, QuantizedStaticCache) and self.config._attn_implementation == TRITON_ATTN_IMPLEMENTATION and query_states.shape[2] == 1 and attention_mask is None
        if decode_position is not None:
            # CUDA-graph decode step: fixed-size buffers, valid length passed as `kv_len` in kwargs.
            key_states, value_states, key_scales, value_scales = past_key_values.layers[self.layer_idx].update_quantized_at(key_states, value_states, decode_position)
            kwargs["key_scales"] = key_scales
            kwargs["value_scales"] = value_scales
        elif read_int8:
            key_states, value_states, key_scales, value_scales = past_key_values.update_quantized(key_states, value_states, self.layer_idx)
            kwargs["key_scales"] = key_scales
            kwargs["value_scales"] = value_scales
        elif past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        # Heart of the attention computation: call the attention interface (which may be a kernelized function)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@triton.jit
def _rmsnorm_kernel(
    X,
    Y,
    W,
    stride_x_row,
    stride_y_row,
    N_COLS,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X += row_idx * stride_x_row
    Y += row_idx * stride_y_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N_COLS
    rstd = tl.math.rsqrt(var + EPS)

    y = x * rstd * w
    tl.store(Y + cols, y.to(X.dtype.element_ty), mask=mask)


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not _can_use_triton(hidden_states, self.weight):
            input_dtype = hidden_states.dtype
            x = hidden_states.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return self.weight * x.to(input_dtype)

        hidden_states_2d = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        N_ROWS, N_COLS = hidden_states_2d.shape
        Y = torch.empty_like(hidden_states_2d)

        BLOCK_SIZE = triton.next_power_of_2(N_COLS)
        _rmsnorm_kernel[(N_ROWS,)](
            hidden_states_2d,
            Y,
            self.weight,
            hidden_states_2d.stride(0),
            Y.stride(0),
            N_COLS,
            self.variance_epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return Y.view_as(hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen2PreTrainedModel(PreTrainedModel):
    config: Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }


@auto_docstring
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # A direct forward call does not know how many tokens will follow, so it cannot size a
            # static cache. `generate()` knows, and creates a QuantizedStaticCache instead
            # (see `Qwen2ForCausalLM._prepare_cache_for_generation`).
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length):
        """Use the INT8 cache for `generate()` with the Triton backend, unless the caller chose a cache.

        The cache (and the CUDA graph captured for it) is kept on the model and reused by later
        `generate()` calls while it is big enough, so a chat does not re-allocate and re-capture every turn.
        """
        if model_kwargs.get("past_key_values") is None and generation_config.use_cache is not False and generation_config.cache_implementation is None and _use_int8_kv_cache(self.config):
            single_sequence = (generation_config.num_beams or 1) == 1 and (generation_config.num_return_sequences or 1) == 1
            cache = getattr(self, "_int8_kv_cache", None)
            reusable = single_sequence and cache is not None and cache.max_cache_len >= max_cache_length and (not cache.is_initialized or cache.batch_size == batch_size)
            if reusable:
                cache.reset()
            else:
                self._int8_kv_cache = None  # free the old buffers before allocating new ones
                # Round up so slightly longer conversations still fit in the same cache.
                cache = QuantizedStaticCache(self.config, max_cache_len=-(-max_cache_length // 2048) * 2048)
                if single_sequence:
                    self._int8_kv_cache = cache
            model_kwargs["past_key_values"] = cache
            return
        return super()._prepare_cache_for_generation(generation_config, model_kwargs, generation_mode, batch_size, max_cache_length)

    def _can_graph_decode(self, input_ids, inputs_embeds, past_key_values, labels, kwargs) -> bool:
        return (
            isinstance(past_key_values, QuantizedStaticCache)
            and past_key_values.graph_safe
            and past_key_values.get_seq_length() > 0
            and input_ids is not None
            and input_ids.is_cuda
            and input_ids.shape[1] == 1
            and inputs_embeds is None
            and labels is None
            and not torch.is_grad_enabled()
            and self.config._attn_implementation == TRITON_ATTN_IMPLEMENTATION
            and getattr(self.config, "use_cuda_graph", True)
            and not kwargs.get("output_attentions")
            and not kwargs.get("output_hidden_states")
        )

    def _decode_step(self, input_ids: torch.Tensor, position: torch.Tensor, cache: "QuantizedStaticCache") -> torch.Tensor:
        """One decode step with fixed shapes, written to be captured in a CUDA graph.

        `position` is a device tensor of shape (1,) holding the new token's position, which is also its
        cache slot (the prompt had no padding). Returns logits of shape (batch, 1, vocab).
        """
        hidden_states = self.model.embed_tokens(input_ids)
        position_ids = position.view(1, 1).expand(input_ids.shape[0], 1)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        kv_len = (position + 1).to(torch.int32)
        for decoder_layer in self.model.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                past_key_values=cache,
                decode_position=position,
                kv_len=kv_len,
            )
        return self.lm_head(self.model.norm(hidden_states))

    def _graph_decode(self, input_ids: torch.Tensor, cache: "QuantizedStaticCache") -> torch.Tensor:
        """Run a decode step by replaying a CUDA graph: one launch instead of ~1000 small kernel launches.

        At batch size 1, decoding is limited by CPU launch overhead rather than the GPU, so this is
        where most of the speedup over eager PyTorch comes from.
        """
        position = cache.get_seq_length()
        if position >= cache.max_cache_len:
            raise ValueError(f"QuantizedStaticCache is full: max_cache_len is {cache.max_cache_len}.")

        state = cache.decode_graph
        key = (input_ids.shape[0], cache.layers[0].keys.data_ptr())
        if state is None or state["key"] != key:
            static_ids = input_ids.clone()
            static_position = torch.full((1,), position, dtype=torch.long, device=input_ids.device)
            # Run the step once eagerly on a side stream: this computes this step's real result (and
            # writes its keys/values), and triggers Triton compilation/autotuning before capture.
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                logits = self._decode_step(static_ids, static_position, cache)
            torch.cuda.current_stream().wait_stream(side_stream)
            # Capture records the kernels without running them, so the cache is not written twice.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_logits = self._decode_step(static_ids, static_position, cache)
            cache.decode_graph = {"key": key, "graph": graph, "ids": static_ids, "position": static_position, "logits": static_logits}
        else:
            state["ids"].copy_(input_ids)
            state["position"].fill_(position)
            state["graph"].replay()
            # The graph overwrites its output buffer on every replay.
            logits = state["logits"].clone()

        for layer in cache.layers:
            layer.cumulative_length += 1
        return logits

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if self._can_graph_decode(input_ids, inputs_embeds, past_key_values, labels, kwargs):
            return CausalLMOutputWithPast(logits=self._graph_decode(input_ids, past_key_values), past_key_values=past_key_values)

        if isinstance(past_key_values, QuantizedStaticCache) and past_key_values.get_seq_length() == 0:
            # Prefill: decode steps can only be graph-replayed when no prompt token is padding.
            past_key_values.graph_safe = attention_mask is None or bool(attention_mask.all())

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2ForSequenceClassification(GenericForSequenceClassification, Qwen2PreTrainedModel):
    pass


class Qwen2ForTokenClassification(GenericForTokenClassification, Qwen2PreTrainedModel):
    pass


class Qwen2ForQuestionAnswering(GenericForQuestionAnswering, Qwen2PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen2PreTrainedModel",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "Qwen2RMSNorm",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
    "Qwen2ForQuestionAnswering",
]
