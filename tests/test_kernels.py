"""Triton kernels against PyTorch references."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]

HEADS, KV_HEADS, HEAD_DIM = 16, 2, 128  # Qwen2.5-3B shapes (grouped-query attention)
SCALE = HEAD_DIM**-0.5


def reference_attention(q, k, v, causal):
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal, scale=SCALE)


def random_qkv(seq_len, batch=2, requires_grad=False):
    torch.manual_seed(0)
    q = torch.randn(batch, HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, KV_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, KV_HEADS, seq_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return [t.requires_grad_(requires_grad) for t in (q, k, v)]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seq_len", [1, 7, 100, 333])
def test_prefill_forward(causal, seq_len):
    from triton_custom.flash_attention_triton import TritonAttention

    q, k, v = random_qkv(seq_len)
    out = TritonAttention.apply(q, k, v, causal, SCALE)
    torch.testing.assert_close(out.float(), reference_attention(q, k, v, causal), atol=2e-2, rtol=0)


def test_prefill_backward():
    from triton_custom.flash_attention_triton import TritonAttention

    q, k, v = random_qkv(256, batch=1, requires_grad=True)
    grad_out = torch.randn_like(q)
    reference_attention(q, k, v, True).backward(grad_out.float())
    expected = [t.grad.float() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    TritonAttention.apply(q, k, v, True, SCALE).backward(grad_out)
    for name, t, ref in zip("qkv", (q, k, v), expected):
        rel_err = ((t.grad.float() - ref).abs().max() / ref.abs().max()).item()
        assert rel_err < 2e-2, f"d{name} relative error {rel_err:.3g}"


def test_prefill_backward_rejects_unsupported_length():
    from triton_custom.flash_attention_triton import TritonAttention

    q, k, v = random_qkv(200, batch=1, requires_grad=True)
    with pytest.raises(ValueError, match="multiple of 128"):
        TritonAttention.apply(q, k, v, True, SCALE)


@pytest.mark.parametrize("kv_len", [1, 5, 129, 1000])
def test_flash_decode(kv_len):
    from triton_custom.flash_decoding_triton import flash_decode

    torch.manual_seed(0)
    q = torch.randn(2, HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, KV_HEADS, kv_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, KV_HEADS, kv_len, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    out = flash_decode(q, k, v, sm_scale=SCALE)
    torch.testing.assert_close(out.float(), reference_attention(q, k, v, False), atol=2e-2, rtol=0)


@pytest.mark.parametrize("kv_len", [1, 5, 129, 1000])
def test_flash_decode_int8_with_device_length(kv_len):
    """INT8 keys/values in a larger buffer, with the valid length read from device memory (CUDA-graph mode)."""
    from triton_custom.flash_decoding_triton import flash_decode

    torch.manual_seed(0)
    capacity = kv_len + 300
    q = torch.randn(2, HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, KV_HEADS, capacity, HEAD_DIM, device="cuda")
    v = torch.randn(2, KV_HEADS, capacity, HEAD_DIM, device="cuda")
    k_scale, v_scale = k.abs().amax(-1, keepdim=True) / 127, v.abs().amax(-1, keepdim=True) / 127
    k_q, v_q = torch.round(k / k_scale).to(torch.int8), torch.round(v / v_scale).to(torch.int8)
    length = torch.tensor([kv_len], device="cuda", dtype=torch.int32)

    out = flash_decode(q, k_q, v_q, sm_scale=SCALE, key_scales=k_scale, value_scales=v_scale, kv_len=length)
    expected = reference_attention(q, (k_q.float() * k_scale)[:, :, :kv_len], (v_q.float() * v_scale)[:, :, :kv_len], False)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=0)
