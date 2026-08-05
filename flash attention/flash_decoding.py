import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_decoding_stage1_kernel(
    Q,
    K,
    V,
    sm_scale,
    Mid_O,
    Mid_M,
    Mid_L,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    stride_mid_o_batch,
    stride_mid_o_head,
    stride_mid_o_seq,
    stride_mid_o_dim,
    stride_mid_m_batch,
    stride_mid_m_head,
    stride_mid_m_seq,
    stride_mid_l_batch,
    stride_mid_l_head,
    stride_mid_l_seq,
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # This kernel processes one block of K and V for a specific batch and head
    # Grid: (triton.cdiv(N_CTX, BLOCK_KV), Z * H)
    block_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)

    batch_id = batch_head_id // H
    head_id = batch_head_id % H

    # Offsets
    q_offset = batch_id * stride_q_batch + head_id * stride_q_head
    # Query length is 1, so seq offset is 0

    k_offset = batch_id * stride_k_batch + head_id * stride_k_head
    v_offset = batch_id * stride_v_batch + head_id * stride_v_head

    # The starting sequence index for this block
    start_kv = block_id * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Query (length 1)
    # Shape: (HEAD_DIM,)
    q_ptrs = Q + q_offset + offs_d * stride_q_dim
    q = tl.load(q_ptrs)

    # Load K block
    # Shape: (BLOCK_KV, HEAD_DIM)
    k_ptrs = K + k_offset + offs_kv[:, None] * stride_k_seq + offs_d[None, :] * stride_k_dim
    # Apply masking for the last block if N_CTX is not perfectly divisible by BLOCK_KV
    mask = offs_kv[:, None] < N_CTX
    k = tl.load(k_ptrs, mask=mask, other=0.0)

    # Compute Q * K^T
    qk = tl.sum(q[None, :] * k, axis=1) * sm_scale  # Shape: (BLOCK_KV,)

    # Apply sequence masking by setting out-of-bounds tokens to -inf
    qk = tl.where(offs_kv < N_CTX, qk, float("-inf"))

    # Find local max for numerical stability
    m_i = tl.max(qk, axis=0)

    # Compute exponential and sum (l_i)
    p = tl.math.exp(qk - m_i)
    p = tl.where(offs_kv < N_CTX, p, 0.0)
    l_i = tl.sum(p, axis=0)

    # Load V block
    v_ptrs = V + v_offset + offs_kv[:, None] * stride_v_seq + offs_d[None, :] * stride_v_dim
    v = tl.load(v_ptrs, mask=mask, other=0.0)

    # Compute O partial
    # p is (BLOCK_KV,), V is (BLOCK_KV, HEAD_DIM)
    out_i = tl.sum(p[:, None] * v, axis=0)  # Shape: (HEAD_DIM,)

    # Store intermediate results
    # Mid_O: (batch, head, num_blocks, head_dim)
    mid_o_offset = batch_id * stride_mid_o_batch + head_id * stride_mid_o_head + block_id * stride_mid_o_seq
    mid_o_ptrs = Mid_O + mid_o_offset + offs_d * stride_mid_o_dim
    tl.store(mid_o_ptrs, out_i)

    # Mid_M: (batch, head, num_blocks)
    mid_m_offset = batch_id * stride_mid_m_batch + head_id * stride_mid_m_head + block_id * stride_mid_m_seq
    tl.store(Mid_M + mid_m_offset, m_i)

    # Mid_L: (batch, head, num_blocks)
    mid_l_offset = batch_id * stride_mid_l_batch + head_id * stride_mid_l_head + block_id * stride_mid_l_seq
    tl.store(Mid_L + mid_l_offset, l_i)


@triton.jit
def _flash_decoding_stage2_kernel(
    Mid_O,
    Mid_M,
    Mid_L,
    Out,
    stride_mid_o_batch,
    stride_mid_o_head,
    stride_mid_o_seq,
    stride_mid_o_dim,
    stride_mid_m_batch,
    stride_mid_m_head,
    stride_mid_m_seq,
    stride_mid_l_batch,
    stride_mid_l_head,
    stride_mid_l_seq,
    stride_out_batch,
    stride_out_head,
    stride_out_seq,
    stride_out_dim,
    Z,
    H,
    num_blocks,
    HEAD_DIM: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
):
    # This kernel reduces the partial results across blocks for each batch and head
    # Grid: (Z * H,)
    batch_head_id = tl.program_id(0)

    batch_id = batch_head_id // H
    head_id = batch_head_id % H

    offs_blocks = tl.arange(0, BLOCK_NUM)
    offs_d = tl.arange(0, HEAD_DIM)

    mid_m_offset = batch_id * stride_mid_m_batch + head_id * stride_mid_m_head
    mid_l_offset = batch_id * stride_mid_l_batch + head_id * stride_mid_l_head
    mid_o_offset = batch_id * stride_mid_o_batch + head_id * stride_mid_o_head

    # Load Mid_M
    mid_m_ptrs = Mid_M + mid_m_offset + offs_blocks * stride_mid_m_seq
    mask_blocks = offs_blocks < num_blocks
    mid_m = tl.load(mid_m_ptrs, mask=mask_blocks, other=float("-inf"))

    # Global max
    m_global = tl.max(mid_m, axis=0)

    # Load Mid_L
    mid_l_ptrs = Mid_L + mid_l_offset + offs_blocks * stride_mid_l_seq
    mid_l = tl.load(mid_l_ptrs, mask=mask_blocks, other=0.0)

    # Scale l
    weight = tl.math.exp(mid_m - m_global)
    l_scaled = mid_l * weight
    l_global = tl.sum(l_scaled, axis=0)

    # Load Mid_O and reduce
    mid_o_ptrs = Mid_O + mid_o_offset + offs_blocks[:, None] * stride_mid_o_seq + offs_d[None, :] * stride_mid_o_dim
    mid_o = tl.load(mid_o_ptrs, mask=mask_blocks[:, None], other=0.0)

    # Scale and sum Mid_O
    out = tl.sum(mid_o * weight[:, None], axis=0) / l_global

    # Store to final Out
    out_offset = batch_id * stride_out_batch + head_id * stride_out_head
    # Output seq len is 1, so out_seq is 0
    out_ptrs = Out + out_offset + offs_d * stride_out_dim
    tl.store(out_ptrs, out)


def flash_decode(q, k, v, sm_scale=None):
    """
    Computes flash decoding attention for seq_len_q = 1.
    Args:
        q: (batch_size, num_heads, 1, head_dim)
        k: (batch_size, num_heads, seq_len_kv, head_dim)
        v: (batch_size, num_heads, seq_len_kv, head_dim)
    Returns:
        out: (batch_size, num_heads, 1, head_dim)
    """
    assert q.shape[2] == 1, "Flash decoding expects query sequence length to be 1."
    assert k.shape == v.shape, "K and V must have the same shape."

    Z, H, _, HEAD_DIM = q.shape
    _, _, N_CTX, _ = k.shape

    BLOCK_KV = 128
    num_blocks = triton.cdiv(N_CTX, BLOCK_KV)

    # Ensure next power of 2 for BLOCK_NUM in stage 2 (required by triton reduction if we don't loop,
    # but here we load all blocks at once in stage 2, so BLOCK_NUM must be >= num_blocks and a power of 2)
    BLOCK_NUM = triton.next_power_of_2(num_blocks) if num_blocks > 0 else 1

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Intermediate buffers
    mid_o = torch.empty((Z, H, num_blocks, HEAD_DIM), dtype=torch.float32, device=q.device)
    mid_m = torch.empty((Z, H, num_blocks), dtype=torch.float32, device=q.device)
    mid_l = torch.empty((Z, H, num_blocks), dtype=torch.float32, device=q.device)

    # Output buffer
    out = torch.empty_like(q)

    grid1 = (num_blocks, Z * H)
    _flash_decoding_stage1_kernel[grid1](
        q,
        k,
        v,
        sm_scale,
        mid_o,
        mid_m,
        mid_l,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_m.stride(0),
        mid_m.stride(1),
        mid_m.stride(2),
        mid_l.stride(0),
        mid_l.stride(1),
        mid_l.stride(2),
        Z,
        H,
        N_CTX,
        HEAD_DIM,
        BLOCK_KV=BLOCK_KV,
    )

    grid2 = (Z * H,)
    _flash_decoding_stage2_kernel[grid2](
        mid_o,
        mid_m,
        mid_l,
        out,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_m.stride(0),
        mid_m.stride(1),
        mid_m.stride(2),
        mid_l.stride(0),
        mid_l.stride(1),
        mid_l.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        Z,
        H,
        num_blocks,
        HEAD_DIM,
        BLOCK_NUM=BLOCK_NUM,
    )

    return out


def test_flash_decoding():
    torch.manual_seed(42)
    Z = 2  # Batch size
    H = 8  # Num heads
    HEAD_DIM = 64
    N_CTX = 1000  # Seq len KV

    # Query length is 1
    q = torch.randn((Z, H, 1, HEAD_DIM), dtype=torch.float16, device="cuda")
    k = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")
    v = torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda")

    # PyTorch SDPA reference
    import torch.nn.functional as F

    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Triton Flash Decoding
    tri_out = flash_decode(q, k, v).to(torch.float16)

    print("Testing Flash Decoding...")
    print(f"Shapes -> Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    print(f"Max diff: {(ref_out - tri_out).abs().max().item():.6f}")

    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=1e-2), "Outputs do not match!"
    print("PASSED!")


if __name__ == "__main__":
    test_flash_decoding()
