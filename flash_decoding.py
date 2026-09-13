import torch
import triton
import triton.language as tl
import math


# Cache for intermediate buffers to avoid torch.empty overhead during decoding
_buffer_cache = {}


def _get_buffers(Z, H, SPLIT_K, HEAD_DIM, dtype, device):
    padded_blocks = max(1, triton.next_power_of_2(SPLIT_K)) if SPLIT_K > 0 else 1
    key = (Z, H, HEAD_DIM, dtype, device, padded_blocks)

    if key not in _buffer_cache:
        _buffer_cache[key] = {"mid_o": torch.empty((Z, H, padded_blocks, HEAD_DIM), dtype=torch.float32, device=device), "mid_m": torch.empty((Z, H, padded_blocks), dtype=torch.float32, device=device), "mid_l": torch.empty((Z, H, padded_blocks), dtype=torch.float32, device=device)}

    cache = _buffer_cache[key]
    # `out` is intentionally NOT cached/reused: every call returns it directly to
    # the caller, and every decoder layer shares the same (Z, H, HEAD_DIM, dtype)
    # key, so a cached `out` would have one layer's result silently overwritten
    # by the next layer's call as soon as anyone stops copying it immediately.
    out = torch.empty((Z, H, 1, HEAD_DIM), dtype=dtype, device=device)
    return cache["mid_o"], cache["mid_m"], cache["mid_l"], out


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _flash_decoding_stage1_kernel(
    Q,
    K,
    V,
    sm_scale,
    Mid_O,
    Mid_M,
    Mid_L,
    K_scale,
    V_scale,
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
    stride_ks_batch,
    stride_ks_head,
    stride_ks_seq,
    stride_ks_dim,
    stride_vs_batch,
    stride_vs_head,
    stride_vs_seq,
    stride_vs_dim,
    IS_INT8: tl.constexpr,
    Z,
    H,
    H_KV,
    N_CTX,
    SPLIT_K,
    total_kv_blocks,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # Grid: (SPLIT_K, Z * H)
    split_k_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)

    batch_id = batch_head_id // H
    head_id = batch_head_id % H

    num_queries_per_kv = H // H_KV
    kv_head_id = head_id // num_queries_per_kv

    # Offsets
    q_offset = batch_id * stride_q_batch + head_id * stride_q_head
    k_offset = batch_id * stride_k_batch + kv_head_id * stride_k_head
    v_offset = batch_id * stride_v_batch + kv_head_id * stride_v_head

    if IS_INT8:
        ks_offset = batch_id * stride_ks_batch + kv_head_id * stride_ks_head
        vs_offset = batch_id * stride_vs_batch + kv_head_id * stride_vs_head

    offs_d = tl.arange(0, HEAD_DIM)

    # Load Query (length 1)
    q_ptrs = Q + q_offset + offs_d * stride_q_dim
    q = tl.load(q_ptrs)
    q = q * sm_scale

    # Running sums in SRAM/Registers
    m_i = float("-inf")
    l_i = 0.0
    out_i = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Determine loop bounds for this split chunk
    blocks_per_split = (total_kv_blocks + SPLIT_K - 1) // SPLIT_K
    start_block = split_k_id * blocks_per_split
    end_block = tl.minimum(start_block + blocks_per_split, total_kv_blocks)

    # Initialize sequence offsets and pointers OUTSIDE the loop for maximum speed
    start_kv = start_block * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)
    k_ptrs = K + k_offset + offs_kv[:, None] * stride_k_seq + offs_d[None, :] * stride_k_dim
    v_ptrs = V + v_offset + offs_kv[:, None] * stride_v_seq + offs_d[None, :] * stride_v_dim

    if IS_INT8:
        ks_ptrs = K_scale + ks_offset + offs_kv * stride_ks_seq
        vs_ptrs = V_scale + vs_offset + offs_kv * stride_vs_seq

    for block_idx in range(start_block, end_block):
        # Load K block
        mask = offs_kv[:, None] < N_CTX
        if IS_INT8:
            k_int8 = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.int8)
            ks = tl.load(ks_ptrs, mask=offs_kv < N_CTX, other=0.0)
            k = k_int8.to(tl.float32) * ks[:, None]
        else:
            k = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Compute Q * K^T
        qk = tl.sum(q[None, :] * k, axis=1)  # Shape: (BLOCK_KV,)
        qk = tl.where(offs_kv < N_CTX, qk, float("-inf"))

        # Online softmax mathematics
        m_ij = tl.max(qk, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)

        # Update coefficients
        alpha = tl.math.exp2((m_i - m_i_new) * 1.4426950408889634)
        p = tl.math.exp2((qk - m_i_new) * 1.4426950408889634)
        p = tl.where(offs_kv < N_CTX, p, 0.0)

        l_ij = tl.sum(p, axis=0)
        l_i_new = l_i * alpha + l_ij

        # Load V block and update output
        if IS_INT8:
            v_int8 = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.int8)
            vs = tl.load(vs_ptrs, mask=offs_kv < N_CTX, other=0.0)
            v = v_int8.to(tl.float32) * vs[:, None]
        else:
            v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)

        out_i = out_i * alpha + tl.sum(p[:, None] * v, axis=0)

        m_i = m_i_new
        l_i = l_i_new

        # Advance pointers for the next iteration (much faster than recomputing)
        k_ptrs += BLOCK_KV * stride_k_seq
        v_ptrs += BLOCK_KV * stride_v_seq
        if IS_INT8:
            ks_ptrs += BLOCK_KV * stride_ks_seq
            vs_ptrs += BLOCK_KV * stride_vs_seq
        offs_kv += BLOCK_KV

    # Store intermediate results to global memory ONCE at the end
    mid_o_offset = batch_id * stride_mid_o_batch + head_id * stride_mid_o_head + split_k_id * stride_mid_o_seq
    mid_o_ptrs = Mid_O + mid_o_offset + offs_d * stride_mid_o_dim
    tl.store(mid_o_ptrs, out_i)

    mid_m_offset = batch_id * stride_mid_m_batch + head_id * stride_mid_m_head + split_k_id * stride_mid_m_seq
    tl.store(Mid_M + mid_m_offset, m_i)

    mid_l_offset = batch_id * stride_mid_l_batch + head_id * stride_mid_l_head + split_k_id * stride_mid_l_seq
    tl.store(Mid_L + mid_l_offset, l_i)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["HEAD_DIM"],
)
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

    # Scale l using native exp2
    weight = tl.math.exp2((mid_m - m_global) * 1.4426950408889634)
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


def flash_decode(q, k, v, sm_scale=None, key_scales=None, value_scales=None):
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

    IS_INT8 = key_scales is not None and value_scales is not None

    Z, H, _, HEAD_DIM = q.shape
    _, H_KV, N_CTX, _ = k.shape

    BLOCK_KV = 128
    total_kv_blocks = triton.cdiv(N_CTX, BLOCK_KV)

    # Dynamic hardware-aware SPLIT_K heuristic
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count

    # We want to launch enough blocks to keep all SMs busy.
    # A good target is ~4 blocks per SM to hide latency.
    target_grid_size = num_sms * 4

    # The stage 1 grid size is (SPLIT_K, Z * H)
    desired_split_k = target_grid_size // (Z * H)

    # Clamp SPLIT_K between 1 and total_kv_blocks
    SPLIT_K = min(total_kv_blocks, max(1, desired_split_k))

    # Ensure next power of 2 for BLOCK_NUM in stage 2
    BLOCK_NUM = triton.next_power_of_2(SPLIT_K) if SPLIT_K > 0 else 1

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Fetch intermediate buffers from cache
    mid_o, mid_m, mid_l, out = _get_buffers(Z, H, SPLIT_K, HEAD_DIM, q.dtype, q.device)

    grid1 = (SPLIT_K, Z * H)
    _flash_decoding_stage1_kernel[grid1](
        q,
        k,
        v,
        sm_scale,
        mid_o,
        mid_m,
        mid_l,
        key_scales,
        value_scales,
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
        key_scales.stride(0) if IS_INT8 else 0,
        key_scales.stride(1) if IS_INT8 else 0,
        key_scales.stride(2) if IS_INT8 else 0,
        key_scales.stride(3) if IS_INT8 else 0,
        value_scales.stride(0) if IS_INT8 else 0,
        value_scales.stride(1) if IS_INT8 else 0,
        value_scales.stride(2) if IS_INT8 else 0,
        value_scales.stride(3) if IS_INT8 else 0,
        IS_INT8,
        Z,
        H,
        H_KV,
        N_CTX,
        SPLIT_K,
        total_kv_blocks,
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
        num_blocks=SPLIT_K,
        HEAD_DIM=HEAD_DIM,
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
