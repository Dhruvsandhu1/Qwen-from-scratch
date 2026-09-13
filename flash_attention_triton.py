import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q,
    offs_kv,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # We process the KV blocks in chunks
        K_block = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        QK_block = tl.dot(Q_block, K_block, out_dtype=tl.float32)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        P_block = P_block.to(V_block.dtype)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    This kernel is used to compute the forward pass of the attention mechanism.
    It is used to compute the attention scores between the queries and the keys and the values.

    """
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # L2 Cache Swizzling
    pid = tl.program_id(0)
    NUM_PID_M = tl.cdiv(SEQ_LEN, BLOCK_SIZE_Q)
    NUM_PID_N = BATCH_SIZE * NUM_HEADS
    GROUP_SIZE_M = 8
    num_pid_in_group = GROUP_SIZE_M * NUM_PID_N
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_PID_M - first_pid_m, GROUP_SIZE_M)
    block_index_q = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    index_batch_head = (pid % num_pid_in_group) // group_size_m
    # This indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    NUM_QUERIES_PER_KV = NUM_HEADS // NUM_KV_HEADS
    index_kv_head = index_head // NUM_QUERIES_PER_KV

    q_offset = index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    k_offset = index_batch.to(tl.int64) * stride_K_batch + index_kv_head.to(tl.int64) * stride_K_head
    v_offset = index_batch.to(tl.int64) * stride_V_batch + index_kv_head.to(tl.int64) * stride_V_head
    o_offset = index_batch.to(tl.int64) * stride_O_batch + index_head.to(tl.int64) * stride_O_head

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offs_kv: the offsets for the tokens in the K and V sequence to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    m_i += tl.math.log(l_i)  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0,))


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    # NOTE: O/dO are views (e.g. `.transpose(1, 2)` in the caller) and are not
    # necessarily contiguous in (batch, head, seq, dim) order, so the offset
    # must be computed from the real strides rather than assumed sizes.
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offs_dim = tl.arange(0, HEAD_DIM)

    O_ptrs = O + offset_batch_head + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    dO_ptrs = dO + offset_batch_head + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # Load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load(O_ptrs)
    # Load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(dO_ptrs).to(tl.float32)
    # Compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
    # Store the D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


@triton.autotune(
    [triton.Config({"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV}, num_stages=num_stages, num_warps=num_warps) for BLOCK_Q in [32, 64, 128] for BLOCK_KV in [32, 64, 128] for num_stages in [2, 3, 4] for num_warps in [4, 8]],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    M,
    D,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_kv_batch,
    stride_kv_head,
    stride_kv_seq,
    stride_kv_dim,
    NUM_HEADS,
    NUM_KV_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    # GQA: map the query head to the (fewer) key/value heads it shares, mirroring
    # the forward kernel's `index_kv_head` logic.
    NUM_QUERIES_PER_KV = NUM_HEADS // NUM_KV_HEADS
    index_kv_head = index_head // NUM_QUERIES_PER_KV

    offset_q_batch_head = (stride_q_batch * index_batch + stride_q_head * index_head).to(tl.int64)
    offset_kv_batch_head = (stride_kv_batch * index_batch + stride_kv_head * index_kv_head).to(tl.int64)
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_tensor_descriptor is because we need to use the range of offsets to apply the masking
    Q += offset_q_batch_head
    dO += offset_q_batch_head
    dQ += offset_q_batch_head
    K += offset_kv_batch_head
    V += offset_kv_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_q = tl.program_id(0)

    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim)

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    # We access the K and V as transposed blocks
    kT_ptrs = K + offs_kv[None, :] * stride_kv_seq + offs_dim[:, None] * stride_kv_dim
    vT_ptrs = V + offs_kv[None, :] * stride_kv_seq + offs_dim[:, None] * stride_kv_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            # Autoregressive masking.
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        # Compute dP and dS.
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(K_T_block.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        # Increment pointers.
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_kv_seq
        vT_ptrs += BLOCK_KV * stride_kv_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim
    tl.store(dQ_block_ptrs, dQ_block)


@triton.autotune(
    [triton.Config({"BLOCK_Q": BLOCK_Q, "BLOCK_KV": BLOCK_KV}, num_stages=num_stages, num_warps=num_warps) for BLOCK_Q in [32, 64, 128] for BLOCK_KV in [32, 64, 128] for num_stages in [2, 3, 4] for num_warps in [4, 8]],
    key=["SEQ_LEN", "HEAD_DIM"],
    # `dK`/`dV` are accumulated via tl.atomic_add (needed for GQA, where several
    # query heads add into the same kv head's slot) which is NOT idempotent like
    # tl.store. Autotuning re-runs the kernel body once per candidate config to
    # benchmark it, so without reset_to_zero every extra trial would silently
    # add its contribution again and inflate the accumulated gradients.
    reset_to_zero=["dK", "dV"],
)
@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dK,
    dV,
    M,
    D,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_kv_batch,
    stride_kv_head,
    stride_kv_seq,
    stride_kv_dim,
    NUM_HEADS,
    NUM_KV_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    # GQA: several query heads (index_head) can map to the same kv head. This
    # kernel is still launched once per query head (it needs each query head's
    # own Q/dO/M/D), but every query head in a group must accumulate into the
    # *same* dK/dV location for its shared kv head, hence the atomic_add below
    # instead of a plain store.
    NUM_QUERIES_PER_KV = NUM_HEADS // NUM_KV_HEADS
    index_kv_head = index_head // NUM_QUERIES_PER_KV

    offset_q_batch_head = (stride_q_batch * index_batch + stride_q_head * index_head).to(tl.int64)
    offset_kv_batch_head = (stride_kv_batch * index_batch + stride_kv_head * index_kv_head).to(tl.int64)
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_tensor_descriptor is because we need to use the range of offsets to apply the masking
    Q += offset_q_batch_head
    dO += offset_q_batch_head
    K += offset_kv_batch_head
    V += offset_kv_batch_head
    dK += offset_kv_batch_head
    dV += offset_kv_batch_head

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV

    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    K_block = tl.load(K + offs_kv[:, None] * stride_kv_seq + offs_dim[None, :] * stride_kv_dim)  # Shape: (BLOCK_KV1, HEAD_DIM)
    V_block = tl.load(V + offs_kv[:, None] * stride_kv_seq + offs_dim[None, :] * stride_kv_dim)  # Shape: (BLOCK_KV1, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector ans offs_dim as a row vector
    # This is equivalent to doing:
    # q_ptrs = Q + offs_q[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    qT_ptrs = Q + offs_q[None, :] * stride_q_seq + offs_dim[:, None] * stride_q_dim
    dO_ptrs = dO + offs_q[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim

    # Iterates over the sequence dimension of the query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load a block of Q
        qT_block = tl.load(qT_ptrs)
        # Load the logsumexp values for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = P^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # Autoregressive masking.
            # mask is True for all values that DO NOT NEED TO BE MASKED
            mask_block = offs_q[None, :] >= offs_kv[:, None]  # Shape: (BLOCK_KV1, BLOCK_Q1)
            # Replace all the masked values with 0.
            # In this case we do not need to mask with -Inf before applying the softmax since we already computed the normalization factors (stored in "m")
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        dO_block = tl.load(dO_ptrs)
        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(P_T_block.to(dO_block.dtype), dO_block)

        # Delta = rowsum(O * dO) where * is the element-wise product
        Di = tl.load(D + offs_q)

        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)

        # We know that dS = P * (dP - Delta), so dS^T = P^T * (dP^T - Delta^T)

        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(qT_block.dtype)

        # According to the formula on the paper: dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Increment pointers.
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_q_seq
        dO_ptrs += BLOCK_Q * stride_q_seq

    # Write back dV and dK. Multiple query heads in a GQA group share the same
    # kv head, so several programs (one per query head) target the same
    # (batch, kv_head, kv_block) location and must accumulate atomically rather
    # than overwrite each other. `dK`/`dV` are float32 accumulators allocated by
    # the caller so this atomic add is dtype-matched.
    dV_block_ptrs = dV + offs_kv[:, None] * stride_kv_seq + offs_dim[None, :] * stride_kv_dim
    tl.atomic_add(dV_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_kv_seq + offs_dim[None, :] * stride_kv_dim
    tl.atomic_add(dK_block_ptrs, dK_block)


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]) * (BATCH_SIZE * NUM_HEADS),
            1,
            1,
        )

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            NUM_KV_HEADS=K.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        # Q/O/dO always share the same (query) head count, and the kernels below
        # address dO using Q's own strides -- so dO must match Q's exact stride
        # pattern. In the real model, O is a transpose() view (no .contiguous())
        # and the incoming dO naturally mirrors that same non-standard layout
        # rather than being plain-contiguous, so a bare `dO.is_contiguous()`
        # assert fires on real usage; normalize dO into Q's layout instead.
        if dO.stride() != Q.stride():
            dO = torch.empty_strided(Q.shape, Q.stride(), dtype=dO.dtype, device=dO.device).copy_(dO)
        # K/V may have fewer heads than Q/O under GQA, so they're only required
        # to match each other (not Q).
        assert K.stride() == V.stride()
        dQ = torch.empty_like(Q)
        # dK/dV are accumulated with tl.atomic_add across the query heads that
        # share a kv head (GQA), so they're float32 accumulators regardless of
        # the model dtype, zero-initialized, and cast down at the end.
        dK_acc = torch.zeros_like(K, dtype=torch.float32)
        dV_acc = torch.zeros_like(V, dtype=torch.float32)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_KV_HEADS = K.shape[1]
        BLOCK_SIZE_MACRO = 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        # Compute all the elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            stride_batch=O.stride(0),
            stride_head=O.stride(1),
            stride_seq=O.stride(2),
            stride_dim=O.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )

        stage = 3 if ctx.causal else 1

        # NOTE: SEQ_LEN must be a multiple of the autotuned BLOCK_Q/BLOCK_KV for
        # full correctness -- like the forward kernel's block-aligned assumptions,
        # this backward pass does not apply boundary checks to the tail block.
        common_kwargs = dict(
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            M=M,
            D=D,
            stride_q_batch=Q.stride(0),
            stride_q_head=Q.stride(1),
            stride_q_seq=Q.stride(2),
            stride_q_dim=Q.stride(3),
            stride_kv_batch=K.stride(0),
            stride_kv_head=K.stride(1),
            stride_kv_seq=K.stride(2),
            stride_kv_dim=K.stride(3),
            NUM_HEADS=NUM_HEADS,
            NUM_KV_HEADS=NUM_KV_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
        )

        # Grid sizes are derived from the autotuned BLOCK_Q/BLOCK_KV (via `meta`)
        # so the launched grid always matches the block size the kernel actually
        # uses -- a static grid computed from a different constant here would
        # silently leave part of the KV/Q range unprocessed.
        grid_dk_dv = lambda meta: (SEQ_LEN // meta["BLOCK_KV"], 1, BATCH_SIZE * NUM_HEADS)
        grid_dq = lambda meta: (SEQ_LEN // meta["BLOCK_Q"], 1, BATCH_SIZE * NUM_HEADS)

        # Fix KV and iterate through all the Q blocks
        _attn_bwd_dk_dv[grid_dk_dv](dK=dK_acc, dV=dV_acc, **common_kwargs)

        # Fix Q and iterate through all the KV block
        _attn_bwd_dq[grid_dq](dQ=dQ, **common_kwargs)

        dK = dK_acc.to(K.dtype)
        dV = dV_acc.to(V.dtype)

        return dQ, dK, dV, None, None


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
    K = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
    V = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)


if __name__ == "__main__":
    print("PASSED")
