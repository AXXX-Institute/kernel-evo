import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_dpfp_associate_kernel(
    Q_ptr, W_MEM_ptr, Z_ptr, OUT_ptr,
    stride_qb, stride_qs, stride_qd,
    stride_wb, stride_wh, stride_wk, stride_wv,
    stride_zb, stride_zh, stride_zk,
    stride_ob, stride_os, stride_od,
    n_heads, d_mem_head, d_key_head, d_model_head, seq_len, nu,
    USE_DENOM: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_V: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    batch_idx = pid_bh // n_heads
    head_idx = pid_bh % n_heads
    
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < seq_len

    # Load Q for this block: (BLOCK_S, d_mem_head)
    # d_mem_head is small (e.g., 8), so we use a fixed power-of-2 for loading
    d_mem_offsets = tl.arange(0, 16)
    q_ptr = Q_ptr + batch_idx * stride_qb + s_offsets[:, None] * stride_qs + (head_idx * d_mem_head + d_mem_offsets[None, :])
    q = tl.load(q_ptr, mask=(mask_s[:, None]) & (d_mem_offsets[None, :] < d_mem_head), other=0.0).to(tl.float32)
    
    x_pos = tl.maximum(0.0, q)
    x_neg = tl.maximum(0.0, -q)
    
    # Construct mq = x_repeat * x_rolled
    # x_cat is [x_pos, x_neg] of length 2 * d_mem_head
    # mq has length nu * (2 * d_mem_head)
    mq = tl.zeros((BLOCK_S, 128), dtype=tl.float32)
    norm_sq = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for j in range(nu):
        shift = j + 1
        for i in range(d_mem_head):
            # x_repeat part
            v_p = tl.sum(tl.where(d_mem_offsets == i, x_pos, 0.0), axis=1)
            v_n = tl.sum(tl.where(d_mem_offsets == i, x_neg, 0.0), axis=1)

            # x_rolled part: index (i - shift) % (2 * d_mem_head)
            idx_p_roll = (i - shift) % (2 * d_mem_head)
            idx_n_roll = (i + d_mem_head - shift) % (2 * d_mem_head)

            # Get rolled values from x_pos/x_neg
            r_p = tl.where(idx_p_roll < d_mem_head, 
                           tl.sum(tl.where(d_mem_offsets == idx_p_roll, x_pos, 0.0), axis=1),
                           tl.sum(tl.where(d_mem_offsets == (idx_p_roll - d_mem_head), x_neg, 0.0), axis=1))
            
            r_n = tl.where(idx_n_roll < d_mem_head, 
                           tl.sum(tl.where(d_mem_offsets == idx_n_roll, x_pos, 0.0), axis=1),
                           tl.sum(tl.where(d_mem_offsets == (idx_n_roll - d_mem_head), x_neg, 0.0), axis=1))

            m_p = v_p * r_p
            m_n = v_n * r_n
            
            norm_sq += m_p * m_p + m_n * m_n
            
            idx_mq_p = j * (2 * d_mem_head) + i
            idx_mq_n = j * (2 * d_mem_head) + i + d_mem_head
            mq = tl.where((tl.arange(0, 128) == idx_mq_p)[None, :], m_p[:, None], mq)
            mq = tl.where((tl.arange(0, 128) == idx_mq_n)[None, :], m_n[:, None], mq)

    inv_norm = tl.extra.cuda.libdevice.rsqrt(norm_sq + 1e-12)
    mq = mq * inv_norm[:, None]

    denom = 1e-5
    if USE_DENOM:
        z_offs = tl.arange(0, 128)
        z_ptr = Z_ptr + batch_idx * stride_zb + head_idx * stride_zh + z_offs
        z_val = tl.load(z_ptr, mask=z_offs < d_key_head, other=0.0).to(tl.float32)
        denom += tl.sum(mq * z_val[None, :], axis=1)

    for v_start in range(0, d_model_head, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        k_offs = tl.arange(0, 128)
        w_ptr = W_MEM_ptr + batch_idx * stride_wb + head_idx * stride_wh + k_offs[:, None] * stride_wk + v_offs[None, :]
        w = tl.load(w_ptr, mask=(k_offs[:, None] < d_key_head) & (v_offs[None, :] < d_model_head), other=0.0).to(tl.float32)
        
        out_chunk = tl.dot(mq.to(tl.float16), w.to(tl.float16)).to(tl.float32)
        if USE_DENOM:
            out_chunk = out_chunk / denom[:, None]
        
        out_ptr = OUT_ptr + batch_idx * stride_ob + s_offsets[:, None] * stride_os + head_idx * d_model_head + v_offs[None, :]
        tl.store(out_ptr, out_chunk.to(OUT_ptr.dtype.element_ty), mask=(mask_s[:, None]) & (v_offs[None, :] < d_model_head))

class ModelNew(nn.Module):
    def __init__(self, d_model, d_mem, n_heads, use_denom, nu, batch_size=1):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.use_denom = use_denom
        self.nu = nu
        self.d_key_head = (2 * nu * d_mem) // n_heads
        self.d_model_head = d_model // n_heads
        self.d_mem_head = d_mem // n_heads

        self.W_mq = nn.Linear(d_model, d_mem, bias=False)
        self.register_buffer("W_mem", torch.randn(batch_size, n_heads, self.d_key_head, self.d_model_head) * 0.01)
        if self.use_denom:
            self.register_buffer("z", torch.ones(batch_size, n_heads, self.d_key_head))

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        q = self.W_mq(hidden_states)
        output = torch.empty((bsz, seq_len, self.d_model), device=hidden_states.device, dtype=hidden_states.dtype)
        
        grid = (bsz * self.n_heads, triton.cdiv(seq_len, 32))
        fused_dpfp_associate_kernel[grid](
            q, self.W_mem, self.z if self.use_denom else q,
            output,
            q.stride(0), q.stride(1), q.stride(2),
            self.W_mem.stride(0), self.W_mem.stride(1), self.W_mem.stride(2), self.W_mem.stride(3),
            self.z.stride(0) if self.use_denom else 0, self.z.stride(1) if self.use_denom else 0, self.z.stride(2) if self.use_denom else 0,
            output.stride(0), output.stride(1), output.stride(2),
            self.n_heads, self.d_mem_head, self.d_key_head, self.d_model_head, seq_len, self.nu,
            USE_DENOM=self.use_denom, BLOCK_S=32, BLOCK_V=64
        )
        return output