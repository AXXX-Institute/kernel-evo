import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_attention_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len, d_model,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_base = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_D)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load Q tile (assuming BLOCK_D == d_model for simplicity in this specific fix, 
    # but reducing BLOCK_M/N to fit shared memory)
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh + rm[:, None] * stride_qm + rk[None, :] * stride_qk
    q = tl.load(q_ptr, mask=(rm[:, None] < seq_len) & (rk[None, :] < d_model), other=0.0)

    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + rn_base
        k_ptr = K + pid_b * stride_kb + pid_h * stride_kh + rn[None, :] * stride_kn + rk[:, None] * stride_kk
        k = tl.load(k_ptr, mask=(rn[None, :] < seq_len) & (rk[:, None] < d_model), other=0.0)
        
        qk = tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(rm[:, None] < seq_len, qk, float('-inf'))
        qk = tl.where(rn[None, :] < seq_len, qk, float('-inf'))
        
        m_j = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v_ptr = V + pid_b * stride_vb + pid_h * stride_vh + rn[:, None] * stride_vn + rk[None, :] * stride_vk
        v = tl.load(v_ptr, mask=(rn[:, None] < seq_len) & (rk[None, :] < d_model), other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        
        m_i = m_new

    acc = acc / l_i[:, None]
    
    out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh + rm[:, None] * stride_om + rk[None, :] * stride_ok
    tl.store(out_ptr, acc.to(tl.float16), mask=(rm[:, None] < seq_len) & (rk[None, :] < d_model))

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_model = Q.shape
        sm_scale = 1.0 / (d_model ** 0.5)
        out = torch.empty_like(Q)

        # Reduced BLOCK_M and BLOCK_N to fit 1024-dim d_model into shared memory
        # Shared memory usage is roughly (BLOCK_M*d_model + BLOCK_N*d_model + BLOCK_M*BLOCK_N) * sizeof(fp16)
        BLOCK_M = 16
        BLOCK_N = 16
        BLOCK_D = d_model

        grid = (triton.cdiv(seq_len, BLOCK_M), num_heads, batch_size)

        _fused_attention_kernel[grid](
            Q, K, V, out,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seq_len, d_model,
            sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2
        )

        return out