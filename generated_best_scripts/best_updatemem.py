import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_dpfp_norm_kernel(
    K_ptr, MK_ptr, MK_NORM_SQ_ptr,
    B, H, S, D_HEAD, NU,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_mkb, stride_mkh, stride_mks, stride_mkk,
    stride_nsqb, stride_nsqh, stride_nsqs,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr
):
    # Grid: (B * H, (S + BLOCK_S - 1) // BLOCK_S)
    bh_idx = tl.program_id(0)
    s_start = tl.program_id(1) * BLOCK_S
    b_idx = bh_idx // H
    h_idx = bh_idx % H

    s_offsets = s_start + tl.arange(0, BLOCK_S)
    d_offsets = tl.arange(0, BLOCK_D)
    
    # Load K: (BLOCK_S, D_HEAD)
    k_mask = (s_offsets[:, None] < S) & (d_offsets[None, :] < D_HEAD)
    k = tl.load(K_ptr + b_idx * stride_kb + h_idx * stride_kh + s_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd, mask=k_mask, other=0.0)
    
    # Compute x = [relu(k), relu(-k)]
    xp = tl.maximum(0.0, k)
    xn = tl.maximum(0.0, -k)
    
    # We need to compute mk = [x * x.roll(j) for j in 1..NU]
    # Total features DK = 2 * NU * D_HEAD
    # For simplicity and speed, we compute and store directly
    # We also compute norm_sq for normalization
    
    # x is (BLOCK_S, 2 * D_HEAD)
    # We'll handle the roll logic by indexing
    norm_sq = tl.zeros([BLOCK_S], dtype=tl.float32)
    
    for j in range(1, NU + 1):
        # Roll x by j: x_rolled = cat([xp, xn]).roll(j)
        # Since we are in a kernel, we can just compute the product for each d
        for d in range(D_HEAD):
            # Feature d in xp
            val_xp = xp[:, d]
            # Feature d in xn
            val_xn = xn[:, d]
            
            # Roll logic for index (d-j) % (2*D_HEAD)
            # This is complex to do fully in Triton without large registers
            # Simplified: we only need to store the result and sum squares
            pass

@triton.jit
def update_mem_kernel(
    MK_ptr, MV_ptr, MB_ptr, WMEM_ptr, Z_ptr, NEW_INFO_COEF_ptr,
    B, H, S, DK, DV, USE_DENOM: tl.constexpr,
    stride_mkb, stride_mkh, stride_mks, stride_mkk,
    stride_mvb, stride_mvh, stride_mvs, stride_mvd,
    stride_mbb, stride_mbh, stride_mbs, stride_mbd,
    stride_wb, stride_wh, stride_wk, stride_wv,
    stride_zb, stride_zh, stride_zk,
    BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_S: tl.constexpr
):
    bh_idx = tl.program_id(0)
    k_idx = tl.program_id(1)
    v_idx = tl.program_id(2)
    b_idx = bh_idx // H
    h_idx = bh_idx % H

    k_offsets = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offsets = v_idx * BLOCK_V + tl.arange(0, BLOCK_V)
    
    acc = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
    z_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for s_seg in range(0, S, BLOCK_S):
        s_offsets = s_seg + tl.arange(0, BLOCK_S)
        s_mask = s_offsets < S
        
        mk = tl.load(MK_ptr + b_idx * stride_mkb + h_idx * stride_mkh + s_offsets[None, :] * stride_mks + k_offsets[:, None] * stride_mkk, mask=(s_mask[None, :]) & (k_offsets[:, None] < DK), other=0.0)
        mv = tl.load(MV_ptr + b_idx * stride_mvb + h_idx * stride_mvh + s_offsets[:, None] * stride_mvs + v_offsets[None, :] * stride_mvd, mask=(s_mask[:, None]) & (v_offsets[None, :] < DV), other=0.0)
        mb = tl.load(MB_ptr + b_idx * stride_mbb + h_idx * stride_mbh + s_offsets[:, None] * stride_mbs + v_offsets[None, :] * stride_mbd, mask=(s_mask[:, None]) & (v_offsets[None, :] < DV), other=0.0)
        
        # Gating
        mv_gated = mv * tl.sigmoid(mb.to(tl.float32))
        acc += tl.dot(mk.to(tl.float16), mv_gated.to(tl.float16))
        
        if USE_DENOM and v_idx == 0:
            # Only one block per K needs to update Z
            coef = tl.load(NEW_INFO_COEF_ptr + b_idx * stride_mkb // stride_mks + h_idx * stride_mkh // stride_mks + s_offsets, mask=s_mask, other=0.0)
            z_acc += tl.sum(mk * coef[None, :], axis=1)

    # Store W_mem
    w_ptr = WMEM_ptr + b_idx * stride_wb + h_idx * stride_wh + k_offsets[:, None] * stride_wk + v_offsets[None, :] * stride_wv
    mask = (k_offsets[:, None] < DK) & (v_offsets[None, :] < DV)
    tl.store(w_ptr, tl.load(w_ptr, mask=mask) + acc.to(tl.float16), mask=mask)
    
    if USE_DENOM and v_idx == 0:
        z_ptr = Z_ptr + b_idx * stride_zb + h_idx * stride_zh + k_offsets * stride_zk
        tl.store(z_ptr, tl.load(z_ptr, mask=k_offsets < DK) + z_acc.to(tl.float32), mask=k_offsets < DK)

class ModelNew(nn.Module):
    def __init__(self, d_model, d_mem, n_heads, use_denom, nu, correction=True, gating=True, batch_size=1):
        super().__init__()
        self.d_model, self.d_mem, self.n_heads = d_model, d_mem, n_heads
        self.use_denom, self.nu, self.correction, self.gating = use_denom, nu, correction, gating
        self.first_seg = False
        self.d_key = 2 * self.nu * (self.d_mem // self.n_heads)
        
        self.W_mk = nn.Linear(d_model, d_mem, bias=False)
        self.W_mv = nn.Linear(d_model, d_model, bias=False)
        self.W_mb = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer("W_mem", torch.randn(batch_size, n_heads, self.d_key, d_model // n_heads) * 0.01, persistent=False)
        if self.use_denom:
            self.register_buffer("z", torch.ones(batch_size, n_heads, self.d_key), persistent=False)

    def _phi(self, x):
        # Optimized DPFP in Torch for correctness, but we'll fuse the rest
        x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
        res = []
        for j in range(1, self.nu + 1):
            res.append(x * x.roll(shifts=j, dims=-1))
        return torch.cat(res, dim=-1)

    def forward(self, mem_tokens):
        bsz, seq_len, _ = mem_tokens.shape
        dv = self.d_model // self.n_heads
        
        k_raw = self.W_mk(mem_tokens).view(bsz, seq_len, self.n_heads, -1).permute(0, 2, 1, 3)
        new_mv = self.W_mv(mem_tokens).view(bsz, seq_len, self.n_heads, -1).permute(0, 2, 1, 3)
        mb_raw = self.W_mb(mem_tokens).view(bsz, seq_len, self.n_heads, -1).permute(0, 2, 1, 3)

        mk = self._phi(k_raw)
        mk = F.normalize(mk, dim=-1, p=2.0)

        if not self.first_seg:
            num = torch.matmul(mk, self.W_mem)
            if self.use_denom:
                denom = torch.einsum("ihj,ihkj->ihk", self.z, mk)[..., None] + 1e-5
                prev_mv = num / denom
                if self.correction:
                    mk_norm_sq = torch.sum(mk * mk, dim=-1)
                    new_info_coef = torch.clamp(1 - denom.squeeze(-1) / (mk_norm_sq + 1e-6), 0, 1).detach()
                else:
                    new_info_coef = torch.ones((bsz, self.n_heads, seq_len), device=mk.device)
            else:
                prev_mv = num
                new_info_coef = torch.ones((bsz, self.n_heads, seq_len), device=mk.device)
        else:
            prev_mv = torch.zeros_like(new_mv)
            new_info_coef = torch.ones((bsz, self.n_heads, seq_len), device=mk.device)

        mv = new_mv - prev_mv
        
        grid = (bsz * self.n_heads, (self.d_key + 63) // 64, (dv + 63) // 64)
        update_mem_kernel[grid](
            mk, mv, mb_raw, self.W_mem, self.z if self.use_denom else mk, new_info_coef,
            bsz, self.n_heads, seq_len, self.d_key, dv, self.use_denom,
            mk.stride(0), mk.stride(1), mk.stride(2), mk.stride(3),
            mv.stride(0), mv.stride(1), mv.stride(2), mv.stride(3),
            mb_raw.stride(0), mb_raw.stride(1), mb_raw.stride(2), mb_raw.stride(3),
            self.W_mem.stride(0), self.W_mem.stride(1), self.W_mem.stride(2), self.W_mem.stride(3),
            self.z.stride(0) if self.use_denom else 0, self.z.stride(1) if self.use_denom else 0, self.z.stride(2) if self.use_denom else 0,
            BLOCK_K=64, BLOCK_V=64, BLOCK_S=32
        )

        return self.W_mem