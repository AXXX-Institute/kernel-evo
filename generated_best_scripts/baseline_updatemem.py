# Auto-generated seed program.
# Generated from KernelBench level=None problem_id=None
#
# IMPORTANT: this file is evaluated directly (no entrypoint wrapper).
# It must define `class ModelNew(torch.nn.Module)`.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPFP:
    """
    Deterministic Positive Feature Projection (as used in `language_modeling.py`).
    in (..., d)
    out (..., 2 * nu * d).
    """

    def __init__(self, nu: int):
        self.nu = int(nu)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        nu = self.nu
        x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)
        x_repeat = torch.cat([x] * nu, dim=-1)
        return x_repeat * x_rolled


class ModelNew(nn.Module):
    """
    Standalone benchmark wrapper for `AssociativeLayerWrapper.update_mem(...)`.

    This isolates `language_modeling.py:158-204` into a self-contained module.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        n_heads: int,
        use_denom: bool,
        nu: int,
        correction: bool = True,
        gating: bool = True,
        batch_size: int = 1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.use_denom = bool(use_denom)
        self.nu = nu
        self.correction = correction
        self.gating = gating
        self.first_seg = False  # Set to False to test the more complex path
        self.seg_num = 0

        assert self.d_model % self.n_heads == 0
        assert self.d_mem % self.n_heads == 0

        self.d_key = 2 * self.nu * (self.d_mem // self.n_heads)

        self.phi = DPFP(self.nu)
        self.W_mk = nn.Linear(self.d_model, self.d_mem, bias=False)
        self.W_mv = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_mb = nn.Linear(self.d_model, self.d_model, bias=False) # Changed from d_mem to d_model

        # Persistent memory tensors
        W_mem = torch.randn(
            batch_size,
            self.n_heads,
            self.d_key,
            self.d_model // self.n_heads,
        ) * 0.01
        self.register_buffer("W_mem", W_mem, persistent=False)

        if self.use_denom:
            z = torch.ones(batch_size, self.n_heads, self.d_key)
            self.register_buffer("z", z, persistent=False)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = x.shape
        x = x.reshape(bsz, seq_len, self.n_heads, d // self.n_heads)
        x = x.permute(0, 2, 1, 3)
        return x

    def update_mem(self, mem_tokens: torch.Tensor):
        # Isolated from `language_modeling.py:158-204`
        self.W_mem = self.W_mem.to(mem_tokens.device)
        if self.use_denom:
            self.z = self.z.to(mem_tokens.device)
        k = self._to_heads(self.W_mk(mem_tokens))  # mem_tokens has d_model features, W_mk outputs d_mem features
        mk = self.phi(k)
        mk = F.normalize(mk, dim=-1, p=2.0)

        new_mv = self._to_heads(self.W_mv(mem_tokens)) # (bsz, n_heads, num_mem_tokens, d_model // n_heads)
        if not self.first_seg:
            num = torch.einsum('ihjk,ihkt->ihjt', mk, self.W_mem)
            if self.use_denom:
                denom = torch.einsum("ihj,ihkj->ihk", self.z, mk)[..., None] + 1e-5
                prev_mv = num / denom
                if self.correction:
                    new_info_coef = (1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2)[..., None])
                    new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
                else:
                    new_info_coef = 1
            else:
                prev_mv = num
        else: 
            prev_mv = torch.zeros_like(new_mv, device=new_mv.device)
            new_info_coef = 1
        
        mv = new_mv - prev_mv

        mb = self._to_heads(torch.sigmoid(self.W_mb(mem_tokens)))

        einop = f"ihjk,ihjt,ihj{'t' if self.gating else 'x'}->ihkt"
        associations =  torch.einsum(einop, mk, mv, mb) # (bsz, n_heads, d_key, d_model // n_heads)

        self.W_mem = self.W_mem + associations

        if self.use_denom:
            self.z = self.z + (new_info_coef*mk).sum(dim=-2)
        self.seg_num += 1
        return self.W_mem, getattr(self, "z", None)

    def forward(self, mem_tokens: torch.Tensor):
        # Result is sensitive to fp16 overflows; ensure W_mem doesn't explode
        W_mem, z = self.update_mem(mem_tokens)
        return W_mem



# Benchmark sizes / defaults