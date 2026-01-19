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


class Model(nn.Module):
    """
    Standalone benchmark wrapper for `AssociativeLayerWrapper.associate(...)`.

    This isolates `language_modeling.py:108-129` into a self-contained module:
      - in: hidden_states (bsz, seq_len, d_model)
      - out: hidden_states (bsz, seq_len, d_model)
    """

    def __init__(
        self, d_model: int, d_mem: int, n_heads: int, use_denom: bool, nu: int, batch_size: int = 1
    ):
        super().__init__()

        # Default shapes (tuned to be non-trivial but GPU/CI-friendly).
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.use_denom = bool(use_denom)
        self.nu = nu

        assert self.d_model % self.n_heads == 0
        assert self.d_mem % self.n_heads == 0

        self.d_key = 2 * self.nu * self.d_mem

        self.phi = DPFP(self.nu)
        self.W_mq = nn.Linear(self.d_model, self.d_mem, bias=False)

        # Persistent memory tensors required by associate(). In the original code these
        # are updated by `update_mem(...)` and become batch-shaped. For this benchmark
        # we initialize them directly with the batch dimension.
        #
        # Shapes expected by einsums in associate():
        # - W_mem: (bsz, n_heads, d_key/n_heads, d_model/n_heads)
        # - z:     (bsz, n_heads, d_key/n_heads)
        W_mem = (
            torch.randn(
                batch_size,
                self.n_heads,
                self.d_key // self.n_heads,
                self.d_model // self.n_heads,
            )
            * 0.01
        )
        # Buffer (not a Parameter): should follow model.to(device, dtype)
        self.register_buffer("W_mem", W_mem, persistent=False)
        if self.use_denom:
            # DPFP output is non-negative; keep denom positive and away from 0.
            z = torch.ones(batch_size, self.n_heads, self.d_key // self.n_heads)
            self.register_buffer("z", z, persistent=False)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = x.shape
        x = x.reshape(bsz, seq_len, self.n_heads, d // self.n_heads)
        x = x.permute(0, 2, 1, 3)
        return x

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads_, seq_len, d_head = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads_ * d_head)
        return x

    # Isolated from `language_modeling.py:108-129` (minimal edits: none).
    def associate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model_ = hidden_states.shape

        # Ensure internal memory tensors match input device + dtype (e.g. fp16 on CUDA).
        if self.W_mem.device != hidden_states.device or self.W_mem.dtype != hidden_states.dtype:
            self.W_mem = self.W_mem.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if self.use_denom:
            if self.z.device != hidden_states.device or self.z.dtype != hidden_states.dtype:
                self.z = self.z.to(device=hidden_states.device, dtype=hidden_states.dtype)

        q = self._to_heads(self.W_mq(hidden_states))
        mq = self.phi(q)  # (bsz, n_heads, seq_len, 2 * d_head * nu)
        mq = F.normalize(mq, dim=-1, p=2.0)

        num = torch.einsum("ihjk,ihkt->ihjt", mq, self.W_mem)
        if self.use_denom:
            denom = torch.einsum("ihk,ihjk->ihj", self.z, mq)[..., None] + 1e-5
            hidden_states = num / denom  # (bsz, n_heads, seq_len, d_model // n_heads)
        else:
            hidden_states = num
        hidden_states = self._from_heads(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.associate(hidden_states)


# Benchmark sizes / defaults (KernelBench-style globals)
batch_size = 8
seq_len = 1024
d_model = 512
n_heads = 8
d_mem = 64
nu = 3
use_denom = True


def get_inputs():
    hidden_states = torch.rand(batch_size, seq_len, d_model)
    return [hidden_states]


def get_init_inputs():
    # (self, d_model: int, d_mem: int, n_heads: int, use_denom: bool, nu: int):
    return [d_model, d_mem, n_heads, use_denom, nu, batch_size]
