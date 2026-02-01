import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Standard Matmul grouping to improve L2 cache hit rate
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Since result is lower triangular, skip blocks where the entire block is in the upper triangle
    # A block (pid_m, pid_n) is upper triangular if its start row is less than its end column
    if (pid_m * BLOCK_SIZE_M) < (pid_n * BLOCK_SIZE_N):
        # Note: This is a coarse check. Fine-grained tril is handled at the end.
        # However, for tril(A*B) where A, B are tril, we only care about pid_m >= pid_n
        if pid_m < pid_n:
            return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Optimization: A and B are lower triangular.
    # For C[m, n] = sum(A[m, k] * B[k, n]), 
    # A[m, k] is non-zero only if k <= m.
    # B[k, n] is non-zero only if k >= n.
    # So k ranges from n to m.
    k_start = (pid_n * BLOCK_SIZE_N) // BLOCK_SIZE_K
    k_end = tl.cdiv((pid_m + 1) * BLOCK_SIZE_M, BLOCK_SIZE_K)
    
    a_ptrs += k_start * BLOCK_SIZE_K * stride_ak
    b_ptrs += k_start * BLOCK_SIZE_K * stride_bk

    for k in range(k_start, k_end):
        k_idx = k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < N) & (k_idx + offs_k[None, :] < N), other=0.0)
        b = tl.load(b_ptrs, mask=(k_idx + offs_k[:, None] < N) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Final lower triangular mask for the output
    mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_m[:, None] < N) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator.to(C_ptr.dtype.element_ty), mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        if not A.is_cuda:
            return torch.tril(torch.matmul(A, B))
        
        N = A.shape[0]
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)

        # Grid covers all blocks; the kernel handles triangular skipping internally
        grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

        tril_matmul_kernel[grid](
            A, B, C,
            N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1)
        )

        return C