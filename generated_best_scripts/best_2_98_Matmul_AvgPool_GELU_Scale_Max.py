import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_pool_gelu_scale_max_kernel(
    x_ptr, out_ptr,
    stride_x_row, stride_x_col,
    n_cols, pool_size, scale_factor,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    
    # Number of elements after pooling
    # AvgPool1d with default stride=kernel_size: out_len = n_cols // pool_size
    out_len = n_cols // pool_size
    
    max_val = -float('inf')
    
    # Iterate over the pooled outputs
    for i in range(0, out_len, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < out_len
        
        # Compute average pool for each window
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(pool_size):
            val = tl.load(x_row_ptr + offsets * pool_size + j, mask=mask, other=0.0).to(tl.float32)
            acc += val
        
        avg = acc / pool_size
        
        # GELU implementation: 0.5 * x * (1 + erf(x / sqrt(2)))
        # Using approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Triton has tl.extra.cuda.libdevice.erf or we can use a simple approximation
        # For simplicity and speed, we use the standard GELU formula
        gelu_in = avg
        # tl.math.erf is available in newer Triton, otherwise use approximation
        # Here we use the exact erf for correctness
        gelu_out = 0.5 * gelu_in * (1.0 + tl.extra.cuda.libdevice.erf(gelu_in * 0.70710678118))
        
        # Scale
        scaled = gelu_out * scale_factor
        
        # Local max reduction
        local_max = tl.max(tl.where(mask, scaled, -float('inf')), axis=0)
        max_val = tl.maximum(max_val, local_max)

    tl.store(out_ptr + row_idx, max_val)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.out_features = out_features

    def forward(self, x):
        # Linear layer (Matmul + Bias)
        x = self.matmul(x)
        
        batch_size = x.shape[0]
        n_cols = x.shape[1]
        output = torch.empty((batch_size,), device=x.device, dtype=x.dtype)
        
        # Launch fused kernel to handle AvgPool -> GELU -> Scale -> Max
        # We use a block size for the loop inside the kernel
        grid = (batch_size,)
        fused_pool_gelu_scale_max_kernel[grid](
            x, output,
            x.stride(0), x.stride(1),
            n_cols, self.pool_kernel_size, self.scale_factor,
            BLOCK_SIZE=256
        )
        
        return output