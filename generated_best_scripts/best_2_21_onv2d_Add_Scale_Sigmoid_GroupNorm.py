import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gn_welford_kernel(
    x_ptr, bias_ptr, scale_ptr, 
    gn_weight_ptr, gn_bias_ptr, out_ptr,
    N, C, HW, G, C_per_G,
    BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr
):
    # Each program handles one (batch, group)
    group_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    group_base_offset = (batch_idx * C * HW) + (group_idx * C_per_G * HW)
    
    # Welford's algorithm for single-pass mean/variance
    mean = 0.0
    m2 = 0.0
    count = 0.0

    for c in range(0, C_per_G, BLOCK_C):
        c_offsets = c + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C_per_G
        
        curr_c_idx = group_idx * C_per_G + c_offsets
        b_val = tl.load(bias_ptr + curr_c_idx[:, None], mask=c_mask[:, None], other=0.0).to(tl.float32)
        s_val = tl.load(scale_ptr + curr_c_idx[:, None], mask=c_mask[:, None], other=0.0).to(tl.float32)

        for hw_off in range(0, HW, BLOCK_HW):
            hw_offsets = hw_off + tl.arange(0, BLOCK_HW)
            hw_mask = hw_offsets < HW
            
            ptr = x_ptr + group_base_offset + (c_offsets[:, None] * HW) + hw_offsets[None, :]
            mask = c_mask[:, None] & hw_mask[None, :]
            val = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            
            # Elementwise fusion: sigmoid((x + bias) * scale)
            val = tl.sigmoid((val + b_val) * s_val)
            
            # Welford update
            mask_f = mask.to(tl.float32)
            num_elements = tl.sum(mask_f)
            if num_elements > 0:
                batch_mean = tl.sum(tl.where(mask, val, 0.0)) / num_elements
                # Simplified Welford for block-based updates
                delta = tl.where(mask, val - mean, 0.0)
                mean += tl.sum(delta) / (count + num_elements)
                delta2 = tl.where(mask, val - mean, 0.0)
                m2 += tl.sum(delta * delta2)
                count += num_elements

    var = m2 / count
    inv_std = tl.extra.cuda.libdevice.rsqrt(var + 1e-5)

    # Second pass (still needed for GN logic, but now we only load once per normalization)
    for c in range(0, C_per_G, BLOCK_C):
        c_offsets = c + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C_per_G
        
        curr_c_idx = group_idx * C_per_G + c_offsets
        b_val = tl.load(bias_ptr + curr_c_idx[:, None], mask=c_mask[:, None], other=0.0).to(tl.float32)
        s_val = tl.load(scale_ptr + curr_c_idx[:, None], mask=c_mask[:, None], other=0.0).to(tl.float32)
        gn_w = tl.load(gn_weight_ptr + curr_c_idx[:, None], mask=c_mask[:, None], other=0.0).to(tl.float32)
        gn_b = tl.load(gn_bias_ptr + curr_c_idx[:, None], mask=c_mask[:, None], other=0.0).to(tl.float32)

        for hw_off in range(0, HW, BLOCK_HW):
            hw_offsets = hw_off + tl.arange(0, BLOCK_HW)
            hw_mask = hw_offsets < HW
            
            ptr = x_ptr + group_base_offset + (c_offsets[:, None] * HW) + hw_offsets[None, :]
            out_p = out_ptr + group_base_offset + (c_offsets[:, None] * HW) + hw_offsets[None, :]
            mask = c_mask[:, None] & hw_mask[None, :]
            
            val = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            val = tl.sigmoid((val + b_val) * s_val)
            
            res = (val - mean) * inv_std * gn_w + gn_b
            tl.store(out_p, res.to(out_ptr.dtype.element_ty), mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        G = self.num_groups
        C_per_G = C // G
        HW = H * W
        
        out = torch.empty_like(x)
        
        # Heuristic for block sizes
        BLOCK_HW = 1024
        BLOCK_C = 4 if C_per_G >= 4 else 1
            
        grid = (G, N)
        fused_gn_welford_kernel[grid](
            x, self.bias.view(-1), self.scale.view(-1), 
            self.group_norm.weight, self.group_norm.bias, out,
            N, C, HW, G, C_per_G,
            BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C
        )
        
        return out