# Auto-generated seed program.
# Generated from KernelBench level=1 problem_id=97
#
# IMPORTANT: this file is evaluated directly (no entrypoint wrapper).
# It must define `class ModelNew(torch.nn.Module)`.

import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out