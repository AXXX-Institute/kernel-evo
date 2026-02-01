# Auto-generated seed program.
# Generated from KernelBench level=1 problem_id=15
#
# IMPORTANT: this file is evaluated directly (no entrypoint wrapper).
# It must define `class ModelNew(torch.nn.Module)`.

import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return torch.tril(torch.matmul(A, B))