from __future__ import annotations

import pytest

pytest.importorskip("loguru")

from kernel_evo.resources.validate import _find_disallowed_forward_float32_casts  # noqa: E402


def test_precision_policy_flags_float_cast_in_forward_for_non_fp32() -> None:
    src = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        x = x.float()
        return x
"""
    violations = _find_disallowed_forward_float32_casts(src, "bf16")
    assert violations
    assert ".float()" in violations[0]


def test_precision_policy_flags_to_float32_in_forward_for_non_fp32() -> None:
    src = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return x.to(dtype=torch.float32)
"""
    violations = _find_disallowed_forward_float32_casts(src, "bf16")
    assert violations
    assert "torch.float32" in violations[0]


def test_precision_policy_ignores_float32_inside_cuda_source_strings() -> None:
    src = '''
import torch
import torch.nn as nn

cuda_source = """
__global__ void kernel(float* x) {}
"""

class ModelNew(nn.Module):
    def forward(self, x):
        return x
'''
    assert _find_disallowed_forward_float32_casts(src, "fp16") == []


def test_precision_policy_allows_forward_float_cast_for_fp32() -> None:
    src = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return x.float()
    """
    assert _find_disallowed_forward_float32_casts(src, "fp32") == []
