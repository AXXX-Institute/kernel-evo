from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kernel_evo.core.eval.eval import get_torch_dtype_from_string  # noqa: E402
from kernel_evo.core.precision import resolve_runtime_precision_string  # noqa: E402


def test_fp8_torch_dtype_falls_back_to_bf16() -> None:
    assert get_torch_dtype_from_string("fp8") == torch.bfloat16


def test_runtime_precision_defaults_to_bf16_for_fp8() -> None:
    assert resolve_runtime_precision_string("fp8", "") == "bf16"


def test_runtime_precision_defaults_to_requested_precision_otherwise() -> None:
    assert resolve_runtime_precision_string("bf16", "") == "bf16"
    assert resolve_runtime_precision_string("fp16", "") == "fp16"


def test_runtime_precision_override_is_respected() -> None:
    assert resolve_runtime_precision_string("fp8", "fp32") == "fp32"
