from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kernel_evo.core.eval.eval import _comparison_view  # noqa: E402


def test_comparison_view_promotes_float16_to_float32() -> None:
    x = torch.ones(4, dtype=torch.float16)
    y = _comparison_view(x)
    assert y.dtype == torch.float32


def test_comparison_view_leaves_int_tensor_unchanged() -> None:
    x = torch.ones(4, dtype=torch.int32)
    y = _comparison_view(x)
    assert y.dtype == torch.int32
