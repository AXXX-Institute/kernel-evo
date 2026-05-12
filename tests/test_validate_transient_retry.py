from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kernel_evo.resources import validate as validate_mod  # noqa: E402


class _FakeKernelExecResult:
    def __init__(
        self,
        *,
        compiled: bool,
        correctness: bool = False,
        metadata: dict | None = None,
        runtime: float | None = None,
        ref_runtime: float | None = None,
    ) -> None:
        self.compiled = compiled
        self.correctness = correctness
        self.metadata = metadata or {}
        self.runtime = runtime
        self.ref_runtime = ref_runtime


def _install_fake_eval_module(monkeypatch: pytest.MonkeyPatch, evaluator) -> None:
    fake_module = types.ModuleType("kernel_evo.core.eval.eval")
    fake_module.eval_kernel_against_ref = evaluator
    fake_module.get_torch_dtype_from_string = lambda precision: precision
    fake_module.KernelExecResult = _FakeKernelExecResult
    monkeypatch.setitem(sys.modules, "kernel_evo.core.eval.eval", fake_module)


def test_run_local_validation_retries_transient_none_result(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    attempts: list[int] = []

    def evaluator(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            return None
        return SimpleNamespace(
            compiled=True,
            correctness=True,
            metadata={},
            runtime=5.0,
            ref_runtime=10.0,
        )

    _install_fake_eval_module(monkeypatch, evaluator)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validate_mod.time, "sleep", lambda _: None)

    result = validate_mod.run_local_validation(
        problem_dir=tmp_path,
        cfg={
            "device": "cuda",
            "validator_transient_retries": 2,
            "validator_transient_retry_delay": 0.0,
        },
        payload={"program_id": "p1"},
        custom_model_src="class ModelNew:\n    pass\n",
        ref_arch_src="class Model:\n    pass\n",
    )

    assert len(attempts) == 3
    assert result["compiled"] == 1.0
    assert result["correctness"] == 1.0
    assert result["is_valid"] == 1.0
    assert result["speedup"] == 2.0


def test_run_local_validation_raises_after_exhausting_transient_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    attempts: list[int] = []

    def evaluator(*args, **kwargs):
        attempts.append(1)
        return None

    _install_fake_eval_module(monkeypatch, evaluator)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validate_mod.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match=r"after 3 attempt\(s\)"):
        validate_mod.run_local_validation(
            problem_dir=tmp_path,
            cfg={
                "device": "cuda",
                "validator_transient_retries": 2,
                "validator_transient_retry_delay": 0.0,
            },
            payload={"program_id": "p2"},
            custom_model_src="class ModelNew:\n    pass\n",
            ref_arch_src="class Model:\n    pass\n",
        )

    assert len(attempts) == 3
