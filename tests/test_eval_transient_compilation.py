from __future__ import annotations

import pytest

pytest.importorskip("torch")

from kernel_evo.core.eval.eval import _is_transient_compilation_error  # noqa: E402


def test_transient_compilation_error_detects_lock_file_cases() -> None:
    err = RuntimeError("FileNotFoundError: [Errno 2] No such file or directory: '/tmp/torch_extensions/lock'")
    assert _is_transient_compilation_error(err) is True


def test_transient_compilation_error_rejects_failed_build_so_load_case() -> None:
    err = RuntimeError(
        "Error building extension 'x': /tmp/torch_extensions/x/x.so: "
        "cannot open shared object file: No such file or directory"
    )
    assert _is_transient_compilation_error(err) is False


def test_transient_compilation_error_rejects_normal_compiler_failure() -> None:
    err = RuntimeError("error: namespace \"at::cuda\" has no member \"getCurrentCUDAStream\"")
    assert _is_transient_compilation_error(err) is False
