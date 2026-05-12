from __future__ import annotations

import os
from pathlib import Path

from kernel_evo.core.code import cuda_backend_utils as cuda_utils


def test_discover_cuda_userland_paths_finds_nvidia_python_package_dirs(
    monkeypatch, tmp_path: Path
) -> None:
    site_root = tmp_path / "site-packages"
    cublas_include = site_root / "nvidia" / "cublas" / "include"
    cusparse_lib = site_root / "nvidia" / "cusparse" / "lib"
    cublas_include.mkdir(parents=True)
    cusparse_lib.mkdir(parents=True)

    monkeypatch.setattr(cuda_utils.site, "getsitepackages", lambda: [str(site_root)])
    monkeypatch.setattr(cuda_utils.site, "getusersitepackages", lambda: str(tmp_path / "missing-user-site"))

    include_dirs, library_dirs = cuda_utils.discover_cuda_userland_paths({})

    assert str(cublas_include.resolve()) in include_dirs
    assert str(cusparse_lib.resolve()) in library_dirs


def test_apply_cuda_build_env_exports_include_and_library_paths(monkeypatch, tmp_path: Path) -> None:
    cuda_home = tmp_path / "cuda"
    include_dir = cuda_home / "include"
    lib_dir = cuda_home / "lib64"
    include_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)

    monkeypatch.setenv("CPATH", "")
    monkeypatch.setenv("CPLUS_INCLUDE_PATH", "")
    monkeypatch.setenv("LIBRARY_PATH", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(cuda_utils.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(cuda_utils.site, "getusersitepackages", lambda: str(tmp_path / "missing-user-site"))

    cuda_utils.apply_cuda_build_env({"cuda_home": str(cuda_home)})

    assert str(include_dir.resolve()) in os.environ["CPATH"]
    assert str(include_dir.resolve()) in os.environ["CPLUS_INCLUDE_PATH"]
    assert str(lib_dir.resolve()) in os.environ["LIBRARY_PATH"]
    assert str(lib_dir.resolve()) in os.environ["LD_LIBRARY_PATH"]
