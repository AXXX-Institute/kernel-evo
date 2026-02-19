import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REF_ARCH_BEGIN = "<<<REF_ARCH_SRC_BEGIN>>>"
REF_ARCH_END = "<<<REF_ARCH_SRC_END>>>"

CPP_BACKENDS: set[str] = {
    "cuda",
    "cutlass_cpp",
}


def is_cpp_backend(backend: str) -> bool:
    return str(backend).lower() in CPP_BACKENDS


@dataclass(frozen=True)
class TorchCudaExtensionSpec:
    module_name: str
    binding_cpp: str
    kernel_cu: str
    extra_cxx: list[str]
    extra_nvcc: list[str]


def default_torch_extension_spec(
    *, module_name: str = "mytorch_ext", binding_cpp: str, kernel_cu: str
) -> TorchCudaExtensionSpec:
    return TorchCudaExtensionSpec(
        module_name=module_name,
        binding_cpp=binding_cpp,
        kernel_cu=kernel_cu,
        extra_cxx=["-O3"],
        extra_nvcc=["-O3", "--use_fast_math", "-lineinfo"],
    )


def render_setup_py(spec: TorchCudaExtensionSpec) -> str:
    # Matches the structure you provided; build flags are configurable via spec.
    cxx_args = repr(list(spec.extra_cxx))
    nvcc_args = repr(list(spec.extra_nvcc))
    return (
        "from setuptools import setup\n"
        "from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n"
        "\n"
        "setup(\n"
        f"    name={spec.module_name!r},\n"
        "    ext_modules=[\n"
        "        CUDAExtension(\n"
        f"            name={spec.module_name!r},\n"
        "            sources=['binding.cpp', 'kernel.cu'],\n"
        "            extra_compile_args={\n"
        f"                'cxx': {cxx_args},\n"
        f"                'nvcc': {nvcc_args},\n"
        "            },\n"
        "        )\n"
        "    ],\n"
        "    cmdclass={'build_ext': BuildExtension},\n"
        ")\n"
    )


def write_torch_extension_scaffold(
    *,
    build_dir: Path,
    spec: TorchCudaExtensionSpec,
    clean: bool = False,
) -> None:
    """
    Write `binding.cpp`, `kernel.cu`, and `setup.py` into build_dir.
    """
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if clean and build_dir.exists():
        for p in build_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    (build_dir / "binding.cpp").write_text(spec.binding_cpp, encoding="utf-8")
    (build_dir / "kernel.cu").write_text(spec.kernel_cu, encoding="utf-8")
    (build_dir / "setup.py").write_text(render_setup_py(spec), encoding="utf-8")


def build_torch_extension_inplace(*, build_dir: Path, python_exe: str = "python") -> None:
    """
    Build the extension using `python setup.py build_ext --inplace`.
    """
    build_dir = Path(build_dir)
    cmd = [python_exe, "setup.py", "build_ext", "--inplace"]
    subprocess.run(cmd, cwd=str(build_dir), check=True)


def raise_bindings_not_implemented(*args: Any, **kwargs: Any) -> None:
    """
    Placeholder for the C++-based backend integration.

    The intended flow is:
      1) evolve kernel source (and optionally binding code),
      2) compile via torch cpp_extension,
      3) load and run in the validator.

    For now, this is deliberately unimplemented.
    """
    raise NotImplementedError("C++ backend bindings/compilation integration is not implemented yet.")


def build_task_description_cpp(*, run_cfg: dict[str, Any], ref_arch_src: str) -> str:
    """
    C++-style template: the evolving artifact is intended to be *kernel source generation*.

    Current status:
      - validator side compilation/bindings are NOT implemented yet (will mark programs invalid)
      - this prompt exists to lock down the expected interface for future work
    """
    prompt_cfg = {k: v for k, v in run_cfg.items() if k not in {"ref_arch_src"}}
    return (
        "KERNELBENCH KERNEL GENERATION (GigaEvo problem)\n"
        "\n"
        "### Goal\n"
        "Evolve a program that generates a C++/CUDA kernel implementation.\n"
        "\n"
        "### What your evolving program must do (IMPORTANT)\n"
        "There is **NO `entrypoint(...)`**.\n"
        "\n"
        "Your program should be a Python module that defines:\n"
        "\n"
        "  def generate_kernel(context=None) -> dict[str, str]:\n"
        "      return {\n"
        "        'binding.cpp': '...C++ binding code...',\n"
        "        'kernel.cu': '...CUDA kernel code...',\n"
        "      }\n"
        "\n"
        "### Runtime / evaluation config (duplicated here for the prompt)\n"
        "```json\n" + _json_pretty(prompt_cfg) + "\n```\n"
        "\n"
        "### Status\n"
        "**Bindings + compilation are not implemented yet**; the validator will currently mark\n"
        "these programs invalid (placeholder `NotImplementedError`).\n"
        "\n"
        "### Reference model source code\n"
        f"{REF_ARCH_BEGIN}\n{ref_arch_src.rstrip()}\n{REF_ARCH_END}\n"
        "\n"
    )


def seed_program_cpp_stub() -> str:
    """
    Minimal seed that satisfies the expected interface for the C++ backend path.
    """
    return (
        "def generate_kernel(context=None):\n"
        "    # TODO: return real binding.cpp + kernel.cu sources.\n"
        "    return {\n"
        "        'binding.cpp': r'''// Not implemented''',\n"
        "        'kernel.cu': r'''// Not implemented''',\n"
        "    }\n"
    )


def _json_pretty(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)
