"""
CUDA inline backend: Python programs that compile inline C++/CUDA via torch.utils.cpp_extension.load_inline.

Similar structure to python_backend_utils: the evolved program is Python source defining
class ModelNew(nn.Module) and using load_inline() to compile CUDA/C++ on the fly.
ARCH_LIST (TORCH_CUDA_ARCH_LIST) can be optionally provided in run_cfg.

Note: cuda_inline and all future backends may have a separate prompt directory
(e.g. resources/prompts/<backend>/ or per-run prompts). See python_backend_utils.BACKENDS_WITH_SEPARATE_PROMPT_DIR.
"""

from pathlib import Path
from typing import Any


# Backend identifier for the inline-CUDA path (evolved code is Python using load_inline)
CUDA_INLINE_BACKENDS: set[str] = {"cuda_inline"}


def is_cuda_inline_backend(backend: str) -> bool:
    return str(backend).lower() in CUDA_INLINE_BACKENDS


def get_prompt_dir_for_backend(backend: str, base_dir: Path) -> Path | None:
    """
    Optional per-backend prompt directory. cuda_inline and future backends may use
    a separate prompt directory (e.g. base_dir / "prompts" / backend). Returns None if not used.
    """
    from kernel_evo.core.code.python_backend_utils import BACKENDS_WITH_SEPARATE_PROMPT_DIR

    if backend.lower() not in BACKENDS_WITH_SEPARATE_PROMPT_DIR:
        return None
    path = base_dir / "prompts" / backend.lower()
    return path if path.exists() else None


def get_arch_list_from_config(run_cfg: dict[str, Any]) -> str | None:
    """
    Return TORCH_CUDA_ARCH_LIST value from run config if set.

    Optional keys: 'arch_list', 'cuda_arch_list', 'TORCH_CUDA_ARCH_LIST'.
    """
    for key in ("arch_list", "cuda_arch_list", "TORCH_CUDA_ARCH_LIST"):
        v = run_cfg.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def apply_cuda_arch_env(run_cfg: dict[str, Any] | None) -> None:
    """
    Set os.environ['TORCH_CUDA_ARCH_LIST'] from run_cfg if provided.
    Call before load_inline so the compiler uses the requested architectures.
    """
    import os

    if not run_cfg:
        return
    arch = get_arch_list_from_config(run_cfg)
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch


def get_cuda_inline_compliance_block(run_cfg: dict[str, Any]) -> str:
    """
    Backend compliance text for task description when backend is cuda_inline.
    """
    backend = str(run_cfg.get("backend", "")).lower().strip()
    arch_note = ""
    if get_arch_list_from_config(run_cfg):
        arch_note = (
            "\n- Optional: run_config may set 'arch_list' (or 'cuda_arch_list') to restrict "
            "TORCH_CUDA_ARCH_LIST for compilation.\n"
        )
    return (
        "### HARD REQUIREMENT: Backend compliance (submission validity)\n"
        f"- Backend is `{backend}` (CUDA inline: compile C++/CUDA on the fly via load_inline).\n"
        "- Your program is INVALID unless it uses `torch.utils.cpp_extension.load_inline` "
        "with both `cpp_sources` and `cuda_sources` to define and load at least one CUDA kernel.\n"
        "- You must call the loaded CUDA function(s) from `ModelNew.forward(...)`; "
        "the kernel output must contribute to the returned output (not a dead/no-op side call).\n"
        "\n"
        "#### CUDA inline compliance (submission invalid otherwise)\n"
        "- Use `from torch.utils.cpp_extension import load_inline`.\n"
        "- Pass inline C++ binding code as `cpp_sources` and CUDA kernel code as `cuda_sources`.\n"
        "- Use `functions=[...]` to expose the kernel entry point(s), and call them in forward.\n"
        "- You may use `extra_cuda_cflags` (e.g. `['--use_fast_math', '-O3']`) for optimization.\n"
        "- Defining inline CUDA but never calling the loaded function from forward is NON-COMPLIANT.\n"
        f"{arch_note}"
    )


def build_task_description_cuda_inline(
    *,
    run_cfg: dict[str, Any],
    ref_arch_src: str,
    ref_model_class_src: str,
    ref_inputs_init_src: str,
) -> str:
    """
    Build full task_description.txt for the cuda_inline backend.
    Same overall shape as python_backend_utils.build_task_description_python, with cuda_inline-specific goal and compliance.
    """
    from kernel_evo.core.code.python_backend_utils import (
        REF_INPUTS_BEGIN,
        REF_INPUTS_END,
        REF_MODEL_BEGIN,
        REF_MODEL_END,
    )

    return (
        "\n"
        "### Goal\n"
        "Evolve a program that is **Python source code** implementing the torch.nn.Module interface.\n"
        "The code must use **inline C++/CUDA** compiled on the fly via `torch.utils.cpp_extension.load_inline`.\n"
        "The code must:\n"
        "- Be correct (matches the reference outputs)\n"
        "- Be more efficient (lower runtime than the reference on GPU)\n"
        f"- Use the cuda_inline backend (load_inline with cpp_sources + cuda_sources)\n"
        f"- Efficient implementation primary for {run_cfg['precision']} precision, "
        "so you can use type-specific optimizations\n"
        "\n"
        "### Output / parsing rules (IMPORTANT)\n"
        "Many prompts in this workflow are **machine-parsed**. Follow the requested format exactly:\n"
        "- If the prompt asks for JSON: output **raw JSON only** (no markdown fences, no extra text, no extra keys).\n"
        "- If the prompt asks for code inside a JSON field (e.g., `code`): put **only valid Python** "
        "in that field (no markdown).\n"
        "\n"
        "### Performance target\n"
        "- Replace dominant Torch ops with your own CUDA kernels built via load_inline.\n"
        "- Prefer fusion: fewer kernel launches, fewer intermediate tensors.\n"
        "- Use appropriate block/grid sizes and vectorized loads/stores (e.g. float4) where beneficial.\n"
        "\n"
        "### Insight traceability (IMPORTANT)\n"
        "- Program insights will be prefixed with IDs like `[id:<type>_<nn>] ...`.\n"
        "- When asked for `insights_used`, copy the insight strings **verbatim**, including the `[id:...]` prefix.\n"
        "\n"
        "### Mutation justification expectation\n"
        "When asked for a justification, include a short **kernel plan**: "
        "which Torch ops were replaced, which kernels were added, and why this reduces launches / memory traffic.\n"
        "\n"
        "### What your evolving program must do (IMPORTANT)\n"
        "Your evolving program is stored as `Program.code` and is evaluated *directly*.\n"
        "\n"
        "- Your program must be a **single self-contained Python module**.\n"
        "- It must define: `class ModelNew(torch.nn.Module): ...` with the same interface as the reference model.\n"
        "\n"
        + get_cuda_inline_compliance_block(run_cfg)
        + "\n"
        "During evaluation, the validator will call:\n"
        "`kernelbench.eval.eval_kernel_against_ref(ref_arch_src, program_code, backend='cuda_inline', precision=...)`\n"
        "\n"
        "### IMPORTANT: Safe-mode restrictions on evolving program code\n"
        "Your evolving program is checked by GigaEvo safe-mode. Do NOT:\n"
        "- `import os`, `import sys`, or `import subprocess` (except as needed for load_inline usage)\n"
        "- Use file I/O (e.g. `open(...)`, `.read()`, `.write()`, `.remove()`, `.unlink()`)\n"
        "\n"
        "### Reference model source code\n"
        f"{REF_MODEL_BEGIN}\n{ref_model_class_src.rstrip()}\n{REF_MODEL_END}\n"
        "\n"
        "### Reference inputs/init (NOT part of your generated code)\n"
        f"{REF_INPUTS_BEGIN}\n"
        f"{(ref_inputs_init_src.rstrip() if ref_inputs_init_src.strip() else '<none>')}\n"
        f"{REF_INPUTS_END}\n"
        "\n"
    )
