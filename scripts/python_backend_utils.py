from __future__ import annotations

from typing import Any


REF_ARCH_BEGIN = "<<<REF_ARCH_SRC_BEGIN>>>"
REF_ARCH_END = "<<<REF_ARCH_SRC_END>>>"
REF_MODEL_BEGIN = "<<<REF_MODEL_CLASS_BEGIN>>>"
REF_MODEL_END = "<<<REF_MODEL_CLASS_END>>>"
REF_INPUTS_BEGIN = "<<<REF_INPUTS_INIT_BEGIN>>>"
REF_INPUTS_END = "<<<REF_INPUTS_INIT_END>>>"


PYTHON_BACKENDS: set[str] = {
    # KernelBench python-driven backends
    "triton",
    "tilelang",
    "cute",
    "thunderkittens",
    # Optional convenience alias (not necessarily supported by KernelBench evaluator)
    "torch",
}


def is_python_backend(backend: str) -> bool:
    return str(backend).lower() in PYTHON_BACKENDS


def split_kernelbench_ref(ref_arch_src: str) -> tuple[str, str]:
    """
    Split KernelBench reference source into:
      - model_src: imports + class Model(...) block
      - inputs_src: everything after the class (e.g., M=..., get_inputs, get_init_inputs)
    """
    lines = ref_arch_src.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("class Model"):
            start = i
            break
    if start is None:
        return ref_arch_src.rstrip() + "\n", ""

    header = lines[:start]
    model_block: list[str] = []
    i = start
    while i < len(lines):
        model_block.append(lines[i])
        i += 1
        if i >= len(lines):
            break
        nxt = lines[i]
        # boundary: first top-level statement after class block
        if nxt and (not nxt.startswith((" ", "\t"))) and (not nxt.lstrip().startswith("#")):
            break

    model_src = "\n".join(header + model_block).rstrip() + "\n"
    inputs_src = "\n".join(lines[i:]).rstrip() + "\n"
    return model_src, inputs_src


def model_to_modelnew(model_src: str) -> str:
    """
    Turn reference `class Model(nn.Module): ...` into a `ModelNew`-only definition.

    Also fixes the common KernelBench `super(Model, self).__init__()` pattern.
    """
    s = model_src.replace("class Model(", "class ModelNew(", 1)
    s = s.replace("super(Model, self).__init__()", "super().__init__()")
    s = s.replace("super(Model, self)", "super(ModelNew, self)")
    return s.rstrip() + "\n"


def build_task_description_python(
    *,
    run_cfg: dict[str, Any],
    ref_arch_src: str,
    ref_model_class_src: str,
    ref_inputs_init_src: str,
) -> str:
    """
    Return a full `task_description.txt` for python-based backends where:
      - the evolving Program.code IS the final python source (no entrypoint)
      - validator receives context via context.py, but all context is also duplicated here for the LLM
    """
    # Keep the context in the prompt, but avoid duplicating the full reference code twice.
    prompt_cfg = {k: v for k, v in run_cfg.items() if k not in {"ref_arch_src"}}

    return (
        "KERNELBENCH KERNEL GENERATION (GigaEvo problem)\n"
        "\n"
        "### Goal\n"
        "Evolve a program that is **Python source code** for a KernelBench “custom model”.\n"
        "The code must:\n"
        "- Be correct (matches the reference outputs)\n"
        "- Be more efficient (lower runtime than the reference on GPU)\n"
        f"- Use {run_cfg['backend']} backend as primary framework for kernel implementation\n"
        f"- Efficient primary for {run_cfg['precision']} precision, so you can use type specific optimizations"
        "\n"
        "### What your evolving program must do (IMPORTANT)\n"
        "Your evolving program is stored as `Program.code` and is evaluated *directly*.\n"
        "\n"
        # "- There is **NO `entrypoint(...)`**.\n"
        "- Your program must be a **single self-contained Python module**.\n"
        "- It must define: `class ModelNew(torch.nn.Module): ...` and use same interface as the reference model\n"
        # "- Prefer **minimal diffs** from the current program: tune parameters, avoid rewriting the whole file.\n"
        # "- Correctness-first: keep changes small until you have a consistently valid program.\n"
        # "- If you introduce backend-specific/JIT kernel code, keep a safe fallback path to a known-correct implementation.\n"
        "\n"
        "During evaluation, the validator will call:\n"
        "`kernelbench.eval.eval_kernel_against_ref(ref_arch_src, program_code, backend=..., precision=...)`\n"
        "\n"
        # "### Runtime / evaluation config (duplicated here for the prompt)\n"
        # "(The validator also receives this dict via `context.py` + `run_config.json`.)\n"
        # "\n"
        # "```json\n"
        # # + _json_pretty(prompt_cfg)
        # + _json_pretty({
        #     "backend": run_cfg["backend"],
        #     "precision": run_cfg["precision"],
        # })
        # + "\n```\n"
        # "\n"
        "### IMPORTANT: Safe-mode restrictions on evolving program code\n"
        "Your evolving program is checked by GigaEvo safe-mode. Do NOT:\n"
        "- `import os`, `import sys`, or `import subprocess`\n"
        "- Use file I/O (e.g. `open(...)`, `.read()`, `.write()`, `.remove()`, `.unlink()`)\n"
        "\n"
        # "### Reference model source code\n"
        # f"{REF_ARCH_BEGIN}\n{ref_arch_src.rstrip()}\n{REF_ARCH_END}\n"
        # "\n"
        # "### Reference model (class-only)\n"
        "### Reference model source code\n"
        f"{REF_MODEL_BEGIN}\n{ref_model_class_src.rstrip()}\n{REF_MODEL_END}\n"
        "\n"
        "### Reference inputs/init (NOT part of your generated code)\n"
        f"{REF_INPUTS_BEGIN}\n{(ref_inputs_init_src.rstrip() if ref_inputs_init_src.strip() else '<none>')}\n{REF_INPUTS_END}\n"
        "\n"
    )


def _json_pretty(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)
