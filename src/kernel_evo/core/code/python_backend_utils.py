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
    return (
        "\n"
        "### Goal\n"
        "Evolve a program that is **Python source code** implemented in torch.nn.Module interface.\n"
        "The code must:\n"
        "- Be correct (matches the reference outputs)\n"
        "- Be more efficient (lower runtime than the reference on GPU)\n"
        f"- Use {run_cfg['backend']} backend as primary framework for kernel implementation\n"
        f"- Efficient implementation primary for {run_cfg['precision']} precision, "
        "so you can use type specific optimizations\n"
        "\n"
        "### Output / parsing rules (IMPORTANT)\n"
        "Many prompts in this workflow are **machine-parsed**. Follow the requested format exactly:\n"
        "- If the prompt asks for JSON: output **raw JSON only** (no markdown fences, no extra text, no extra keys).\n"
        "- If the prompt asks for code inside a JSON field (e.g., `code`): put **only valid Python** "
        "in that field (no markdown).\n"
        "\n"
        "### Performance target (Triton-first)\n"
        "- Replace dominant Torch ops (e.g., einsum/matmul/reductions + elementwise chains) with backend kernels.\n"
        "- Prefer fusion: fewer kernel launches, fewer intermediate tensors, "
        "fewer full passes over HBM-resident tensors.\n"
        "- For Triton: use masked `tl.load`/`tl.store` for tail tiles, "
        "prefer matmul-style blocking (`tl.dot`) for GEMM-like work,\n"
        "  and accumulate sensitive reductions/dot products in fp32 (cast at the end).\n"
        "\n"
        "### Insight traceability (IMPORTANT)\n"
        "- Program insights will be prefixed with IDs like `[id:<type>_<nn>] ...`.\n"
        "- When asked for `insights_used`, copy the insight strings **verbatim**, including the `[id:...]` prefix.\n"
        "\n"
        "### Mutation justification expectation\n"
        "When asked for a justification, include a short **kernel plan**: "
        "which Torch ops were replaced, which kernels were added,\n"
        "what was fused, and why this reduces launches / memory traffic.\n"
        "\n"
        "### What your evolving program must do (IMPORTANT)\n"
        "Your evolving program is stored as `Program.code` and is evaluated *directly*.\n"
        "\n"
        "- Your program must be a **single self-contained Python module**.\n"
        "- It must define: `class ModelNew(torch.nn.Module): ...` and use same interface as the reference model\n"
        "\n" + _backend_compliance_block(run_cfg) + "\n"
        "\n"
        "During evaluation, the validator will call:\n"
        "`kernelbench.eval.eval_kernel_against_ref(ref_arch_src, program_code, backend=..., precision=...)`\n"
        "\n"
        "### IMPORTANT: Safe-mode restrictions on evolving program code\n"
        "Your evolving program is checked by GigaEvo safe-mode. Do NOT:\n"
        "- `import os`, `import sys`, or `import subprocess`\n"
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


def _backend_compliance_block(run_cfg: dict[str, Any]) -> str:
    backend = str(run_cfg.get("backend", "")).lower().strip()

    base = (
        "### HARD REQUIREMENT: Backend compliance (submission validity)\n"
        f"- Backend is `{backend}`.\n"
        "- Your program is INVALID unless it uses that backend *at least once* in the forward pass.\n"
        "- Defining backend code but never calling it is NON-COMPLIANT.\n"
        "- The backend call must contribute to the returned output (not a dead/no-op side call).\n"
    )

    if backend == "triton":
        return (
            base + "\n"
            "#### NON-NEGOTIABLE TRITON COMPLIANCE GATE (submission invalid otherwise)\n"
            "- `import triton` and `import triton.language as tl`\n"
            "- at least one kernel decorated with `@triton.jit`\n"
            "- at least one kernel launch executed from `ModelNew.forward(...)` (e.g. `kernel[grid](...)`)\n"
            "- the Triton kernel output must contribute to the returned output (not a dead/no-op side call)\n"
            "\n"
            "Not sufficient:\n"
            "- importing Triton but never launching a kernel\n"
            "- defining a Triton kernel but never calling it\n"
            "- relying on `torch.compile` / Inductor without an explicit `@triton.jit` kernel\n"
        )

    return base


def _json_pretty(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, ensure_ascii=False, indent=2, sort_keys=True)
