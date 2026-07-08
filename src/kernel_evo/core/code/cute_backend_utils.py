"""
CuTe DSL backend: Python programs that implement kernels with the NVIDIA CUTLASS CuTe DSL.

Similar structure to cuda_backend_utils: the evolved program is Python source defining
class ModelNew(nn.Module). Instead of compiling C++/CUDA via load_inline, it builds kernels
with `cutlass.cute` (the CuTe Python DSL) and JIT-compiles them with `cute.compile(...)`.

Kernels are written as Python functions decorated with `@cute.kernel` and launched from a
`@cute.jit` host function via `.launch(grid=..., block=...)`. Torch tensors are handed to the
DSL with `from cutlass.cute.runtime import from_dlpack`. The DSL targets the current torch CUDA
stream by default, so no explicit stream wiring is required.

ARCH_LIST (TORCH_CUDA_ARCH_LIST) can be optionally provided in run_cfg, like cuda_inline.

Note: cute (like cuda_inline) uses a separate prompt directory. See
python_backend_utils.BACKENDS_WITH_SEPARATE_PROMPT_DIR and resources/prompts/backends/cute/.
"""

from typing import Any

from kernel_evo.core.code.python_backend_utils import _precision_contract_block

# Backend identifier for the CuTe DSL path (evolved code is Python using cutlass.cute)
CUTE_BACKENDS: set[str] = {"cute"}


def is_cute_backend(backend: str) -> bool:
    return str(backend).lower() in CUTE_BACKENDS


# Verified-working CuTe identity (copy) kernel used to make the initial seed a *compliant* CuTe
# program. The seed keeps the reference torch compute and routes its output through this kernel, so
# the archive is seeded with a valid CuTe program (~1x) that the LLM then mutates by *editing
# working code* (fuse the torch epilogue into the kernel) instead of authoring CuTe from scratch.
_CUTE_SEED_PRELUDE = '''import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _cute_identity_kernel(gA, gC, cC, shape, thr_layout, val_layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    coord = ((None, None), bidx)
    blkA, blkC, blkCrd = gA[coord], gC[coord], cC[coord]
    ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)
    tcA = cute.make_tiled_copy_tv(ld, thr_layout, val_layout)
    tcC = cute.make_tiled_copy_tv(st, thr_layout, val_layout)
    sA, sC = tcA.get_slice(tidx), tcC.get_slice(tidx)
    thrA, thrC, thrCrd = sA.partition_S(blkA), sC.partition_S(blkC), sC.partition_S(blkCrd)
    frgA, frgC = cute.make_fragment_like(thrA), cute.make_fragment_like(thrC)
    pred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    for i in range(cute.size(pred)):
        pred[i] = cute.elem_less(thrCrd[i], shape)
    cute.copy(ld, thrA, frgA, pred=pred)
    frgC.store(frgA.load())   # identity epilogue: mutate this register-compute line to fuse ops
    cute.copy(st, frgC, thrC, pred=pred)


@cute.jit
def _cute_identity_launch(mA, mC, copy_bits: cutlass.Constexpr = 128):
    vec = copy_bits // mA.element_type.width
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vec), order=(1, 0))
    tiler, _ = cute.make_layout_tv(thr_layout, val_layout)
    gA, gC = cute.zipped_divide(mA, tiler), cute.zipped_divide(mC, tiler)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler)
    _cute_identity_kernel(gA, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1], block=[cute.size(thr_layout), 1, 1])


class _CuteSeedMixin:
    """Routes a torch.Tensor through the verified CuTe identity kernel (compliant + correct)."""

    _cute_compiled = None

    def _cute_apply(self, y):
        if not (isinstance(y, torch.Tensor) and y.is_cuda and y.numel() > 0):
            return y
        y2 = y.detach().contiguous()
        flat = y2.reshape(y2.shape[0], -1) if y2.dim() > 1 else y2.reshape(1, -1)
        out = torch.empty_like(flat)
        ca = from_dlpack(flat, assumed_align=16).mark_layout_dynamic()
        cc = from_dlpack(out, assumed_align=16).mark_layout_dynamic()
        if type(self)._cute_compiled is None:
            type(self)._cute_compiled = cute.compile(_cute_identity_launch, ca, cc)
        type(self)._cute_compiled(ca, cc)
        return out.reshape(y.shape)
'''


def build_cute_seed(model_src: str) -> str:
    """
    Build the initial seed program for the cute backend.

    The reference `class Model(...)` is renamed to `_RefModel`; `ModelNew` subclasses it and routes
    the torch forward output through a verified CuTe identity kernel. This guarantees the seed is a
    valid, *backend-compliant* CuTe program (a real `@cute.kernel` launched from forward and
    contributing to the output), so evolution starts from working CuTe code.
    """
    s = model_src.replace("class Model(", "class _RefModel(", 1)
    s = s.replace("super(Model, self).__init__()", "super().__init__()")
    s = s.replace("super(Model, self)", "super(_RefModel, self)")
    wrapper = (
        "\n\nclass ModelNew(_RefModel, _CuteSeedMixin):\n"
        "    def forward(self, *args, **kwargs):\n"
        "        return self._cute_apply(super().forward(*args, **kwargs))\n"
    )
    return _CUTE_SEED_PRELUDE + "\n\n" + s.rstrip() + "\n" + wrapper


def apply_cute_build_env(run_cfg: dict[str, Any] | None) -> None:
    """
    Set environment needed before the CuTe DSL JIT-compiles kernels.

    The CuTe DSL ships its own toolchain and compiles in-process, so this mostly mirrors
    cuda_inline: honor an optional TORCH_CUDA_ARCH_LIST and expose nvidia userland include/lib
    paths discovered from the active environment (harmless if unused).
    """
    from kernel_evo.core.code.cuda_backend_utils import apply_cuda_build_env

    apply_cuda_build_env(run_cfg)


def get_cute_compliance_block(run_cfg: dict[str, Any]) -> str:
    """
    Backend compliance text for task description when backend is cute.
    """
    backend = str(run_cfg.get("backend", "")).lower().strip()
    return (
        "### HARD REQUIREMENT: Backend compliance (submission validity)\n"
        f"- Backend is `{backend}` (NVIDIA CUTLASS CuTe DSL: build + JIT-compile kernels in Python).\n"
        "- Your program is INVALID unless it defines at least one CuTe kernel "
        "(a function decorated with `@cute.kernel`) launched from a `@cute.jit` host function, "
        "JIT-compiled via `cute.compile(...)`.\n"
        "- You must call the compiled CuTe kernel from `ModelNew.forward(...)`; "
        "the kernel output must contribute to the returned output (not a dead/no-op side call).\n"
        "\n"
        "#### CuTe DSL compliance (submission invalid otherwise)\n"
        "- `import cutlass` and `import cutlass.cute as cute`.\n"
        "- Convert torch tensors to CuTe tensors with "
        "`from cutlass.cute.runtime import from_dlpack` "
        "(e.g. `from_dlpack(t, assumed_align=16).mark_layout_dynamic()`).\n"
        "- Define the GPU kernel with `@cute.kernel`; get thread/block indices via "
        "`cute.arch.thread_idx()` / `cute.arch.block_idx()`; launch it from a `@cute.jit` host "
        "function with `kernel(...).launch(grid=[...], block=[...])`.\n"
        "- Compile once with `compiled = cute.compile(jit_fn, *cute_tensors)` and reuse it across "
        "calls; the compiled object runs on torch's current CUDA stream by default.\n"
        "- Use `mark_layout_dynamic()` so a single compile generalizes across input shapes, and "
        "cache the compiled kernel on the module (recompiling every forward is NON-COMPLIANT-slow).\n"
        "- Guard non-divisible tile boundaries with a predicate built from "
        "`cute.make_identity_tensor(...)` + `cute.elem_less(...)` so out-of-bounds threads are masked.\n"
        "- Defining a CuTe kernel but never launching/calling it from forward is NON-COMPLIANT.\n"
        "\n"
        "#### Common CuTe mistakes that make the program FAIL (avoid these)\n"
        "- `from_dlpack(t)` raises `BufferError` on tensors that require grad. Convert "
        "`nn.Parameter`/grad tensors with `t.detach()` (and `.contiguous()` if needed) FIRST, "
        "e.g. `from_dlpack(self.bias.detach().contiguous(), assumed_align=16)`.\n"
        "- Use ONLY real arch intrinsics: `cute.arch.thread_idx()`, `cute.arch.block_idx()`, "
        "`cute.arch.block_dim()`, `cute.arch.grid_dim()` (each returns a 3-tuple `(x, y, z)`). "
        "There is NO `cute.arch.block_size()` / `cute.arch.thread_id()` — inventing intrinsics fails to compile.\n"
        "- Prefer the tiling + copy-atom idiom (`make_layout_tv` -> `zipped_divide` -> "
        "`make_fragment_like` -> `cute.copy`) over ad-hoc Python scalar indexing of `cute.Tensor` "
        "inside the kernel; scalar `tensor[i]` access does not map to general strided GPU loads.\n"
        "- Build the host launch grid/block as 3-element lists of ints, and compute occupancy from "
        "`cute.size(...)`/the TV layout, not from raw Python shape math that ignores tiling.\n"
        "- There is NO `cute.cast(...)`. For higher-precision accumulation, allocate the compute "
        "fragment in the wider type: `acc = cute.make_fragment(thr.shape, cutlass.Float32)`, run the "
        "elementwise math on `frg.load()` (the loaded register values follow the fragment dtype), then "
        "store back into the output fragment (whose dtype matches the output tensor). DSL type names are "
        "capitalized: `cutlass.Float32`, `cutlass.Float16`, `cutlass.BFloat16` (not `cutlass.float32`).\n"
        "- A per-channel bias/scalar that varies along a tensor axis is awkward to gather by scalar index "
        "inside the tiled kernel. Prefer broadcasting it into the elementwise operand on the torch side "
        "(e.g. pass an already-broadcast bias tensor, or fold per-channel affine into a second input "
        "tensor) so the kernel stays a pure tiled elementwise pass over equal-shaped operands.\n"
        "- `cute.where`, comparisons and arithmetic require BOTH operands to be the SAME dtype. Python "
        "float literals (`0.0`, `1.0`, a clamp/scale constant) are `Float32`, but values loaded from a "
        "bf16/fp16 tensor are bf16/fp16 — mixing them raises `ValueError: x and y must have the same "
        "dtype (BFloat16 and Float32)`. When fusing clamp/scale/bias, build constants in the data dtype "
        "(`hi = cutlass.BFloat16(1.0)`) or cast the register to `cutlass.Float32` for the math and cast "
        "the result back before storing. This is the #1 reason a correct-looking fused epilogue fails.\n"
    )


def build_task_description_cute(
    *,
    run_cfg: dict[str, Any],
    ref_arch_src: str,
    ref_model_class_src: str,
    ref_inputs_init_src: str,
) -> str:
    """
    Build full task_description.txt for the cute backend.
    Same overall shape as cuda_backend_utils.build_task_description_cuda_inline,
    with cute-specific goal and compliance.
    """
    from kernel_evo.core.code.python_backend_utils import (
        REF_INPUTS_BEGIN,
        REF_INPUTS_END,
        REF_MODEL_BEGIN,
        REF_MODEL_END,
    )

    precision = str(run_cfg.get("precision", "fp32"))
    runtime_precision = str(run_cfg.get("runtime_precision", precision))
    fp8_cute_guidance = ""
    if precision == "fp8" and runtime_precision != "fp8":
        fp8_cute_guidance = (
            "- For fp8 targets, keep Python/module I/O in the runtime precision but use genuine "
            "fp8 paths inside the CuTe kernel where possible "
            "(e.g. fp8 element types `cutlass.Float8E4M3FN`/`cutlass.Float8E5M2`, fp8-friendly "
            "copies/MMA), with accumulation in a wider type (fp32) when needed.\n"
        )

    return (
        _precision_contract_block(run_cfg)
        +
        "\n"
        "### Goal\n"
        "Evolve a program that is **Python source code** implementing the torch.nn.Module interface.\n"
        "The code must implement its kernel(s) with the **NVIDIA CUTLASS CuTe DSL** "
        "(`cutlass.cute`), JIT-compiled via `cute.compile`.\n"
        "The code must:\n"
        "- Be correct (matches the reference outputs)\n"
        "- Be more efficient (lower runtime than the reference on GPU)\n"
        "- Use the cute backend (a `@cute.kernel` launched from a `@cute.jit` host fn)\n"
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
        "- Replace dominant Torch ops with your own CuTe kernels.\n"
        "- Prefer fusion: fewer kernel launches, fewer intermediate tensors.\n"
        "- Use vectorized copies (128-bit copy atoms, e.g. `vector_size = 128 // dtype.width`), "
        "good thread/value tiling (`cute.make_layout_tv`), and register-resident compute.\n"
        "- Compile once and cache the compiled kernel on the module; recompiling each forward is slow.\n"
        f"{fp8_cute_guidance}"
        "\n"
        "### Insight traceability (IMPORTANT)\n"
        "- Program insights will be prefixed with IDs like `[id:<type>_<nn>] ...`.\n"
        "- When asked for `insights_used`, copy the insight strings **verbatim**, including the `[id:...]` prefix.\n"
        "\n"
        "### Mutation justification expectation\n"
        "When asked for a justification, include a short **kernel plan**: "
        "which Torch ops were replaced, which CuTe kernels were added, "
        "and why this reduces launches / memory traffic.\n"
        "\n"
        "### What your evolving program must do (IMPORTANT)\n"
        "Your evolving program is stored as `Program.code` and is evaluated *directly*.\n"
        "\n"
        "- Your program must be a **single self-contained Python module**.\n"
        "- It must define: `class ModelNew(torch.nn.Module): ...` with the same interface as the reference model.\n"
        "\n"
        + get_cute_compliance_block(run_cfg)
        + "\n"
        "### COMPLETE working CuTe DSL pattern (verified; adapt this exact structure)\n"
        "This is a known-good elementwise kernel. CuTe is NOT CUDA-C: do NOT scalar-index a "
        "`cute.Tensor` with a runtime thread index (`y[idx]` raises a DSLRuntimeError). Use the "
        "tiling + copy-atom + fragment idiom shown here, with identity-tensor predication for "
        "non-divisible sizes. To fuse an elementwise epilogue, change only the register compute line.\n"
        "```python\n"
        "import torch\n"
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "from cutlass.cute.runtime import from_dlpack\n"
        "\n"
        "@cute.kernel\n"
        "def _elementwise_kernel(gA, gB, gC, cC, shape, thr_layout, val_layout):\n"
        "    tidx, _, _ = cute.arch.thread_idx()\n"
        "    bidx, _, _ = cute.arch.block_idx()\n"
        "    coord = ((None, None), bidx)\n"
        "    blkA, blkB, blkC, blkCrd = gA[coord], gB[coord], gC[coord], cC[coord]\n"
        "    ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)\n"
        "    st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)\n"
        "    tcA = cute.make_tiled_copy_tv(ld, thr_layout, val_layout)\n"
        "    tcC = cute.make_tiled_copy_tv(st, thr_layout, val_layout)\n"
        "    sA, sC = tcA.get_slice(tidx), tcC.get_slice(tidx)\n"
        "    thrA, thrB, thrC, thrCrd = sA.partition_S(blkA), sA.partition_S(blkB), "
        "sC.partition_S(blkC), sC.partition_S(blkCrd)\n"
        "    frgA, frgB, frgC = cute.make_fragment_like(thrA), cute.make_fragment_like(thrB), "
        "cute.make_fragment_like(thrC)\n"
        "    pred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)\n"
        "    for i in range(cute.size(pred)):\n"
        "        pred[i] = cute.elem_less(thrCrd[i], shape)\n"
        "    cute.copy(ld, thrA, frgA, pred=pred)\n"
        "    cute.copy(ld, thrB, frgB, pred=pred)\n"
        "    frgC.store(frgA.load() + frgB.load())   # <-- fuse your epilogue here (clamp/scale/bias/...)\n"
        "    cute.copy(st, frgC, thrC, pred=pred)\n"
        "\n"
        "@cute.jit\n"
        "def _launch(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):\n"
        "    vec = copy_bits // mA.element_type.width\n"
        "    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))\n"
        "    val_layout = cute.make_ordered_layout((4, vec), order=(1, 0))\n"
        "    tiler, _ = cute.make_layout_tv(thr_layout, val_layout)\n"
        "    gA, gB, gC = cute.zipped_divide(mA, tiler), cute.zipped_divide(mB, tiler), cute.zipped_divide(mC, tiler)\n"
        "    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler)\n"
        "    _elementwise_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(\n"
        "        grid=[cute.size(gC, mode=[1]), 1, 1], block=[cute.size(thr_layout), 1, 1])\n"
        "\n"
        "class ModelNew(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self._compiled = None\n"
        "    def forward(self, a, b):\n"
        "        a, b = a.contiguous(), b.contiguous()   # 2D, same dtype; reshape ND tensors to 2D first\n"
        "        c = torch.empty_like(a)\n"
        "        # IMPORTANT: detach grad/Parameter tensors before from_dlpack (else BufferError)\n"
        "        ca, cb, cc = (from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic() for t in (a, b, c))\n"
        "        if self._compiled is None:\n"
        "            self._compiled = cute.compile(_launch, ca, cb, cc)\n"
        "        self._compiled(ca, cb, cc)\n"
        "        return c\n"
        "```\n"
        "\n"
        "During evaluation, the validator will call:\n"
        "`kernelbench.eval.eval_kernel_against_ref(ref_arch_src, program_code, backend='cute', precision=...)`\n"
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
