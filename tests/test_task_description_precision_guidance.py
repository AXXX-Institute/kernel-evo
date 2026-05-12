from __future__ import annotations

from kernel_evo.core.code.cuda_backend_utils import build_task_description_cuda_inline
from kernel_evo.core.code.evolve import build_task_description_for_backend
from kernel_evo.core.code.python_backend_utils import build_task_description_python


RUN_CFG_FP8 = {
    "backend": "triton",
    "precision": "fp8",
    "runtime_precision": "bf16",
}

CUDA_RUN_CFG_FP8 = {
    "backend": "cuda_inline",
    "precision": "fp8",
    "runtime_precision": "bf16",
}


def test_python_task_description_keeps_fp8_as_kernel_target() -> None:
    task_description = build_task_description_python(
        run_cfg=RUN_CFG_FP8,
        ref_arch_src="",
        ref_model_class_src="class Model(torch.nn.Module):\n    pass\n",
        ref_inputs_init_src="",
    )

    assert "Requested kernel precision target is `fp8`." in task_description
    assert (
        "Runtime tensor precision for validator/profiler inputs, module parameters, "
        "and output comparison is `bf16`."
    ) in task_description
    assert "Do NOT reinterpret the task as a `bf16` kernel request." in task_description


def test_cuda_inline_task_description_adds_fp8_kernel_guidance() -> None:
    task_description = build_task_description_cuda_inline(
        run_cfg=CUDA_RUN_CFG_FP8,
        ref_arch_src="",
        ref_model_class_src="class Model(torch.nn.Module):\n    pass\n",
        ref_inputs_init_src="",
    )

    assert "genuine fp8 paths inside CUDA code where possible" in task_description
    assert "load_inline" in task_description
    assert "Triton-first" not in task_description


def test_backend_specific_builder_uses_cuda_inline_task_description() -> None:
    task_description = build_task_description_for_backend(
        run_cfg=CUDA_RUN_CFG_FP8,
        ref_arch_src="",
        ref_model_class_src="class Model(torch.nn.Module):\n    pass\n",
        ref_inputs_init_src="",
    )

    assert "genuine fp8 paths inside CUDA code where possible" in task_description
    assert "The code must use **inline C++/CUDA** compiled on the fly" in task_description
