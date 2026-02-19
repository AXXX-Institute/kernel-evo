#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from kernel_evo.core.eval.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.dataset import construct_kernelbench_dataset


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class EvalSummary:
    compiled: bool
    correctness: bool
    runtime_us: float | None
    ref_runtime_us: float | None
    speedup_vs_ref: float | None
    metadata: dict[str, Any]


def _to_eval_summary(result: Any) -> EvalSummary:
    compiled = bool(getattr(result, "compiled", False))
    correctness = bool(getattr(result, "correctness", False))
    runtime = getattr(result, "runtime", None)
    ref_runtime = getattr(result, "ref_runtime", None)

    runtime_us = float(runtime) if runtime is not None and float(runtime) > 0 else None
    ref_runtime_us = float(ref_runtime) if ref_runtime is not None and float(ref_runtime) > 0 else None
    speedup = (ref_runtime_us / runtime_us) if (runtime_us and ref_runtime_us) else None

    md = getattr(result, "metadata", None)
    metadata = md if isinstance(md, dict) else {}
    return EvalSummary(
        compiled=compiled,
        correctness=correctness,
        runtime_us=runtime_us,
        ref_runtime_us=ref_runtime_us,
        speedup_vs_ref=speedup,
        metadata=metadata,
    )


@dataclass(frozen=True)
class CompareResult:
    sum_a: EvalSummary
    sum_b: EvalSummary
    problem_meta: dict[str, Any]
    program_a_src: str
    program_b_src: str
    program_a_path: Path
    program_b_path: Path


@dataclass(frozen=True)
class EvalConfig:
    backend: str
    precision: str
    timing_method: str
    num_correct_trials: int
    num_perf_trials: int
    seed: int
    measure_perf: bool
    output_rtol: float | None
    output_atol: float | None
    device: str


def _resolve_problem_file(problem_path: str) -> Path:
    pp = Path(problem_path).expanduser().resolve()
    if pp.is_dir():
        candidate = pp / "task.py"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Custom problem dir must contain task.py: {pp}")
    if not pp.exists():
        raise FileNotFoundError(f"Custom problem file not found: {pp}")
    return pp


def run_compare(
    program_a_path: Path,
    program_b_path: Path,
    *,
    problem_path: str | None = None,
    level: int | None = None,
    problem_id: int | None = None,
    dataset_src: str = "huggingface",
    dataset_name: str = "ScalingIntelligence/KernelBench",
    problem_dir: Path | None = None,
    eval_config: EvalConfig,
) -> CompareResult:
    """Run KernelBench evaluation for two programs and return summaries. No I/O beyond reading files."""
    program_a_path = program_a_path.expanduser().resolve()
    program_b_path = program_b_path.expanduser().resolve()
    if not program_a_path.exists():
        raise FileNotFoundError(f"Missing program A: {program_a_path}")
    if not program_b_path.exists():
        raise FileNotFoundError(f"Missing program B: {program_b_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; KernelBench eval requires a GPU.")

    if problem_path and str(problem_path).strip():
        problem_file = _resolve_problem_file(str(problem_path))
        ref_arch_src = read_text(problem_file)
        problem_meta = {"kind": "custom", "problem_path": str(problem_file)}
    else:
        if level is None or problem_id is None:
            raise ValueError("Must provide either problem_path OR both level and problem_id.")

        ds = construct_kernelbench_dataset(
            level=int(level),
            source=dataset_src,
            dataset_name=dataset_name,
        )
        kb_problem = ds.get_problem_by_id(int(problem_id))
        ref_arch_src = kb_problem.code
        problem_meta = {
            "kind": "kernelbench",
            "dataset_src": dataset_src,
            "dataset_name": dataset_name,
            "level": int(level),
            "problem_id": int(problem_id),
        }

    program_a_src = read_text(program_a_path)
    program_b_src = read_text(program_b_path)
    if " Model(" in program_a_src:
        program_a_src = program_a_src.replace(" Model", " ModelNew")
    if " Model(" in program_b_src:
        program_b_src = program_b_src.replace(" Model", " ModelNew")

    precision = get_torch_dtype_from_string(eval_config.precision)
    device = torch.device(eval_config.device)

    res_a = eval_kernel_against_ref(
        ref_arch_src,
        program_a_src,
        seed_num=eval_config.seed,
        num_correct_trials=eval_config.num_correct_trials,
        num_perf_trials=eval_config.num_perf_trials,
        measure_performance=eval_config.measure_perf,
        timing_method=eval_config.timing_method,
        verbose=False,
        backend=eval_config.backend,
        precision=precision,
        output_rtol=eval_config.output_rtol,
        output_atol=eval_config.output_atol,
        device=device,
    )
    if res_a is None:
        raise RuntimeError("Program A evaluation returned None. Retry.")

    res_b = eval_kernel_against_ref(
        ref_arch_src,
        program_b_src,
        seed_num=eval_config.seed,
        num_correct_trials=eval_config.num_correct_trials,
        num_perf_trials=eval_config.num_perf_trials,
        measure_performance=eval_config.measure_perf,
        timing_method=eval_config.timing_method,
        verbose=False,
        backend=eval_config.backend,
        precision=precision,
        output_rtol=eval_config.output_rtol,
        output_atol=eval_config.output_atol,
        device=device,
    )
    if res_b is None:
        raise RuntimeError("Program B evaluation returned None. Retry.")

    return CompareResult(
        sum_a=_to_eval_summary(res_a),
        sum_b=_to_eval_summary(res_b),
        problem_meta=problem_meta,
        program_a_src=program_a_src,
        program_b_src=program_b_src,
        program_a_path=program_a_path,
        program_b_path=program_b_path,
    )
