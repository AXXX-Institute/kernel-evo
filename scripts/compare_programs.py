#!/usr/bin/env python3
from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import difflib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _add_kernelbench_to_sys_path(problem_dir: Path) -> None:
    kb_src = problem_dir / "KernelBench" / "src"
    if kb_src.exists():
        s = str(kb_src)
        if s not in sys.path:
            sys.path.insert(0, s)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _shorten(s: str, n: int = 120) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


@dataclass(frozen=True)
class EvalSummary:
    compiled: bool
    correctness: bool
    runtime_us: float | None
    ref_runtime_us: float | None
    speedup_vs_ref: float | None
    metadata: dict[str, Any]


def _to_eval_summary(result: Any) -> EvalSummary:
    # KernelBench returns pydantic model KernelExecResult
    compiled = bool(getattr(result, "compiled", False))
    correctness = bool(getattr(result, "correctness", False))
    runtime = getattr(result, "runtime", None)
    ref_runtime = getattr(result, "ref_runtime", None)

    runtime_us = float(runtime) if runtime is not None and float(runtime) > 0 else None
    ref_runtime_us = (
        float(ref_runtime) if ref_runtime is not None and float(ref_runtime) > 0 else None
    )
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


def _print_summary(label: str, s: EvalSummary) -> None:
    print(f"== {label} ==")
    print(f"compiled:     {s.compiled}")
    print(f"correctness:  {s.correctness}")
    print(f"runtime_us:   {s.runtime_us if s.runtime_us is not None else 'n/a'}")
    print(f"ref_runtime:  {s.ref_runtime_us if s.ref_runtime_us is not None else 'n/a'}")
    print(f"speedup(ref): {s.speedup_vs_ref if s.speedup_vs_ref is not None else 'n/a'}")
    if s.metadata:
        interesting = {}
        for k in (
            "correctness_trials",
            "max_difference",
            "avg_difference",
            "runtime_error_name",
            "runtime_error",
            "compilation_error_name",
            "compilation_error",
            "excessive_speedup",
            "hardware",
            "device",
        ):
            if k in s.metadata:
                interesting[k] = s.metadata.get(k)
        if interesting:
            print("metadata (selected):")
            for k, v in interesting.items():
                print(f"  - {k}: {_shorten(v, 200)}")
    print("")


def _print_comparison(a_label: str, a: EvalSummary, b_label: str, b: EvalSummary) -> None:
    if a.runtime_us is not None and b.runtime_us is not None:
        faster = a_label if a.runtime_us < b.runtime_us else b_label
        slower = b_label if faster == a_label else a_label
        fast_t = min(a.runtime_us, b.runtime_us)
        slow_t = max(a.runtime_us, b.runtime_us)
        ratio = slow_t / fast_t if fast_t > 0 else float("inf")
        print("== Comparison ==")
        print(f"faster: {faster} ({fast_t:.3f} us)")
        print(f"slower: {slower} ({slow_t:.3f} us)")
        print(f"ratio:  {ratio:.3f}x")
        print("")
    else:
        print("== Comparison ==")
        print("runtime comparison: n/a (one or both runs did not produce runtime)")
        print("")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compare two extracted KernelBench programs by running both against the same "
            "reference problem and printing correctness + runtime deltas."
        )
    )

    # Program inputs
    p.add_argument("--program-a", required=True, help="Path to first program .py file")
    p.add_argument("--program-b", required=True, help="Path to second program .py file")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument(
        "--show-diff",
        action="store_true",
        help="Print a unified diff of the two program sources before evaluation.",
    )

    # KernelBench problem selection (keep identical to generate_and_eval_single_sample_gigaevo.py)
    p.add_argument("--dataset-src", default="huggingface", choices=["huggingface", "local"])
    p.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--problem-id", type=int, required=True)

    # Evaluation settings
    p.add_argument("--backend", default="cuda", choices=["cuda", "triton", "tilelang", "cute"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--timing-method", default="cuda_event")
    p.add_argument("--num-correct-trials", type=int, default=5)
    p.add_argument("--num-perf-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip performance timing (still checks compilation + correctness).",
    )
    p.add_argument(
        "--json-out",
        default="",
        help="Optional path to write a JSON report with both summaries and raw metadata.",
    )

    # Infra
    p.add_argument(
        "--problem-dir",
        default="",
        help="Path to kernel_generation directory (defaults to repo root next to this script).",
    )

    args = p.parse_args()

    program_a_path = Path(args.program_a).expanduser().resolve()
    program_b_path = Path(args.program_b).expanduser().resolve()
    if not program_a_path.exists():
        raise FileNotFoundError(f"Missing --program-a: {program_a_path}")
    if not program_b_path.exists():
        raise FileNotFoundError(f"Missing --program-b: {program_b_path}")

    if str(args.problem_dir).strip():
        problem_dir = Path(args.problem_dir).expanduser().resolve()
    else:
        # kernel_generation/scripts/compare_programs.py -> kernel_generation/
        problem_dir = Path(__file__).resolve().parents[1]

    _add_kernelbench_to_sys_path(problem_dir)

    # Import after sys.path fix
    import torch  # noqa: PLC0415
    from kernelbench.dataset import construct_kernelbench_dataset  # noqa: PLC0415
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; KernelBench eval requires a GPU.")

    ds = construct_kernelbench_dataset(
        level=int(args.level),
        source=str(args.dataset_src),
        dataset_name=str(args.dataset_name),
    )
    kb_problem = ds.get_problem_by_id(int(args.problem_id))
    ref_arch_src = kb_problem.code

    program_a_src = _read_text(program_a_path)
    program_b_src = _read_text(program_b_path)

    if bool(args.show_diff):
        print("== Program diff (A -> B) ==")
        a_lines = program_a_src.splitlines(keepends=True)
        b_lines = program_b_src.splitlines(keepends=True)
        diff = difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=str(program_a_path),
            tofile=str(program_b_path),
            n=3,
        )
        sys.stdout.writelines(diff)
        print("\n")

    precision = get_torch_dtype_from_string(str(args.precision))
    measure_perf = not bool(args.no_perf)

    print(
        "Running KernelBench eval for both programs with settings:\n"
        f"- level={int(args.level)} problem_id={int(args.problem_id)} source={args.dataset_src}\n"
        f"- backend={args.backend} precision={args.precision}\n"
        f"- num_correct_trials={int(args.num_correct_trials)} num_perf_trials={int(args.num_perf_trials)} measure_perf={measure_perf}\n"
    )

    # Important: eval_kernel_against_ref generates inputs internally from ref code,
    # so both A and B see identical input distributions (same seed and same ref get_inputs()).
    res_a = eval_kernel_against_ref(
        ref_arch_src,
        program_a_src,
        seed_num=int(args.seed),
        num_correct_trials=int(args.num_correct_trials),
        num_perf_trials=int(args.num_perf_trials),
        measure_performance=measure_perf,
        timing_method=str(args.timing_method),
        verbose=False,
        backend=str(args.backend),
        precision=precision,
    )
    if res_a is None:
        raise RuntimeError("Program A evaluation returned None (likely a lock/concurrent compilation issue). Retry.")

    res_b = eval_kernel_against_ref(
        ref_arch_src,
        program_b_src,
        seed_num=int(args.seed),
        num_correct_trials=int(args.num_correct_trials),
        num_perf_trials=int(args.num_perf_trials),
        measure_performance=measure_perf,
        timing_method=str(args.timing_method),
        verbose=False,
        backend=str(args.backend),
        precision=precision,
    )
    if res_b is None:
        raise RuntimeError("Program B evaluation returned None (likely a lock/concurrent compilation issue). Retry.")

    sum_a = _to_eval_summary(res_a)
    sum_b = _to_eval_summary(res_b)

    _print_summary(args.label_a, sum_a)
    _print_summary(args.label_b, sum_b)
    _print_comparison(args.label_a, sum_a, args.label_b, sum_b)

    if str(args.json_out).strip():
        out_path = Path(args.json_out).expanduser().resolve()
        payload = {
            "problem": {
                "dataset_src": str(args.dataset_src),
                "dataset_name": str(args.dataset_name),
                "level": int(args.level),
                "problem_id": int(args.problem_id),
            },
            "settings": {
                "backend": str(args.backend),
                "precision": str(args.precision),
                "timing_method": str(args.timing_method),
                "num_correct_trials": int(args.num_correct_trials),
                "num_perf_trials": int(args.num_perf_trials),
                "seed": int(args.seed),
                "measure_perf": bool(measure_perf),
            },
            "programs": {
                "a": {
                    "label": str(args.label_a),
                    "path": str(program_a_path),
                    "summary": asdict(sum_a),
                },
                "b": {
                    "label": str(args.label_b),
                    "path": str(program_b_path),
                    "summary": asdict(sum_b),
                },
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")


if __name__ == "__main__":
    main()
