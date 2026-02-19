import argparse
import difflib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from kernel_evo.core.program.compare import (
    EvalConfig,
    EvalSummary,
    run_compare,
)


def _default_problem_dir() -> Path:
    """Kernel-evo repo root: works with or without __file__ (e.g. under exec)."""
    def find_root(start: Path, max_up: int = 8) -> Path | None:
        p = start.resolve()
        for _ in range(max_up):
            if not p.exists() or p == p.parent:
                break
            if (p / "pyproject.toml").exists():
                return p
            p = p.parent
        return None

    try:
        start = Path(__file__).resolve().parent
        root = find_root(start)
        if root is not None:
            return root
    except NameError:
        pass
    root = find_root(Path.cwd())
    return root if root is not None else Path.cwd()


def _shorten(s: str, n: int = 120) -> str:
    s = str(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


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


def setup_parser(subparsers: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "compare",
        description=(
            "Compare two extracted KernelBench programs by running both against the same "
            "reference problem and printing correctness + runtime deltas."
        ),
    )
    parser.add_argument("--program-a", required=True, help="Path to first program .py file")
    parser.add_argument("--program-b", required=True, help="Path to second program .py file")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--show-diff", action="store_true", help="Print unified diff of sources")

    parser.add_argument(
        "--problem-path",
        default="",
        help="Path to custom problem file or dir with task.py (bypasses KernelBench dataset).",
    )
    parser.add_argument("--dataset-src", default="huggingface", choices=["huggingface", "local"])
    parser.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--problem-id", type=int, default=None)

    parser.add_argument("--backend", default="cuda", choices=["cuda", "triton", "tilelang", "cute"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--timing-method", default="cuda_event")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-rtol", type=float, default=None)
    parser.add_argument("--output-atol", type=float, default=None)
    parser.add_argument("--no-perf", action="store_true", help="Skip performance timing")
    parser.add_argument("--json-out", default="", help="Path to write JSON report")

    parser.add_argument(
        "--problem-dir",
        default="",
        help="Path to kernel_evo repo root (default: auto-detect via pyproject.toml from this file or cwd).",
    )
    parser.add_argument("--device", default="cuda:7", help="Device for evaluation")


def compare(args: argparse.Namespace) -> None:
    program_a_path = Path(args.program_a).expanduser().resolve()
    program_b_path = Path(args.program_b).expanduser().resolve()

    if args.problem_dir and str(args.problem_dir).strip():
        problem_dir = Path(args.problem_dir).expanduser().resolve()
    else:
        problem_dir = _default_problem_dir()

    problem_path = str(args.problem_path).strip() or None
    eval_config = EvalConfig(
        backend=args.backend,
        precision=args.precision,
        timing_method=args.timing_method,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        seed=args.seed,
        measure_perf=not args.no_perf,
        output_rtol=float(args.output_rtol) if args.output_rtol is not None else None,
        output_atol=float(args.output_atol) if args.output_atol is not None else None,
        device=args.device,
    )

    result = run_compare(
        program_a_path,
        program_b_path,
        problem_path=problem_path,
        level=args.level,
        problem_id=args.problem_id,
        dataset_src=args.dataset_src,
        dataset_name=args.dataset_name,
        problem_dir=problem_dir,
        eval_config=eval_config,
    )

    if args.show_diff:
        print("== Program diff (A -> B) ==")
        a_lines = result.program_a_src.splitlines(keepends=True)
        b_lines = result.program_b_src.splitlines(keepends=True)
        diff = difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=str(result.program_a_path),
            tofile=str(result.program_b_path),
            n=3,
        )
        sys.stdout.writelines(diff)
        print("\n")

    problem_label = (
        f"custom:{result.problem_meta.get('problem_path', '')}"
        if result.problem_meta.get("kind") == "custom"
        else f"kernelbench:level={result.problem_meta.get('level')} problem_id={result.problem_meta.get('problem_id')}"
    )
    print(
        "Running KernelBench eval for both programs with settings:\n"
        f"- problem={problem_label}\n"
        f"- backend={args.backend} precision={args.precision}\n"
        f"- num_correct_trials={args.num_correct_trials} "
        f"num_perf_trials={args.num_perf_trials} "
        f"measure_perf={eval_config.measure_perf}\n"
    )

    _print_summary(args.label_a, result.sum_a)
    _print_summary(args.label_b, result.sum_b)
    _print_comparison(args.label_a, result.sum_a, args.label_b, result.sum_b)

    if args.json_out and str(args.json_out).strip():
        out_path = Path(args.json_out).expanduser().resolve()
        payload = {
            "problem": result.problem_meta,
            "settings": {
                "backend": args.backend,
                "precision": args.precision,
                "timing_method": args.timing_method,
                "num_correct_trials": args.num_correct_trials,
                "num_perf_trials": args.num_perf_trials,
                "seed": args.seed,
                "measure_perf": eval_config.measure_perf,
                "output_rtol": eval_config.output_rtol,
                "output_atol": eval_config.output_atol,
            },
            "programs": {
                "a": {"label": args.label_a, "path": str(result.program_a_path), "summary": asdict(result.sum_a)},
                "b": {"label": args.label_b, "path": str(result.program_b_path), "summary": asdict(result.sum_b)},
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON report: {out_path}")
