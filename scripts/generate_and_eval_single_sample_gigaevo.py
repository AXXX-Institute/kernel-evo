from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _add_kernelbench_to_sys_path(problem_dir: Path) -> None:
    kb_src = problem_dir / "KernelBench" / "src"
    if kb_src.exists():
        s = str(kb_src)
        if s not in sys.path:
            sys.path.insert(0, s)


def _write_initial_seed(problem_dir: Path, *, program_code: str, note: str) -> None:
    seed_path = problem_dir / "initial_programs" / "seed.py"
    seed_code = (
        "# Auto-generated seed program.\n"
        f"# {note}\n"
        "#\n"
        "# IMPORTANT: this file is evaluated directly (no entrypoint wrapper).\n"
        "# It must define `class ModelNew(torch.nn.Module)`.\n"
        "\n"
        + program_code.rstrip()
        + "\n"
    )
    seed_path.write_text(seed_code, encoding="utf-8")

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "KernelBench single-sample run using GigaEvo.\n"
            "This script writes a backend-specific task_description.txt + seed.py, "
            "then runs gigaevo-core-internal/run.py for 1 generation (seed + 1 mutant)."
        )
    )

    # KernelBench problem selection
    p.add_argument("--dataset-src", default="huggingface", choices=["huggingface", "local"])
    p.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--problem-id", type=int, required=True)

    # KernelBench evaluation settings (passed to validator via context.py/run_config.json)
    p.add_argument(
        "--backend",
        default="cuda",
        choices=["cuda", "triton", "tilelang", "cute", "thunderkittens"],
    )
    p.add_argument(
        "--codegen-kind",
        default="auto",
        choices=["auto", "python", "cpp"],
        help="Which program template to generate. 'auto' uses python for triton/tilelang/cute/thunderkittens and cpp for cuda.",
    )
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--timing-method", default="cuda_event")
    p.add_argument("--num-correct-trials", type=int, default=5)
    p.add_argument("--num-perf-trials", type=int, default=100)

    # GigaEvo / infra
    p.add_argument(
        "--problem-dir",
        default="/home/sivtsov/kernel_generation",
        help="Directory containing metrics.yaml, validate.py, task_description.txt, initial_programs/.",
    )
    p.add_argument(
        "--gigaevo-dir",
        default="/home/sivtsov/gigaevo-core-internal",
        help="Path to gigaevo-core-internal repo (contains run.py and config/).",
    )
    p.add_argument("--experiment", default="base", help="Hydra experiment preset")
    p.add_argument("--redis-db", type=int, default=0)
    p.add_argument("--redis-resume", action="store_true")
    p.add_argument("--validator-debug", action="store_true", help="Enable validator debug logs (writes to kernel_generation/validator_debug/ by default).")
    p.add_argument("--validator-debug-dir", default="", help="Optional directory for validator debug logs.")
    p.add_argument("--validator-debug-max-code-chars", type=int, default=50000, help="Max number of code characters to write into each validator debug log.")

    # LLM config (LangChain ChatOpenAI in gigaevo uses OPENAI_API_KEY + base_url)
    p.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--model-name", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--llm-log-dir", default="", help="If set, start a local proxy that logs all LLM prompts/responses to this directory.")
    p.add_argument("--llm-log-port", type=int, default=8801, help="Port for the local LLM logging proxy.")
    p.add_argument(
        "--disable-insights-lineage",
        action="store_true",
        help=(
            "Disable InsightsStage/LineageStage (LLM analysis stages). This avoids structured-output "
            "parsing failures (invalid JSON / finish_reason=length) that are common on some OpenRouter models."
        ),
    )

    # Single-sample evolution controls
    p.add_argument("--max-generations", type=int, default=1)
    p.add_argument("--max-elites-per-generation", type=int, default=1)
    p.add_argument("--max-mutations-per-generation", type=int, default=1)
    p.add_argument("--num-parents", type=int, default=1)

    args = p.parse_args()

    problem_dir = Path(args.problem_dir).resolve()
    gigaevo_dir = Path(args.gigaevo_dir).resolve()

    if not (problem_dir / "metrics.yaml").exists():
        raise FileNotFoundError(f"Missing {problem_dir}/metrics.yaml")
    if not (problem_dir / "validate.py").exists():
        raise FileNotFoundError(f"Missing {problem_dir}/validate.py")
    if not (problem_dir / "initial_programs").exists():
        raise FileNotFoundError(f"Missing {problem_dir}/initial_programs/")

    # Load reference architecture source from KernelBench
    _add_kernelbench_to_sys_path(problem_dir)
    from kernelbench.dataset import construct_kernelbench_dataset

    ds = construct_kernelbench_dataset(
        level=args.level,
        source=args.dataset_src,
        dataset_name=args.dataset_name,
    )
    kb_problem = ds.get_problem_by_id(args.problem_id)
    ref_arch_src = kb_problem.code

    task_path = problem_dir / "task_description.txt"
    from cpp_backend_utils import is_cpp_backend as _is_cpp_backend
    from python_backend_utils import is_python_backend as _is_python_backend

    # Split ref into class vs inputs/init
    from python_backend_utils import split_kernelbench_ref

    model_src, inputs_src = split_kernelbench_ref(ref_arch_src)

    backend = str(args.backend).lower()
    if str(args.codegen_kind).lower() == "auto":
        codegen_kind = "cpp" if _is_cpp_backend(backend) else "python"
    else:
        codegen_kind = str(args.codegen_kind).lower()
        if codegen_kind == "python" and not _is_python_backend(backend):
            # Allow forcing, but keep a gentle safeguard for obvious mismatches.
            pass

    # Write run config used by context.py + validate.py (no env vars needed)
    run_cfg = {
        "dataset_src": str(args.dataset_src),
        "dataset_name": str(args.dataset_name),
        "level": int(args.level),
        "problem_id": int(args.problem_id),
        "backend": str(args.backend),
        "codegen_kind": str(codegen_kind),
        "precision": str(args.precision),
        "timing_method": str(args.timing_method),
        "num_correct_trials": int(args.num_correct_trials),
        "num_perf_trials": int(args.num_perf_trials),
        "ref_arch_src": ref_arch_src,
        "ref_model_class_src": model_src,
        "ref_inputs_init_src": inputs_src,
        "validator_debug": bool(args.validator_debug),
        "validator_debug_dir": str(args.validator_debug_dir),
        "validator_debug_max_code_chars": int(args.validator_debug_max_code_chars),
    }
    (problem_dir / "run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Write the initial prompt shown to the mutation LLM (ALL context included here)
    if codegen_kind == "cpp":
        from cpp_backend_utils import build_task_description_cpp, seed_program_cpp_stub

        task_path.write_text(
            build_task_description_cpp(run_cfg=run_cfg, ref_arch_src=ref_arch_src),
            encoding="utf-8",
        )
        seed_program_code = seed_program_cpp_stub()
    else:
        from python_backend_utils import build_task_description_python, model_to_modelnew

        task_text = build_task_description_python(
            run_cfg=run_cfg,
            ref_arch_src=ref_arch_src,
            ref_model_class_src=model_src,
            ref_inputs_init_src=inputs_src,
        )
        if bool(args.validator_debug):
            print(f"Task description (debug):\n\n{task_text}\n\n" + "=" * 100 + "\n\n")
        task_path.write_text(
            task_text,
            encoding="utf-8",
        )
        seed_program_code = model_to_modelnew(model_src)

    # Generate per-problem seed.py (direct python code; no entrypoint)
    _write_initial_seed(
        problem_dir,
        program_code=seed_program_code,
        note=f"Generated from KernelBench level={args.level} problem_id={args.problem_id}",
    )

    # Run GigaEvo for exactly 1 generation
    run_py = gigaevo_dir / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"Could not find {run_py}")

    proxy_proc: subprocess.Popen[str] | None = None
    effective_llm_base_url = str(args.llm_base_url)
    if str(args.llm_log_dir).strip():
        log_dir = Path(str(args.llm_log_dir)).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        proxy_script = problem_dir / "scripts" / "openai_proxy_logger.py"
        if not proxy_script.exists():
            raise FileNotFoundError(f"Missing proxy script: {proxy_script}")

        proxy_cmd = [
            sys.executable,
            str(proxy_script),
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            str(int(args.llm_log_port)),
            "--upstream",
            str(args.llm_base_url),
            "--log-dir",
            str(log_dir),
        ]
        proxy_proc = subprocess.Popen(
            proxy_cmd,
            cwd=str(problem_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Give the proxy a moment to bind.
        time.sleep(0.2)
        effective_llm_base_url = f"http://127.0.0.1:{int(args.llm_log_port)}"

    # Allow KernelGeneration-local Hydra configs (pipeline=kernel_generation_direct, etc.).
    config_dir = (problem_dir / "config").resolve()
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Missing KernelGeneration config directory: {config_dir} "
            "(expected after moving pipeline config out of gigaevo-core-internal)."
        )

    cmd = [
        sys.executable,
        str(run_py),
        f"experiment={args.experiment}",
        f"hydra.searchpath=[file://{config_dir}]",
        "pipeline=kernel_generation_direct",
        "problem.name=kernel_generation",
        f"problem.dir={problem_dir}",
        f"redis.db={args.redis_db}",
        f"redis.resume={'true' if args.redis_resume else 'false'}",
        f"llm_base_url={effective_llm_base_url}",
        f"model_name={args.model_name}",
        f"temperature={args.temperature}",
        f"max_tokens={args.max_tokens}",
        f"max_generations={args.max_generations}",
        f"max_elites_per_generation={args.max_elites_per_generation}",
        f"max_mutations_per_generation={args.max_mutations_per_generation}",
        f"num_parents={args.num_parents}",
    ]

    if bool(args.disable_insights_lineage):
        # Use a kernel_generation-local pipeline builder (no gigaevo-core code changes needed).
        cmd.append(
            "pipeline_builder._target_=kernel_generation.pipeline_builders.DirectCodeContextPipelineNoInsightsLineageBuilder"
        )

    print("Running:")
    print("  " + " ".join(cmd))
    print("")
    print("Notes:")
    print("- Redis must be running (default: localhost:6379).")
    print("- If you get 'Redis database is not empty', either flush it or pass --redis-db <new>.")
    print("")

    try:
        # Ensure `kernel_generation.*` is importable for Hydra instantiation (pipeline builders).
        env = os.environ.copy()
        extra_path = str(problem_dir.parent)
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = f"{extra_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = extra_path

        subprocess.run(cmd, cwd=str(gigaevo_dir), check=True, env=env)
    finally:
        if proxy_proc is not None:
            try:
                proxy_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()


