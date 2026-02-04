from __future__ import annotations

import argparse
import json
import os
import socket
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


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        # optional: flush so output appears immediately
        self.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


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

    # Problem selection
    # - KernelBench: use (--level, --problem-id)
    # - Custom: use --problem-path pointing at a KernelBench-style `task.py`
    p.add_argument("--dataset-src", default="huggingface", choices=["huggingface", "local"])
    p.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--problem-id", type=int, default=None)
    p.add_argument(
        "--problem-path",
        default="",
        help="Optional: path to a custom problem file or a directory containing `task.py` (bypasses KernelBench dataset loading).",
    )
    p.add_argument("--experiment-name", required=True, help="Name used for saving the experiment results.")

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
    p.add_argument(
        "--output-rtol",
        type=float,
        default=0.5,
        help=(
            "Relative tolerance for output correctness checks. If unset, KernelBench defaults based on precision."
        ),
    )
    p.add_argument(
        "--output-atol",
        type=float,
        default=0.5,
        help=(
            "Absolute tolerance for output correctness checks. If unset, KernelBench defaults based on precision."
        ),
    )

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
    p.add_argument("--stdout-dir", default="", help="Optional directory to store stdout/stderr logs.")

    # Execution controls
    p.add_argument(
        "--execution-mode",
        default="local_execution",
        choices=["local_execution", "remote_execution"],
        help="Whether to run validation locally or on a remote server.",
    )
    p.add_argument(
        "--remote-validator-url",
        default="http://localhost:15000",
        help="URL of the remote validation server (used if execution-mode is remote_execution).",
    )
    p.add_argument(
        "--remote-poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for remote execution.",
    )

    # LLM config (LangChain ChatOpenAI in gigaevo uses OPENAI_API_KEY + base_url)
    p.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--model-name", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=100000)
    p.add_argument("--llm-log-dir", default="", help="If set, start a local proxy that logs all LLM prompts/responses to this directory.")
    p.add_argument("--llm-log-port", type=int, default=None, help="Port for the local LLM logging proxy.")
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
    p.add_argument(
        "--use-memory-for-errors",
        action="store_true",
        default=False,
        help="Enable memory for errors in RepairStage (default: False).",
    )

    args = p.parse_args()

    problem_dir = Path(args.problem_dir).resolve()
    gigaevo_dir = Path(args.gigaevo_dir).resolve()

    if not (problem_dir / "metrics.yaml").exists():
        raise FileNotFoundError(f"Missing {problem_dir}/metrics.yaml")
    if not (problem_dir / "validate.py").exists():
        raise FileNotFoundError(f"Missing {problem_dir}/validate.py")
    if not (problem_dir / "initial_programs").exists():
        raise FileNotFoundError(f"Missing {problem_dir}/initial_programs/")

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

    # Load reference architecture source
    if str(args.problem_path).strip():
        problem_file = _resolve_problem_file(str(args.problem_path))
        ref_arch_src = problem_file.read_text(encoding="utf-8")
        problem_kind = "custom"
    else:
        if args.level is None or args.problem_id is None:
            raise SystemExit("Must provide either --problem-path OR both --level and --problem-id.")

        _add_kernelbench_to_sys_path(problem_dir)
        from kernelbench.dataset import construct_kernelbench_dataset  # type: ignore[import-not-found]

        ds = construct_kernelbench_dataset(
            level=args.level,
            source=args.dataset_src,
            dataset_name=args.dataset_name,
        )
        kb_problem = ds.get_problem_by_id(args.problem_id)
        ref_arch_src = kb_problem.code
        problem_kind = "kernelbench"

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

    formatted_time = time.strftime("%Y%m%d_%H%M%S")
    if args.problem_path:
        problem_name = f"{args.experiment_name}_{formatted_time}"
    else:
        problem_name = f"kernelbench_{args.level}_{args.problem_id}_{formatted_time}"

    if str(args.stdout_dir).strip():
        stdout_dir = Path(args.stdout_dir).expanduser().resolve()
        stdout_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = stdout_dir / f"{problem_name}.txt"
        f = open(log_file_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, f)
        sys.stderr = Tee(sys.__stderr__, f)

    # Write run config used by context.py + validate.py (no env vars needed)
    run_cfg = {
        "dataset_src": str(args.dataset_src),
        "dataset_name": str(args.dataset_name),
        "level": int(args.level) if args.level is not None else 0,
        "problem_id": int(args.problem_id) if args.problem_id is not None else 0,
        "problem_kind": problem_kind,
        "problem_path": str(args.problem_path) if str(args.problem_path).strip() else "",
        "backend": str(args.backend),
        "codegen_kind": str(codegen_kind),
        "precision": str(args.precision),
        "timing_method": str(args.timing_method),
        "num_correct_trials": int(args.num_correct_trials),
        "num_perf_trials": int(args.num_perf_trials),
        "output_rtol": (float(args.output_rtol) if args.output_rtol is not None else None),
        "output_atol": (float(args.output_atol) if args.output_atol is not None else None),
        "ref_arch_src": ref_arch_src,
        "ref_model_class_src": model_src,
        "ref_inputs_init_src": inputs_src,
        "validator_debug": bool(args.validator_debug),
        "validator_debug_dir": f"{args.validator_debug_dir}/{problem_name}",
        "validator_debug_max_code_chars": int(args.validator_debug_max_code_chars),
        "execution_mode": str(args.execution_mode),
        "remote_validator_url": str(args.remote_validator_url),
        "remote_poll_interval": float(args.remote_poll_interval),
        "use_memory_for_errors": bool(args.use_memory_for_errors),
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
        llm_log_dir = f"{args.llm_log_dir}/{problem_name}"
        log_dir = Path(llm_log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        proxy_script = problem_dir / "scripts" / "openai_proxy_logger.py"
        if not proxy_script.exists():
            raise FileNotFoundError(f"Missing proxy script: {proxy_script}")

        if args.llm_log_port is not None:
            llm_log_port = int(args.llm_log_port)
        else:
            # Pick a random free port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                llm_log_port = s.getsockname()[1]

        proxy_cmd = [
            sys.executable,
            str(proxy_script),
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            str(llm_log_port),
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
        effective_llm_base_url = f"http://127.0.0.1:{llm_log_port}"

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
        f"problem.name={problem_name}",
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

        # If stdout_dir is set, we use Popen and manually tee the output from the subprocess.
        if str(args.stdout_dir).strip():
            proc = subprocess.Popen(
                cmd,
                cwd=str(gigaevo_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # We are already inside the main process which has sys.stdout redirected to Tee.
            # Reading from proc.stdout and printing will automatically write to both terminal and log file.
            if proc.stdout:
                for line in proc.stdout:
                    print(line, end="", flush=True)
            
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        else:
            subprocess.run(cmd, cwd=str(gigaevo_dir), check=True, env=env)
    finally:
        if proxy_proc is not None:
            try:
                proxy_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()


