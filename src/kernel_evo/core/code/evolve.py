import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kernel_evo.core.precision import resolve_runtime_precision_string
from kernel_evo.resources.prompt_loader import prepare_prompts_for_experiment
from kernel_evo.resources.paths import get_repo_root, get_resources_dir
from kernel_evo.resources.workspace import prepare_problem_workspace


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
        "\n" + program_code.rstrip() + "\n"
    )
    seed_path.write_text(seed_code, encoding="utf-8")


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


def _get_gigaevo_dir() -> Path:
    """Bundled gigaevo entrypoint with config files and the run script."""
    gigaevo_dir = get_resources_dir() / "gigaevo"
    if not (gigaevo_dir / "run.py").exists():
        raise FileNotFoundError(f"Bundled run.py not found at {gigaevo_dir / 'run.py'}")
    return gigaevo_dir


# Subdirs inside each experiment directory (under --log-dir / <problem_name>)
LOG_SUBDIR_VALIDATE = "validate_logs"
LOG_SUBDIR_TRACES = "traces"
LOG_SUBDIR_TENSORBOARD = "tensorboard"
LOG_SUBDIR_ARTIFACTS = "artifacts"


def build_task_description_for_backend(
    *,
    run_cfg: dict[str, Any],
    ref_arch_src: str,
    ref_model_class_src: str,
    ref_inputs_init_src: str,
) -> str:
    from kernel_evo.core.code.cuda_backend_utils import build_task_description_cuda_inline, is_cuda_inline_backend
    from kernel_evo.core.code.cute_backend_utils import build_task_description_cute, is_cute_backend
    from kernel_evo.core.code.python_backend_utils import build_task_description_python

    backend = str(run_cfg.get("backend", "")).lower().strip()
    if is_cuda_inline_backend(backend):
        return build_task_description_cuda_inline(
            run_cfg=run_cfg,
            ref_arch_src=ref_arch_src,
            ref_model_class_src=ref_model_class_src,
            ref_inputs_init_src=ref_inputs_init_src,
        )
    if is_cute_backend(backend):
        return build_task_description_cute(
            run_cfg=run_cfg,
            ref_arch_src=ref_arch_src,
            ref_model_class_src=ref_model_class_src,
            ref_inputs_init_src=ref_inputs_init_src,
        )
    return build_task_description_python(
        run_cfg=run_cfg,
        ref_arch_src=ref_arch_src,
        ref_model_class_src=ref_model_class_src,
        ref_inputs_init_src=ref_inputs_init_src,
    )


def run_evolve(args: Any) -> None:
    """Run GigaEvo evolution: write task_description + seed, then run gigaevo run.py."""
    # Fail fast on memory misconfiguration BEFORE creating any on-disk state.
    _effective_memory_choice = "local" if args.enable_memory and args.memory == "none" else args.memory
    if _effective_memory_choice == "api":
        raise NotImplementedError(
            "--memory=api (remote GigaEvo Memory Module backend) is not supported by "
            "kernel-evo. Only local mode is wired (writer in `kernel-evo memory append`, "
            "reader via stock gigaevo SelectorMemoryProvider). Use --memory=local."
        )
    if _effective_memory_choice == "local" and not args.memory_dir:
        raise SystemExit(
            "--enable-memory / --memory=local requires --memory-dir pointing at a "
            "memory bank. Build one with `kernel-evo memory append`."
        )

    if not (str(getattr(args, "log_dir", "") or "").strip()):
        args.stdout_dir = args.validator_debug_dir = args.llm_log_dir = args.tensorboard_dir = ""

    resources_dir = get_resources_dir()
    repo_root = get_repo_root()
    gigaevo_dir = _get_gigaevo_dir()

    if not (resources_dir / "metrics.yaml").exists():
        raise FileNotFoundError(f"Missing {resources_dir}/metrics.yaml")
    if not (resources_dir / "validate.py").exists():
        raise FileNotFoundError(f"Missing {resources_dir}/validate.py")

    # Load reference architecture source
    if str(args.problem_path).strip():
        problem_file = _resolve_problem_file(str(args.problem_path))
        ref_arch_src = problem_file.read_text(encoding="utf-8")
        problem_kind = "custom"
    else:
        if args.level is None or args.problem_id is None:
            raise SystemExit("Must provide either --problem-path OR both --level and --problem-id.")

        from kernelbench.dataset import construct_kernelbench_dataset  # type: ignore[import-not-found]

        ds = construct_kernelbench_dataset(
            level=args.level,
            source=args.dataset_src,
            dataset_name=args.dataset_name,
        )
        kb_problem = ds.get_problem_by_id(args.problem_id)
        ref_arch_src = kb_problem.code
        problem_kind = "kernelbench"

    from kernel_evo.core.code import python_backend_utils as _py_utils

    split_kernelbench_ref = _py_utils.split_kernelbench_ref
    if str(args.backend).lower() not in _py_utils.PYTHON_BACKENDS:
        raise ValueError(
            "Unsupported backend: "
            f"{args.backend}. Only triton, cuda_inline and cute are supported; "
            "cpp/cuda/torch are not variants."
        )

    codegen_kind = "python"  # only triton, cuda_inline and cute are supported

    model_src, inputs_src = split_kernelbench_ref(ref_arch_src)

    formatted_time = time.strftime("%Y%m%d_%H%M%S")
    if args.problem_path:
        problem_name = f"{args.experiment_name}_{formatted_time}"
    else:
        problem_name = f"kernelbench_{args.level}_{args.problem_id}_{formatted_time}"

    log_dir = str(getattr(args, "log_dir", "") or "").strip()
    run_root = (
        Path(log_dir).expanduser().resolve()
        if log_dir
        else (repo_root / "outputs").resolve()
    )
    experiment_dir = run_root / problem_name
    workspace = prepare_problem_workspace(
        resources_dir=resources_dir,
        workspace_root=experiment_dir / "problem",
    )
    problem_dir = workspace.root_dir
    task_path = workspace.task_description_file

    if log_dir:
        args.stdout_dir = str(experiment_dir)
        args.validator_debug_dir = str(experiment_dir / LOG_SUBDIR_VALIDATE)
        args.llm_log_dir = str(experiment_dir / LOG_SUBDIR_TRACES)
        args.tensorboard_dir = str(experiment_dir / LOG_SUBDIR_TENSORBOARD)
        # Create experiment dir and subdirs (traces, validate_logs, tensorboard) so they exist
        # even if proxy is started later; traces is written when using --llm-log-port.
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / LOG_SUBDIR_TRACES).mkdir(exist_ok=True)
        (experiment_dir / LOG_SUBDIR_VALIDATE).mkdir(exist_ok=True)
        (experiment_dir / LOG_SUBDIR_TENSORBOARD).mkdir(exist_ok=True)
    args.profile_artifacts_dir = str(experiment_dir / LOG_SUBDIR_ARTIFACTS)

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
        "problem_dir": str(problem_dir),
        "backend": str(args.backend),
        "codegen_kind": str(codegen_kind),
        "precision": str(args.precision),
        "runtime_precision": resolve_runtime_precision_string(
            str(args.precision),
            str(getattr(args, "runtime_precision", "") or ""),
        ),
        "timing_method": str(args.timing_method),
        "num_correct_trials": int(args.num_correct_trials),
        "num_perf_trials": int(args.num_perf_trials),
        "output_rtol": (float(args.output_rtol) if args.output_rtol is not None else None),
        "output_atol": (float(args.output_atol) if args.output_atol is not None else None),
        "device": str(args.device),
        "ref_arch_src": ref_arch_src,
        "ref_model_class_src": model_src,
        "ref_inputs_init_src": inputs_src,
        "validator_debug": bool(args.validator_debug),
        "validator_debug_dir": args.validator_debug_dir,
        "validator_debug_max_code_chars": int(args.validator_debug_max_code_chars),
        "stdout_dir": str(getattr(args, "stdout_dir", "") or ""),
        "experiment_dir": str(experiment_dir),
        "execution_mode": str(args.execution_mode),
        "remote_validator_url": str(args.remote_validator_url),
        "remote_poll_interval": float(args.remote_poll_interval),
        "use_memory_for_errors": bool(args.use_memory_for_errors),
        "profile_stage_enabled": bool(args.enable_profiler_stage),
        "profile_runners": [
            item.strip()
            for item in str(getattr(args, "profile_runners", "") or "").split(",")
            if item.strip()
        ],
        "profile_max_insights": int(args.profile_max_insights),
        "profile_torch_warmup_steps": int(args.profile_torch_warmup_steps),
        "profile_torch_active_steps": int(args.profile_torch_active_steps),
        "profile_ncu_path": str(args.profile_ncu_path),
        "profile_ncu_set": str(getattr(args, "profile_ncu_set", "full") or "full").strip() or "full",
        "profile_ncu_kernel_name": str(getattr(args, "profile_ncu_kernel_name", "") or "").strip(),
        "profile_ncu_extra_args": str(getattr(args, "profile_ncu_extra_args", "") or "").strip(),
        "profile_ncu_min_speedup": float(args.profile_ncu_min_speedup),
        "profile_artifacts_dir": str(getattr(args, "profile_artifacts_dir", "") or ""),
        "arch_list": str(getattr(args, "arch_list", "") or "").strip() or None,
    }
    if run_cfg["arch_list"] is None:
        del run_cfg["arch_list"]
    workspace.run_config_file.write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Write the initial prompt shown to the mutation LLM (ALL context included here).
    # cuda_inline and future backends may use a separate prompt directory.
    from kernel_evo.core.code.cute_backend_utils import build_cute_seed, is_cute_backend
    from kernel_evo.core.code.python_backend_utils import model_to_modelnew

    task_text = build_task_description_for_backend(
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
    # For cute, seed with a *compliant working* CuTe program (torch compute + verified CuTe identity
    # passthrough) so evolution starts from working CuTe code instead of authoring it from scratch.
    if is_cute_backend(str(args.backend)):
        seed_program_code = build_cute_seed(model_src)
    else:
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
        llm_log_dir = Path(args.llm_log_dir).expanduser().resolve()
        llm_log_dir.mkdir(parents=True, exist_ok=True)
        # Proxy lives in kernel_evo/core/llm/ (evolve.py is in kernel_evo/core/code/)
        proxy_script = Path(__file__).resolve().parent.parent / "llm" / "openai_proxy_logger.py"
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
            str(llm_log_dir),
        ]
        proxy_proc = subprocess.Popen(
            proxy_cmd,
            cwd=str(experiment_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Give the proxy a moment to bind.
        time.sleep(0.2)
        effective_llm_base_url = f"http://127.0.0.1:{llm_log_port}"

    # Allow KernelGeneration-local Hydra configs (pipeline=kernel_generation_direct, etc.).
    config_dir = (resources_dir / "config").resolve()
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Missing KernelGeneration config directory: {config_dir} "
            "(expected after moving pipeline config out of gigaevo-core-internal)."
        )

    # Hydra searchpath: local overrides (pipeline, prompts, llm) layered on top of
    # the vendored gigaevo defaults in resources/gigaevo/config (used by run.py).
    kg_config = (resources_dir / "gigaevo" / "config").resolve()

    searchpath = [f"file://{kg_config}", f"file://{config_dir}"]

    # Copy default prompts into experiment dir, then overlay backend-specific prompts (e.g. cuda_inline mutation).
    prompts_dir = prepare_prompts_for_experiment(problem_dir, args.backend)
    cmd = [
        sys.executable,
        str(run_py),
        f"experiment={args.experiment}",
        f"hydra.searchpath=[{','.join(searchpath)}]",
        "pipeline=kernel_generation_direct",
        "prompts=kernel",
        f"prompts.dir={prompts_dir}",  # explicit override so subprocess does not rely on env
        "llm=single_recover",
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

    if str(args.tensorboard_dir).strip():
        tensorboard_log_dir = Path(args.tensorboard_dir).expanduser().resolve()
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        cmd.append(f"log_dir={tensorboard_log_dir}")

    if bool(args.disable_insights_lineage):
        cmd.append(
            "pipeline_builder._target_=kernel_evo.core.pipeline.builders.DirectCodeContextPipelineNoInsightsLineageBuilder"
        )

    # gigaevo memory READ stage (1.28+). Evolve never writes memory anymore;
    # build banks separately with `kernel-evo memory append`. --enable-memory
    # is a shorthand for --memory=local and requires --memory-dir to point at
    # an existing bank.
    memory_choice = args.memory
    memory_dir = args.memory_dir
    if args.enable_memory and memory_choice == "none":
        memory_choice = "local"
    # The (memory_choice in {local, api}) ⇒ memory_dir set invariant is enforced
    # at the top of run_evolve so we fail before creating any on-disk state.
    if memory_choice != "none":
        cmd.append(f"memory={memory_choice}")
    if memory_dir:
        # Hydra-side name stays `checkpoint_dir` (gigaevo convention).
        cmd.append(f"checkpoint_dir={Path(memory_dir).expanduser().resolve()}")
    # experiment_dir is exposed for any downstream Hydra config that wants
    # per-run paths; ideas_tracker (post-run write hook) is no longer enabled
    # by kernel-evo evolve, so its logs_dir interpolation is a no-op here.
    cmd.append(f"experiment_dir={experiment_dir.resolve()}")
    if args.namespace:
        cmd.append(f"namespace={args.namespace}")

    # Per-run override of gigaevo's memory_backend.yaml.
    #
    # Why: gigaevo's write_pipeline_config.MEMORY_DIR is resolved at gigaevo
    # import time from `paths.checkpoint_dir` in memory_backend.yaml, with no
    # env-var override. The shipped value is "memory" (relative), so cards land
    # inside site-packages/gigaevo/memory/memory/. To redirect them to the
    # per-run --memory-dir, we materialize a copy of the YAML here with
    # paths.checkpoint_dir set to the resolved absolute path, then point
    # EVO_MEMORY_CONFIG_PATH at it before launching the gigaevo subprocess
    # (run.py uses os.environ.setdefault, so an externally-set value wins).
    #
    # We also force runtime.sync_on_init: false — evolve is read-only on the
    # memory bank, and sync_on_init=true would let the platform backend reset
    # the local file from an empty/stale API server on startup.
    memory_backend_override_path: Path | None = None
    if memory_dir:
        try:
            import yaml  # PyYAML; transitive dep via Hydra
        except ImportError:
            yaml = None
        if yaml is not None:
            mem_abs = Path(memory_dir).expanduser().resolve()
            mem_abs.mkdir(parents=True, exist_ok=True)
            bundled_yaml = gigaevo_dir / "memory_backend.yaml"
            try:
                cfg = yaml.safe_load(bundled_yaml.read_text(encoding="utf-8")) or {}
                cfg.setdefault("paths", {})["checkpoint_dir"] = str(mem_abs)
                cfg.setdefault("runtime", {})["sync_on_init"] = False
                experiment_dir.mkdir(parents=True, exist_ok=True)
                memory_backend_override_path = experiment_dir / "memory_backend.yaml"
                memory_backend_override_path.write_text(
                    yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
                )
            except Exception as exc:
                print(f"Warning: failed to materialize per-run memory_backend.yaml ({exc}); falling back to bundled.")
                memory_backend_override_path = None

    print("Running:")
    print("  " + " ".join(cmd))
    print("")
    print("Notes:")
    print("- Redis must be running (default: localhost:6379).")
    print("- If you get 'Redis database is not empty', either flush it or pass --redis-db <new>.")
    print("")

    try:
        # Ensure kernel_evo is importable when gigaevo run.py runs (Hydra instantiates pipeline builders).
        env = os.environ.copy()
        src_dir = repo_root / "src"
        extra_paths = [str(src_dir)] if src_dir.is_dir() else [str(repo_root)]
        prefix = ":".join(extra_paths)
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = f"{prefix}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = prefix

        # So prompts/kernel.yaml can use ${oc.env:KERNEL_EVO_PROMPTS_DIR}
        env["KERNEL_EVO_PROMPTS_DIR"] = str(prompts_dir)

        # Point gigaevo.memory.runtime_config at the per-run YAML override (see above).
        if memory_backend_override_path is not None:
            env["EVO_MEMORY_CONFIG_PATH"] = str(memory_backend_override_path)
        
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
