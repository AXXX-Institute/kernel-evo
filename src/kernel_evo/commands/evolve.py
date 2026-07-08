"""CLI for evolve: GigaEvo-based kernel evolution."""

import argparse

from kernel_evo.core.code.evolve import run_evolve
from kernel_evo.core.precision import VALID_PRECISIONS, VALID_RUNTIME_PRECISIONS


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evolve",
        help="Run GigaEvo evolution: write task_description + seed, then run gigaevo run.py.",
    )
    # Problem selection
    parser.add_argument("--dataset-src", default="huggingface", choices=["huggingface", "local"])
    parser.add_argument("--dataset-name", default="ScalingIntelligence/KernelBench")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--problem-id", type=int, default=None)
    parser.add_argument(
        "--problem-path",
        default="",
        help="Path to a custom problem file or directory containing task.py.",
    )
    parser.add_argument("--experiment-name", required=True, help="Name for saving experiment results.")

    # KernelBench evaluation
    parser.add_argument(
        "--backend",
        default="triton",
        choices=["triton", "cuda_inline", "cute"],
        help="Supported backends only; cpp/cuda/torch are not variants.",
    )
    parser.add_argument(
        "--codegen-kind",
        default="auto",
        choices=["auto", "python"],
        help="Program template: auto = python (only triton, cuda_inline and cute are supported).",
    )
    parser.add_argument("--precision", default="fp32", choices=list(VALID_PRECISIONS))
    parser.add_argument(
        "--runtime-precision",
        default="",
        choices=["", *VALID_RUNTIME_PRECISIONS],
        help=(
            "Tensor dtype used by validation/profiling for inputs, model params, and output comparison. "
            "Defaults to bf16 for --precision fp8, otherwise matches --precision."
        ),
    )
    parser.add_argument("--timing-method", default="cuda_event")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--output-rtol", type=float, default=0.01)
    parser.add_argument("--output-atol", type=float, default=0.01)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Cuda device for validation. Written to run_config for validator.",
    )
    parser.add_argument(
        "--arch-list",
        default="",
        help="Optional TORCH_CUDA_ARCH_LIST for cuda_inline backend (e.g. '8.0' or '7.0;8.0'). Written to run_config.",
    )

    # GigaEvo / infra
    parser.add_argument("--experiment", default="base")
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-resume", action="store_true")
    parser.add_argument("--validator-debug", action="store_true")
    parser.add_argument("--validator-debug-max-code-chars", type=int, default=50000)
    parser.add_argument(
        "--log-dir",
        default="",
        help=(
            "Base dir for run outputs. Each run gets <log-dir>/<problem_name>/ with "
            "the workspace, log file, tensorboard data, traces, and validate_logs."
        ),
    )

    # Execution
    parser.add_argument(
        "--execution-mode",
        default="local_execution",
        choices=["local_execution", "remote_execution"],
    )
    parser.add_argument("--remote-validator-url", default="http://localhost:15000")
    parser.add_argument("--remote-poll-interval", type=float, default=1.0)

    # LLM
    parser.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--llm-log-port", type=int, default=None)
    parser.add_argument("--disable-insights-lineage", action="store_true")
    parser.add_argument("--enable-profiler-stage", action="store_true")
    parser.add_argument(
        "--profile-runners",
        default="torch",
        help="Comma-separated profiler runners to enable when profiler stage is on. Supported: torch,ncu",
    )
    parser.add_argument("--profile-max-insights", type=int, default=4)
    parser.add_argument("--profile-torch-warmup-steps", type=int, default=2)
    parser.add_argument("--profile-torch-active-steps", type=int, default=3)
    parser.add_argument(
        "--profile-ncu-path",
        default="ncu",
        help="Nsight Compute executable path or name. Resolved from PATH by default.",
    )
    parser.add_argument(
        "--profile-ncu-set",
        default="full",
        help="Nsight Compute section set to collect.",
    )
    parser.add_argument(
        "--profile-ncu-kernel-name",
        default="",
        help="Optional Nsight Compute --kernel-name filter.",
    )
    parser.add_argument(
        "--profile-ncu-extra-args",
        default="",
        help="Extra raw arguments appended to the ncu command line.",
    )
    parser.add_argument(
        "--profile-ncu-min-speedup",
        type=float,
        default=1.0,
        help="Only run NCU when measured speedup is at least this threshold.",
    )

    # Evolution
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument("--max-elites-per-generation", type=int, default=1)
    parser.add_argument("--max-mutations-per-generation", type=int, default=1)
    parser.add_argument("--num-parents", type=int, default=1)
    parser.add_argument("--use-memory-for-errors", action="store_true")

    # gigaevo memory READ stage (1.28+). Evolve only consumes memory; it never
    # writes back. Build memory banks separately with `kernel-evo memory append`.
    # --memory controls MemoryContextStage's MemoryProvider:
    #   none  → NullMemoryProvider (no cards injected; default)
    #   local → SelectorMemoryProvider reading from --memory-dir
    #   api   → SelectorMemoryProvider against gigaevo-memory backend
    parser.add_argument(
        "--enable-memory",
        action="store_true",
        help=(
            "Read-only: feed cards from --memory-dir into the mutation context. "
            "Shorthand for --memory=local. Requires --memory-dir to point at a "
            "memory bank built with `kernel-evo memory append`."
        ),
    )
    parser.add_argument(
        "--memory",
        default="none",
        choices=["none", "local", "api"],
        help="MemoryProvider used by MemoryContextStage (gigaevo 1.28+).",
    )
    parser.add_argument(
        "--memory-dir",
        default="",
        help=(
            "Existing memory bank directory to read cards from. Required when "
            "--enable-memory or --memory=local is set. Use `kernel-evo memory "
            "append` to build / extend a bank."
        ),
    )
    parser.add_argument(
        "--namespace",
        default="",
        help="Memory namespace for reads (e.g. experiment id when sharing a backend).",
    )


def evolve(args: argparse.Namespace) -> None:
    run_evolve(args)
