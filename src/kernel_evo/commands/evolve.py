"""CLI for evolve: GigaEvo-based kernel evolution."""

import argparse

from kernel_evo.core.code.evolve import run_evolve


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
        choices=["triton", "cuda_inline"],
        help="Supported backends only; cpp/cuda/torch are not variants.",
    )
    parser.add_argument(
        "--codegen-kind",
        default="auto",
        choices=["auto", "python"],
        help="Program template: auto = python (only triton and cuda_inline are supported).",
    )
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
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
        help="Base dir for all logs; each run gets <log-dir>/<problem_name>/ with log file, tensorboard, traces, validate_logs.",
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

    # Evolution
    parser.add_argument("--max-generations", type=int, default=1)
    parser.add_argument("--max-elites-per-generation", type=int, default=1)
    parser.add_argument("--max-mutations-per-generation", type=int, default=1)
    parser.add_argument("--num-parents", type=int, default=1)
    parser.add_argument("--use-memory-for-errors", action="store_true")


def evolve(args: argparse.Namespace) -> None:
    run_evolve(args)
