"""CLI for evolve: GigaEvo-based kernel evolution."""

import argparse

from kernel_generation.core.code.evolve import run_evolve


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
        default="cuda",
        choices=["cuda", "triton", "tilelang", "cute", "thunderkittens"],
    )
    parser.add_argument(
        "--codegen-kind",
        default="auto",
        choices=["auto", "python", "cpp"],
        help="Program template: auto = python for triton/tilelang/cute/thunderkittens, cpp for cuda.",
    )
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--timing-method", default="cuda_event")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--output-rtol", type=float, default=0.5)
    parser.add_argument("--output-atol", type=float, default=0.5)

    # GigaEvo / infra
    parser.add_argument(
        "--problem-dir",
        default="/home/sivtsov/kernel_generation/BACKUP",
        help="Directory with metrics.yaml, validate.py, task_description.txt, initial_programs/.",
    )
    parser.add_argument(
        "--gigaevo-dir",
        default="/home/sivtsov/gigaevo-core-internal",
        help="Path to gigaevo-core-internal (run.py, config/).",
    )
    parser.add_argument("--experiment", default="base")
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-resume", action="store_true")
    parser.add_argument("--validator-debug", action="store_true")
    parser.add_argument("--validator-debug-dir", default="")
    parser.add_argument("--validator-debug-max-code-chars", type=int, default=50000)
    parser.add_argument("--stdout-dir", default="")

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
    parser.add_argument("--llm-log-dir", default="")
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
