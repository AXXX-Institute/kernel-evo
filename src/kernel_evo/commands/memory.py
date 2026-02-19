import argparse

from kernel_evo.core.memory.best_programs_memory import (
    create_best_programs_memory,
)
from kernel_evo.core.memory.errors_memory import (
    create_errors_memory,
)


def setup_parser(subparsers: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "memory",
        help="Kernel memory routines (create from validation logs).",
    )
    sub = parser.add_subparsers(dest="memory_subcommand", required=True)
    create_parser = sub.add_parser(
        "create",
        description="Create memory from validation log directories.",
    )
    create_parser.add_argument(
        "--mode",
        choices=["best_programs", "errors"],
        required=True,
        help="best_programs: top-N correct fastest per problem; errors: invalid->valid fix pairs",
    )
    create_parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="Directories containing validate_logs (experiment subdirs with .log files)",
    )
    create_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for memory checkpoint (AmemGamMemory)",
    )
    create_parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help="Max best programs per problem (default: 4, best_programs only)",
    )
    create_parser.add_argument(
        "--rebuild-interval",
        type=int,
        default=1000,
        help="AmemGamMemory rebuild_interval (default: 1000)",
    )


def memory(args: argparse.Namespace) -> None:
    if getattr(args, "memory_subcommand", None) != "create":
        return
    out = args.output_dir
    source_dirs = args.source_dirs
    rebuild = args.rebuild_interval
    if args.mode == "best_programs":
        n = create_best_programs_memory(
            source_dirs,
            out,
            top_n=args.top_n,
            rebuild_interval=rebuild,
        )
        print(f"Saved {n} best-program entries to {out}")
    else:
        n = create_errors_memory(
            source_dirs,
            out,
            rebuild_interval=rebuild,
        )
        print(f"Saved {n} error-fix entries to {out}")
