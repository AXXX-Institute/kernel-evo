"""CLI for `kernel-evo memory append`.

Builds (or extends) a memory bank from a finished evolve run's Redis state by
driving gigaevo's IdeaTracker post-hoc. See core/memory/build.py for the why.

A fresh memory bank is created by running `append` against an empty/non-existent
directory; subsequent runs append to it. Evolve itself never writes memory
anymore — memory is built manually with this command.
"""

import argparse

from kernel_evo.core.memory.build import run_memory_append


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "memory",
        help="Build memory banks from finished evolve runs.",
    )
    action_subs = parser.add_subparsers(dest="memory_action", required=True)

    append_p = action_subs.add_parser(
        "append",
        help=(
            "Append cards from one finished evolve run to a memory bank. "
            "Creates the bank if --memory-dir does not yet contain one."
        ),
    )
    append_p.add_argument(
        "--memory-dir",
        required=True,
        help="Memory bank directory. Created if missing; appended to if present.",
    )
    append_p.add_argument(
        "--redis-prefix",
        required=True,
        help="Redis key prefix of the run to import (= problem.name at evolve "
        "time; appears in evolve's printed gigaevo command as problem.name=...).",
    )
    append_p.add_argument("--redis-host", default="localhost")
    append_p.add_argument("--redis-port", type=int, default=6379)
    append_p.add_argument("--redis-db", type=int, default=0)
    append_p.add_argument(
        "--problem-dir",
        default="",
        help="Optional problem directory (the one passed to evolve). If it has "
        "task_description.txt, its contents are used as the analyzer's task_description.",
    )
    append_p.add_argument(
        "--analyzer-type",
        default="default",
        choices=["default", "fast"],
        help="default = per-pair LLM analyzer; fast = embedding-based clustering.",
    )
    append_p.add_argument("--analyzer-model", default="google/gemini-3-flash-preview")
    append_p.add_argument("--analyzer-base-url", default="https://openrouter.ai/api/v1")
    append_p.add_argument(
        "--namespace",
        default="",
        help="Memory namespace (e.g. an experiment id when sharing a backend).",
    )
    append_p.add_argument(
        "--no-memory-write",
        action="store_true",
        help="Skip the memory write pipeline. Useful for dry-runs that just dump "
        "banks.json/programs.json/best_ideas.json into --session-logs-dir.",
    )
    append_p.add_argument(
        "--no-memory-usage-tracking",
        action="store_true",
        help="Disable building memory_usage_updates from the loaded programs.",
    )
    append_p.add_argument(
        "--session-logs-dir",
        default="",
        help="Where IdeasTrackerLogger writes banks.json/programs.json/best_ideas.json/log.txt "
        "(timestamped subdir is created inside). Defaults to ./outputs/memory_append_<ts>/.",
    )


def memory(args: argparse.Namespace) -> None:
    if args.memory_action != "append":
        raise SystemExit("Use 'kernel-evo memory append'.")
    run_memory_append(args)
