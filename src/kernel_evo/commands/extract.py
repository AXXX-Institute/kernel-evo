import argparse
import asyncio
import math
import sys
from pathlib import Path
from typing import Any

from gigaevo.utils.json import dumps as json_dumps

from kernel_evo.core.program.extract import program_to_row, select_program


DEFAULT_STDOUT_COLS = ["program_id", "metric_is_valid", "metric_speedup", "metadata_iteration"]


def _to_scalar_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return repr(v)
    if isinstance(v, (dict, list, tuple)):
        try:
            return json_dumps(v)
        except Exception:
            return repr(v)
    return str(v)


def _md_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_kv_table(row: dict[str, Any]) -> str:
    items = [(str(k), _to_scalar_str(v)) for k, v in row.items()]
    key_width = max([len("column")] + [len(k) for k, _ in items], 3)
    lines = [
        f"| {'column':<{key_width}} | value |",
        f"| {'-' * key_width} | --- |",
    ]
    for k, v in items:
        lines.append(f"| {_md_escape(k):<{key_width}} | {_md_escape(v)} |")
    return "\n".join(lines)


def _format_markdown_stdout_table(cols: list[str], values: list[str]) -> str:
    headers = [_md_escape(c) for c in cols]
    vals = [_md_escape(v) for v in values]
    widths = [max(len(h), len(v), 3) for h, v in zip(headers, vals)]
    header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    sep_row = "| " + " | ".join("-" * w for w in widths) + " |"
    value_row = "| " + " | ".join(f"{v:<{w}}" for v, w in zip(vals, widths)) + " |"
    return "\n".join([header_row, sep_row, value_row])


def _parse_stdout_cols(arg: str | None) -> list[str]:
    if not arg:
        return list(DEFAULT_STDOUT_COLS)
    return [c.strip() for c in arg.split(",") if c.strip()]


def setup_parser(subparsers: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "extract",
        description="Extract one program's code from Redis by iteration or by best metric_fitness",
    )
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, required=True, help="Redis database")
    parser.add_argument("--redis-prefix", type=str, required=True, help="Redis prefix")

    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument(
        "--iteration",
        type=int,
        help="Select program where metadata.iteration == ITERATION (best fitness if multiple)",
    )
    sel.add_argument("--best", action="store_true", help="Select program with largest metric_fitness")

    parser.add_argument(
        "--add_info",
        action="store_true",
        help="Prefix output file with markdown table of the selected row (excluding code)",
    )
    parser.add_argument(
        "--stdout-cols",
        type=str,
        default=",".join(DEFAULT_STDOUT_COLS),
        help="Comma-separated columns for stdout markdown table",
    )
    parser.add_argument("--output-file", type=str, required=True, help="Path to write extracted code")


def extract(args: argparse.Namespace) -> None:
    redis_url = f"redis://{args.redis_host}:{args.redis_port}/{args.redis_db}"
    stdout_cols = _parse_stdout_cols(getattr(args, "stdout_cols", None))

    selected = asyncio.run(
        select_program(
            redis_url=redis_url,
            redis_prefix=args.redis_prefix,
            best=args.best,
            iteration=getattr(args, "iteration", None),
        )
    )

    if selected is None:
        if args.best:
            print("ERROR: no programs found with numeric metric_fitness in Redis", file=sys.stderr)
        else:
            print(
                f"ERROR: no programs found with metadata.iteration == {args.iteration}",
                file=sys.stderr,
            )
        sys.exit(1)

    row = program_to_row(selected)
    out_values = [_to_scalar_str(row.get(c)) for c in stdout_cols]
    sys.stdout.write(_format_markdown_stdout_table(stdout_cols, out_values) + "\n")
    sys.stdout.flush()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ""
    if args.add_info:
        info_row = {k: v for k, v in row.items() if k != "code"}
        header = '"""\n' + _format_markdown_kv_table(info_row) + '\n"""\n\n'

    output_path.write_text(header + selected.code, encoding="utf-8")
