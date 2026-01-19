#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from redis import asyncio as aioredis  # pyright: ignore[reportMissingImports]

from gigaevo.programs.program import Program
from gigaevo.utils.json import dumps as json_dumps
from gigaevo.utils.json import loads as json_loads


DEFAULT_STDOUT_COLS: list[str] = [
    "program_id",
    "metric_is_valid",
    "metric_speedup",
    "metadata_iteration",
]


def _chunks(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


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
    # Keep it readable in markdown tables.
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_kv_table(row: dict[str, Any]) -> str:
    """
    Render a 2-col markdown table with padded first column so it aligns nicely
    even in plain-text viewing.
    """
    items = [(str(k), _to_scalar_str(v)) for k, v in row.items()]
    key_width = max([len("column")] + [len(k) for k, _ in items])
    key_width = max(3, key_width)

    lines: list[str] = []
    lines.append(f"| {'column':<{key_width}} | value |")
    lines.append(f"| {'-' * key_width} | --- |")
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


def _program_to_row(program: Program) -> dict[str, Any]:
    row: dict[str, Any] = {
        "program_id": program.id,
        "name": program.name or "unnamed",
        "created_at": program.created_at.isoformat(),
        "atomic_counter": program.atomic_counter,
        "state": program.state.value,
        "is_complete": program.is_complete,
        "generation": program.generation,
        "is_root": program.is_root,
        "parent_ids": program.lineage.parents,
        "children_ids": program.lineage.children,
    }

    # metrics
    for mname in sorted(program.metrics.keys()):
        row[f"metric_{mname}"] = program.metrics.get(mname)

    # lineage
    row["lineage_num_parents"] = len(program.lineage.parents)
    row["lineage_num_children"] = len(program.lineage.children)
    row["lineage_mutation"] = program.lineage.mutation
    row["lineage_generation"] = program.lineage.generation

    # metadata
    for k in sorted(program.metadata.keys()):
        row[f"metadata_{k}"] = program.metadata.get(k)

    return row


async def _iter_programs(
    *,
    redis_url: str,
    redis_prefix: str,
    scan_count: int = 1000,
    mget_chunk: int = 1024,
) -> AsyncIterator[Program]:
    r = aioredis.from_url(
        redis_url,
        decode_responses=True,
        max_connections=50,
        health_check_interval=60,
        socket_connect_timeout=30.0,
        socket_timeout=30.0,
        retry_on_timeout=True,
    )
    try:
        await r.ping()
        pattern = f"{redis_prefix}:program:*"
        cursor = 0
        while True:
            cursor, keys = await r.scan(cursor=cursor, match=pattern, count=scan_count)
            if keys:
                for key_chunk in _chunks(keys, mget_chunk):
                    blobs = await r.mget(*key_chunk)
                    for raw in blobs:
                        if not raw:
                            continue
                        try:
                            data = json_loads(raw)
                            program = Program.from_dict(data)
                        except Exception:
                            continue
                        yield program
            if cursor == 0:
                break
    finally:
        await r.aclose()
        await r.connection_pool.disconnect(inuse_connections=True)


def _parse_stdout_cols(arg: str | None) -> list[str]:
    if not arg:
        return list(DEFAULT_STDOUT_COLS)
    cols = [c.strip() for c in arg.split(",")]
    return [c for c in cols if c]


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one program's code from Redis by iteration or by best metric_fitness"
    )
    parser.add_argument("--redis-host", default="127.0.0.1", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, required=True, help="Redis database")
    parser.add_argument("--redis-prefix", type=str, required=True, help="Redis prefix")

    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument(
        "--iteration",
        type=int,
        help="Select a program where metadata.iteration == ITERATION (if multiple, picks max metric_fitness when available)",
    )
    sel.add_argument(
        "--best",
        action="store_true",
        help="Select the single program with the largest metric_fitness",
    )

    parser.add_argument(
        "--add_info",
        action="store_true",
        help="Prefix output file with a Python triple-quoted markdown table of the selected row (excluding code)",
    )
    parser.add_argument(
        "--stdout-cols",
        type=str,
        default=",".join(DEFAULT_STDOUT_COLS),
        help="Comma-separated list of columns to print to stdout as a small markdown table",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write the extracted program code into",
    )
    args = parser.parse_args()

    redis_url = f"redis://{args.redis_host}:{args.redis_port}/{args.redis_db}"
    redis_prefix: str = args.redis_prefix

    stdout_cols = _parse_stdout_cols(args.stdout_cols)

    selected: Program | None = None
    selected_fitness: float = float("-inf")
    first_match: Program | None = None

    async for program in _iter_programs(redis_url=redis_url, redis_prefix=redis_prefix):
        if args.best:
            f = program.metrics.get("fitness")
            if f is None or (isinstance(f, float) and math.isnan(f)):
                continue
            if f > selected_fitness:
                selected = program
                selected_fitness = float(f)
        else:
            it = program.metadata.get("iteration")
            if it is None:
                continue
            try:
                it_int = int(it)
            except Exception:
                continue
            if it_int != int(args.iteration):
                continue

            if first_match is None:
                first_match = program

            f = program.metrics.get("fitness")
            if f is None or (isinstance(f, float) and math.isnan(f)):
                continue
            if f > selected_fitness:
                selected = program
                selected_fitness = float(f)

    if args.best:
        if selected is None:
            print(
                "ERROR: no programs found with numeric metric_fitness in Redis",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        if selected is None:
            selected = first_match
        if selected is None:
            print(
                f"ERROR: no programs found with metadata.iteration == {args.iteration}",
                file=sys.stderr,
            )
            sys.exit(1)

    row = _program_to_row(selected)

    # stdout info (markdown table)
    out_values = [_to_scalar_str(row.get(c)) for c in stdout_cols]
    sys.stdout.write(_format_markdown_stdout_table(stdout_cols, out_values) + "\n")
    sys.stdout.flush()

    # file output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ""
    if args.add_info:
        info_row = dict(row)
        info_row.pop("code", None)
        md = _format_markdown_kv_table(info_row)
        header = '"""\n' + md + '\n"""\n\n'

    output_path.write_text(header + selected.code, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())

