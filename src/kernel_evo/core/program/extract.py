#!/usr/bin/env python3
import math
from collections.abc import AsyncIterator, Iterable
from typing import Any

from redis import asyncio as aioredis

from gigaevo.programs.program import Program
from gigaevo.programs.utils import pickle_b64_deserialize
from gigaevo.utils.json import loads as json_loads


def _chunks(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _program_from_redis_blob(raw: str) -> Program | None:
    """Parse Redis blob into Program, bypassing stage_results to avoid heavy deps (cma, etc.)."""
    try:
        data = json_loads(raw)
        d = dict(data)
        if "metadata" in d and isinstance(d["metadata"], str):
            d["metadata"] = pickle_b64_deserialize(d["metadata"])
        # Skip stage_results deserialization - ProgramStageResult.from_dict pulls in cma
        # and other deps that may be missing. Extract only needs id, code, metrics, metadata.
        d["stage_results"] = {}
        return Program.from_dict(d)
    except Exception:
        return None


async def iter_programs(
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
                        program = _program_from_redis_blob(raw)
                        if program is not None:
                            yield program
            if cursor == 0:
                break
    finally:
        await r.aclose()
        await r.connection_pool.disconnect(inuse_connections=True)


def _fitness_value(program: Program) -> float | None:
    f = program.metrics.get("fitness")
    if f is None or (isinstance(f, float) and math.isnan(f)):
        return None
    try:
        return float(f)
    except (TypeError, ValueError):
        return None


async def select_program(
    *,
    redis_url: str,
    redis_prefix: str,
    best: bool = False,
    iteration: int | None = None,
) -> Program | None:
    """Select one program: by best fitness, or by iteration (best fitness among that iteration)."""
    selected: Program | None = None
    selected_fitness: float = float("-inf")
    first_match: Program | None = None

    async for program in iter_programs(redis_url=redis_url, redis_prefix=redis_prefix):
        if best:
            f = _fitness_value(program)
            if f is None:
                continue
            if f > selected_fitness:
                selected = program
                selected_fitness = f
        else:
            if iteration is None:
                continue
            it = program.metadata.get("iteration")
            if it is None:
                continue
            try:
                it_int = int(it)
            except Exception:
                continue
            if it_int != iteration:
                continue

            if first_match is None:
                first_match = program

            f = _fitness_value(program)
            if f is None:
                continue
            if f > selected_fitness:
                selected = program
                selected_fitness = f

    if not best and selected is None:
        selected = first_match
    return selected


def program_to_row(program: Program) -> dict[str, Any]:
    """Flatten a Program into a dict for display or serialization."""
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
    for mname in sorted(program.metrics.keys()):
        row[f"metric_{mname}"] = program.metrics.get(mname)
    row["lineage_num_parents"] = len(program.lineage.parents)
    row["lineage_num_children"] = len(program.lineage.children)
    row["lineage_mutation"] = program.lineage.mutation
    row["lineage_generation"] = program.lineage.generation
    for k in sorted(program.metadata.keys()):
        row[f"metadata_{k}"] = program.metadata.get(k)
    return row
