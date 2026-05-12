from __future__ import annotations

import csv
import io
from typing import Any


_LAYOUT_TRANSFORM_MARKERS = (
    "nchwtonhwc",
    "nhwctonchw",
    "transpose",
    "permute",
    "reorder",
)


def summarize_profiler_for_llm(*, profiler_name: str, summary: dict[str, Any]) -> dict[str, Any]:
    if profiler_name != "ncu":
        return summary
    return summarize_ncu_for_llm(summary)


def summarize_ncu_for_llm(summary: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "status",
        "reason",
        "returncode",
        "report_exists",
        "report_file",
        "effective_ncu_devices",
        "effective_target_device",
        "warnings",
    ):
        value = summary.get(key)
        if value not in (None, [], {}):
            compact[key] = value

    host_preflight = _compact_preflight(summary.get("host_preflight"))
    if host_preflight:
        compact["host_preflight"] = host_preflight

    attempts = _compact_attempts(summary.get("attempts"))
    if attempts:
        compact["attempts"] = attempts

    kernel_overview = _compact_raw_csv_preview(summary.get("raw_csv_preview"))
    if kernel_overview:
        compact["kernel_overview"] = kernel_overview
    elif summary.get("raw_csv_file"):
        compact["kernel_overview"] = {
            "raw_csv_file": summary["raw_csv_file"],
            "parse_error": "raw_csv_preview was missing or could not be parsed",
        }

    if "kernel_overview" not in compact:
        stdout_excerpt = _shorten_text(summary.get("stdout_excerpt"))
        stderr_excerpt = _shorten_text(summary.get("stderr_excerpt"))
        if stdout_excerpt:
            compact["stdout_excerpt"] = stdout_excerpt
        if stderr_excerpt:
            compact["stderr_excerpt"] = stderr_excerpt

    return compact


def _compact_preflight(preflight: Any) -> dict[str, Any]:
    if not isinstance(preflight, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("available", "reason", "returncode", "devices", "report_exists"):
        value = preflight.get(key)
        if value not in (None, [], {}):
            compact[key] = value
    stdout_excerpt = _shorten_text(preflight.get("stdout_excerpt"), limit=400)
    stderr_excerpt = _shorten_text(preflight.get("stderr_excerpt"), limit=400)
    if stdout_excerpt:
        compact["stdout_excerpt"] = stdout_excerpt
    if stderr_excerpt:
        compact["stderr_excerpt"] = stderr_excerpt
    return compact


def _compact_attempts(attempts: Any) -> list[dict[str, Any]]:
    if not isinstance(attempts, list):
        return []

    compact_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        compact: dict[str, Any] = {}
        for key in ("label", "returncode", "report_exists", "devices", "section_set", "extra_args"):
            value = attempt.get(key)
            if value not in (None, [], {}):
                compact[key] = value
        if not bool(attempt.get("report_exists")):
            stdout_excerpt = _shorten_text(attempt.get("stdout_excerpt"), limit=500)
            stderr_excerpt = _shorten_text(attempt.get("stderr_excerpt"), limit=500)
            if stdout_excerpt:
                compact["stdout_excerpt"] = stdout_excerpt
            if stderr_excerpt:
                compact["stderr_excerpt"] = stderr_excerpt
        compact_attempts.append(compact)
    return compact_attempts


def _compact_raw_csv_preview(raw_csv_preview: Any) -> dict[str, Any]:
    if not isinstance(raw_csv_preview, list) or len(raw_csv_preview) < 3:
        return {}

    parsed_rows: list[list[str]] = []
    for line in raw_csv_preview:
        if not isinstance(line, str) or not line.strip():
            continue
        parsed_rows.extend(list(csv.reader(io.StringIO(line))))

    if len(parsed_rows) < 3:
        return {}

    headers = parsed_rows[0]
    data_rows = parsed_rows[2:]
    if not headers or not data_rows:
        return {}

    kernels: dict[str, dict[str, Any]] = {}
    layout_transform_occurrences = 0

    for row in data_rows:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        record = {headers[idx]: row[idx] for idx in range(len(headers))}
        kernel_name = _first_nonempty(record, "Kernel Name", "launch__kernel_name")
        if not kernel_name:
            continue

        lowered = kernel_name.lower()
        if any(marker in lowered for marker in _LAYOUT_TRANSFORM_MARKERS):
            layout_transform_occurrences += 1

        entry = kernels.setdefault(
            kernel_name,
            {
                "kernel_name": kernel_name,
                "occurrences": 0,
                "sample_ids": [],
                "example_block_size": _first_nonempty(record, "Block Size", "launch__block_size"),
                "example_grid_size": _first_nonempty(record, "Grid Size", "launch__grid_size"),
                "_registers": [],
                "_shared_mem_kbyte": [],
                "_occupancy_gpu_pct": [],
                "_occupancy_pct": [],
                "_max_warps_pct": [],
                "_replayer_passes": [],
            },
        )

        entry["occurrences"] += 1
        sample_id = _clean_scalar(record.get("ID", ""))
        if sample_id and len(entry["sample_ids"]) < 5:
            entry["sample_ids"].append(sample_id)

        _append_number(entry["_registers"], record.get("launch__registers_per_thread"))
        _append_number(entry["_shared_mem_kbyte"], record.get("launch__shared_mem_per_block_allocated"))
        _append_number(entry["_occupancy_gpu_pct"], record.get("launch__occupancy_cluster_gpu_pct"))
        _append_number(entry["_occupancy_pct"], record.get("launch__occupancy_cluster_pct"))
        _append_number(entry["_max_warps_pct"], record.get("sm__maximum_warps_per_active_cycle_pct"))
        _append_number(entry["_replayer_passes"], record.get("profiler__replayer_passes"))

    if not kernels:
        return {}

    compact_kernels = []
    for entry in sorted(kernels.values(), key=lambda item: (-int(item["occurrences"]), str(item["kernel_name"]))):
        compact_entry: dict[str, Any] = {
            "kernel_name": entry["kernel_name"],
            "occurrences": entry["occurrences"],
            "sample_ids": entry["sample_ids"],
        }
        if entry["example_block_size"]:
            compact_entry["example_block_size"] = entry["example_block_size"]
        if entry["example_grid_size"]:
            compact_entry["example_grid_size"] = entry["example_grid_size"]

        for source_key, target_key in (
            ("_registers", "registers_per_thread"),
            ("_shared_mem_kbyte", "shared_mem_per_block_kbyte"),
            ("_occupancy_gpu_pct", "occupancy_cluster_gpu_pct"),
            ("_occupancy_pct", "occupancy_cluster_pct"),
            ("_max_warps_pct", "max_warps_per_active_cycle_pct"),
            ("_replayer_passes", "replayer_passes"),
        ):
            range_value = _number_range(entry[source_key])
            if range_value is not None:
                compact_entry[target_key] = range_value

        compact_kernels.append(compact_entry)

    return {
        "preview_row_count": len(data_rows),
        "unique_kernel_count": len(compact_kernels),
        "layout_transform_occurrences": layout_transform_occurrences,
        "kernels": compact_kernels[:8],
    }


def _first_nonempty(record: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = _clean_scalar(record.get(key, ""))
        if value:
            return value
    return ""


def _clean_scalar(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _append_number(target: list[float], raw: Any) -> None:
    value = _parse_number(raw)
    if value is not None:
        target.append(value)


def _parse_number(raw: Any) -> float | None:
    text = _clean_scalar(raw)
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _number_range(values: list[float]) -> float | dict[str, float] | None:
    if not values:
        return None
    rounded = [round(value, 3) for value in values]
    min_value = min(rounded)
    max_value = max(rounded)
    if min_value == max_value:
        return min_value
    return {"min": min_value, "max": max_value}


def _shorten_text(value: Any, *, limit: int = 1000) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 12].rstrip() + "\n...[truncated]"
