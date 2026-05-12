from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.profiler import ProfilerActivity, profile, schedule

from kernel_evo.core.eval.eval import (
    _process_input_tensor,
    get_torch_dtype_from_string,
    graceful_eval_cleanup,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
    set_seed,
)
from kernel_evo.core.precision import resolve_runtime_precision_string


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _first_available(event: Any, *names: str) -> float:
    for name in names:
        if hasattr(event, name):
            value = getattr(event, name)
            if value is not None:
                return _as_float(value)
    return 0.0


def _event_to_dict(event: Any) -> dict[str, Any]:
    input_shapes = getattr(event, "input_shapes", None)
    if input_shapes is None:
        input_shapes = []
    self_device_time_total_us = _first_available(
        event,
        "self_device_time_total",
        "self_cuda_time_total",
        "self_privateuse1_time_total",
    )
    device_time_total_us = _first_available(
        event,
        "device_time_total",
        "cuda_time_total",
        "privateuse1_time_total",
    )
    return {
        "key": str(getattr(event, "key", "")),
        "count": int(getattr(event, "count", 0) or 0),
        "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0) or 0.0),
        "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0) or 0.0),
        "self_cuda_time_total_us": self_device_time_total_us,
        "cuda_time_total_us": device_time_total_us,
        "self_device_memory_usage_bytes": int(getattr(event, "self_device_memory_usage", 0) or 0),
        "device_memory_usage_bytes": int(getattr(event, "device_memory_usage", 0) or 0),
        "input_shapes": input_shapes,
    }


def _build_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    total_self_cuda = sum(item["self_cuda_time_total_us"] for item in events)
    total_cuda = sum(item["cuda_time_total_us"] for item in events)
    top_ops: list[dict[str, Any]] = []
    heuristics: list[dict[str, str]] = []

    for item in events[:10]:
        share = (item["self_cuda_time_total_us"] / total_self_cuda) if total_self_cuda > 0 else 0.0
        top_ops.append(
            {
                "name": item["key"],
                "count": item["count"],
                "self_cuda_time_total_us": item["self_cuda_time_total_us"],
                "cuda_time_total_us": item["cuda_time_total_us"],
                "share_of_self_cuda_time": share,
                "self_device_memory_usage_bytes": item["self_device_memory_usage_bytes"],
            }
        )

    if top_ops:
        hottest = top_ops[0]
        if hottest["share_of_self_cuda_time"] >= 0.6:
            heuristics.append(
                {
                    "kind": "single_hotspot",
                    "detail": f"{hottest['name']} dominates self CUDA time ({hottest['share_of_self_cuda_time']:.1%}).",
                }
            )

    copies = [item for item in top_ops if "copy" in item["name"].lower() or "to" in item["name"].lower()]
    if copies:
        heuristics.append(
            {
                "kind": "data_movement",
                "detail": "Profiler saw copy/to-style operators in the hot path.",
            }
        )

    return {
        "status": "completed",
        "total_self_cuda_time_us": total_self_cuda,
        "total_cuda_time_us": total_cuda,
        "top_ops": top_ops,
        "heuristics": heuristics,
    }


def run_torch_profile(
    *,
    run_config: dict[str, Any],
    ref_arch_src: str,
    custom_model_src: str,
    out_dir: Path,
    target_only: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    out_dir.mkdir(parents=True, exist_ok=True)

    backend = str(run_config.get("backend", "triton"))
    device = torch.device(str(run_config.get("device", "cuda:0")))
    runtime_precision = resolve_runtime_precision_string(
        str(run_config.get("precision", "fp32")),
        str(run_config.get("runtime_precision", "") or ""),
    )
    precision = get_torch_dtype_from_string(runtime_precision)
    warmup_steps = max(1, int(run_config.get("profile_torch_warmup_steps", 2)))
    active_steps = max(1, int(run_config.get("profile_torch_active_steps", 3)))
    seed_num = 42

    context: dict[str, Any] = {}
    temp_file = None
    try:
        torch.cuda.set_device(device)
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_arch_src, context)
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [_process_input_tensor(x, device, backend, precision) for x in init_inputs]

        if backend.lower() == "cuda_inline":
            from kernel_evo.core.code.cuda_backend_utils import apply_cuda_build_env

            apply_cuda_build_env(run_config)
        ModelNew, temp_file = load_custom_model_with_tempfile(custom_model_src, entry_point="ModelNew")
        with torch.no_grad():
            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs).to(device=device, dtype=precision)
        torch.cuda.synchronize(device=device)

        if target_only:
            steps = warmup_steps + active_steps
            for step in range(steps):
                set_seed(seed_num + step)
                inputs = get_inputs()
                inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
                with torch.no_grad():
                    custom_model(*inputs)
                torch.cuda.synchronize(device=device)
            summary = {"status": "completed", "mode": "target_only", "steps": steps}
            (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        prof_schedule = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1)

        with profile(
            activities=activities,
            schedule=prof_schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for step in range(warmup_steps + active_steps):
                set_seed(seed_num + step)
                inputs = get_inputs()
                inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
                with torch.no_grad():
                    custom_model(*inputs)
                torch.cuda.synchronize(device=device)
                prof.step()

        trace_path = out_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))

        events = sorted(
            (_event_to_dict(event) for event in prof.key_averages(group_by_input_shape=True)),
            key=lambda item: (item["self_cuda_time_total_us"], item["cuda_time_total_us"], item["cpu_time_total_us"]),
            reverse=True,
        )
        key_averages_path = out_dir / "key_averages.json"
        key_averages_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = _build_summary(events)
        summary.update(
            {
                "trace_file": str(trace_path),
                "key_averages_file": str(key_averages_path),
                "warmup_steps": warmup_steps,
                "active_steps": active_steps,
            }
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    except torch.cuda.OutOfMemoryError as exc:
        return _write_oom_summary(out_dir, exc, device)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return _write_oom_summary(out_dir, exc, device)
    finally:
        graceful_eval_cleanup(context, device, temp_file)


def _write_oom_summary(out_dir: Path, exc: BaseException, device: torch.device) -> dict[str, Any]:
    logger.warning(f"[torch_runner] GPU out of memory during profiling on {device}: {exc}")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    summary = {
        "status": "failed",
        "reason": "GPU out of memory",
        "error": str(exc),
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
