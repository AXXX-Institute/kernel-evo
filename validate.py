from __future__ import annotations

import hashlib
import io
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
import sys
from pathlib import Path
from typing import Any


RUNTIME_SENTINEL_US = 1_000_000_000.0


def _find_problem_dir() -> Path:
    """
    exec_runner executes code in a synthetic module without setting __file__.
    So we locate the problem directory via sys.path (exec_runner prepends python_path).
    """
    candidates: list[Path] = []
    for p in sys.path:
        try:
            pp = Path(p)
        except Exception:
            continue
        if not pp.exists() or not pp.is_dir():
            continue
        if (pp / "metrics.yaml").exists():
            candidates.append(pp)

    if candidates:
        candidates.sort(key=lambda x: len(x.parts))
        return candidates[0]

    return Path("/home/sivtsov/kernel_generation")


def _add_kernelbench_to_sys_path(problem_dir: Path) -> None:
    # KernelBench lives under /home/sivtsov/kernel_generation/KernelBench in this workspace
    kb_src = problem_dir / "KernelBench" / "src"
    if kb_src.exists():
        s = str(kb_src)
        if s not in sys.path:
            sys.path.insert(0, s)


def _extract_custom_model_src(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("custom_model_src", "custom_kernel", "code", "src", "model_src"):
            v = payload.get(key)
            if isinstance(v, str):
                return v

    raise TypeError(
        "Validator expected payload to be a str (custom_model_src) or a dict containing it; "
        f"got {type(payload)}"
    )


def _extract_program_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        v = payload.get("program_id")
        if isinstance(v, str) and v:
            return v
    return None


def _debug_dir(problem_dir: Path, cfg: dict[str, Any]) -> Path:
    v = cfg.get("validator_debug_dir")
    if isinstance(v, str) and v.strip():
        return Path(v).expanduser().resolve()
    return (problem_dir / "validator_debug").resolve()


def _safe_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_safe_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe_jsonable(v) for k, v in x.items()}
    return repr(x)


def _write_debug_log(
    *,
    problem_dir: Path,
    cfg: dict[str, Any],
    payload: Any,
    custom_model_src: str,
    result: Any | None,
    captured: str,
    exc: BaseException | None,
) -> None:
    if not bool(cfg.get("validator_debug", False)):
        return

    out_dir = _debug_dir(problem_dir, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    program_id = _extract_program_id(payload) or "unknown_program"
    code_hash = hashlib.sha1(custom_model_src.encode("utf-8", errors="ignore")).hexdigest()[:10]
    out_path = out_dir / f"{program_id}_{code_hash}.log"

    # Avoid gigantic files by default.
    max_code_chars = int(cfg.get("validator_debug_max_code_chars", 50_000))
    code_to_write = custom_model_src if len(custom_model_src) <= max_code_chars else (
        custom_model_src[:max_code_chars] + "\n\n# [truncated]\n"
    )

    lines: list[str] = []
    lines.append("KERNEL_GENERATION VALIDATOR DEBUG LOG")
    lines.append("")
    lines.append(f"program_id: {program_id}")
    lines.append(f"code_sha1_10: {code_hash}")
    lines.append("")
    lines.append("=== context (run_config.json / context.py) ===")
    try:
        lines.append(json.dumps(_safe_jsonable(cfg), indent=2, sort_keys=True, ensure_ascii=False))
    except Exception:
        lines.append(repr(cfg))
    lines.append("")

    if result is not None:
        lines.append("=== kernelbench result ===")
        try:
            # pydantic v2
            if hasattr(result, "model_dump"):
                rd = result.model_dump()
            elif hasattr(result, "dict"):
                rd = result.dict()
            else:
                rd = result
            lines.append(json.dumps(_safe_jsonable(rd), indent=2, sort_keys=True, ensure_ascii=False))
        except Exception:
            lines.append(repr(result))
        lines.append("")

    if exc is not None:
        lines.append("=== exception ===")
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        lines.append("")

    if captured.strip():
        lines.append("=== captured stdout/stderr (kernelbench + validator) ===")
        lines.append(captured.rstrip())
        lines.append("")

    lines.append("=== program code ===")
    lines.append(code_to_write.rstrip())
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def validate(*args: Any) -> dict[str, float]:
    """
    GigaEvo validator.

    Supported call signatures (depending on whether problem has context.py):
      - validate(payload)
      - validate(context, payload)
    """
    problem_dir = _find_problem_dir()
    _add_kernelbench_to_sys_path(problem_dir)

    # Unpack args (CallValidatorFunction passes [context, payload] if context exists)
    if len(args) == 1:
        context = None
        payload = args[0]
    elif len(args) == 2:
        context = args[0]
        payload = args[1]
    else:
        raise TypeError(f"validate() expected 1 or 2 args, got {len(args)}")

    try:
        custom_model_src = _extract_custom_model_src(payload)
    except Exception:
        return {
            "speedup": 0.0,
            "runtime_us": RUNTIME_SENTINEL_US,
            "ref_runtime_us": RUNTIME_SENTINEL_US,
            "compiled": 0.0,
            "correctness": 0.0,
            "is_valid": 0.0,
        }

    if not custom_model_src.strip():
        return {
            "speedup": 0.0,
            "runtime_us": RUNTIME_SENTINEL_US,
            "ref_runtime_us": RUNTIME_SENTINEL_US,
            "compiled": 0.0,
            "correctness": 0.0,
            "is_valid": 0.0,
        }

    # Runtime config: expected to come from `context.py` (run_config.json).
    # This keeps evolving programs free of env reads and os/sys imports.
    cfg: dict[str, Any] = context if isinstance(context, dict) else {}
    backend = str(cfg.get("backend", "cuda"))
    codegen_kind = str(cfg.get("codegen_kind", "python")).lower()
    precision_str = str(cfg.get("precision", "fp32"))
    timing_method = str(cfg.get("timing_method", "cuda_event"))
    num_correct_trials = int(cfg.get("num_correct_trials", 5))
    num_perf_trials = int(cfg.get("num_perf_trials", 100))

    # Reference model source code should be provided in context (preferred).
    ref_arch_src: str | None = None
    v = cfg.get("ref_arch_src") or cfg.get("original_model_src")
    if isinstance(v, str):
        ref_arch_src = v

    try:
        if codegen_kind == "cpp":
            # Placeholder path: treat program payload as "kernel generator" python code.
            # We execute it to locate `generate_kernel(...)`, but compilation/bindings are not wired yet.
            try:
                ns: dict[str, Any] = {}
                exec(custom_model_src, ns)  # noqa: S102 (validator-owned execution)
                gen = ns.get("generate_kernel")
                if callable(gen):
                    _ = gen(cfg)
                from scripts.cpp_backend_utils import raise_bindings_not_implemented

                raise_bindings_not_implemented(backend=backend)
            except NotImplementedError:
                _write_debug_log(
                    problem_dir=problem_dir,
                    cfg=cfg,
                    payload=payload,
                    custom_model_src=custom_model_src,
                    result=None,
                    captured="",
                    exc=NotImplementedError("C++ backend path not implemented"),
                )
                return {
                    "speedup": 0.0,
                    "runtime_us": RUNTIME_SENTINEL_US,
                    "ref_runtime_us": RUNTIME_SENTINEL_US,
                    "compiled": 0.0,
                    "correctness": 0.0,
                    "is_valid": 0.0,
                }

        import torch

        from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

        if not torch.cuda.is_available():
            return {
                "speedup": 0.0,
                "runtime_us": RUNTIME_SENTINEL_US,
                "ref_runtime_us": RUNTIME_SENTINEL_US,
                "compiled": 0.0,
                "correctness": 0.0,
                "is_valid": 0.0,
            }

        if ref_arch_src is None:
            # Without ref code we cannot evaluate correctness/perf.
            return {
                "speedup": 0.0,
                "runtime_us": RUNTIME_SENTINEL_US,
                "ref_runtime_us": RUNTIME_SENTINEL_US,
                "compiled": 0.0,
                "correctness": 0.0,
                "is_valid": 0.0,
            }

        captured_buf = io.StringIO()
        result = None
        exc: BaseException | None = None
        try:
            with redirect_stdout(captured_buf), redirect_stderr(captured_buf):
                result = eval_kernel_against_ref(
                    ref_arch_src,
                    custom_model_src,
                    verbose=bool(cfg.get("validator_debug", False)),
                    measure_performance=True,
                    timing_method=timing_method,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                    backend=backend,
                    precision=get_torch_dtype_from_string(precision_str),
                )
        except BaseException as e:  # keep validator resilient
            exc = e
        finally:
            captured = captured_buf.getvalue()

        if exc is not None:
            _write_debug_log(
                problem_dir=problem_dir,
                cfg=cfg,
                payload=payload,
                custom_model_src=custom_model_src,
                result=result,
                captured=captured,
                exc=exc,
            )
            return {
                "speedup": 0.0,
                "runtime_us": RUNTIME_SENTINEL_US,
                "ref_runtime_us": RUNTIME_SENTINEL_US,
                "compiled": 0.0,
                "correctness": 0.0,
                "is_valid": 0.0,
            }

        compiled = 1.0 if bool(result.compiled) else 0.0
        correctness = 1.0 if bool(result.correctness) else 0.0

        runtime_us = (
            float(result.runtime) if (result.runtime is not None and float(result.runtime) > 0) else -1.0
        )
        ref_runtime_us = (
            float(result.ref_runtime)
            if (result.ref_runtime is not None and float(result.ref_runtime) > 0)
            else -1.0
        )

        if compiled and correctness and runtime_us > 0 and ref_runtime_us > 0:
            speedup = ref_runtime_us / runtime_us
            is_valid = 1.0
        else:
            speedup = 0.0
            is_valid = 0.0

        _write_debug_log(
            problem_dir=problem_dir,
            cfg=cfg,
            payload=payload,
            custom_model_src=custom_model_src,
            result=result,
            captured=captured,
            exc=None,
        )

        return {
            "speedup": float(speedup),
            "runtime_us": float(runtime_us if runtime_us > 0 else RUNTIME_SENTINEL_US),
            "ref_runtime_us": float(ref_runtime_us if ref_runtime_us > 0 else RUNTIME_SENTINEL_US),
            "compiled": float(compiled),
            "correctness": float(correctness),
            "is_valid": float(is_valid),
        }
    except Exception:
        # Keep validator resilient: any exception => invalid w/ sentinels
        _write_debug_log(
            problem_dir=problem_dir,
            cfg=cfg,
            payload=payload,
            custom_model_src=custom_model_src,
            result=None,
            captured="",
            exc=Exception("validate() outer exception; see upstream logs"),
        )
        return {
            "speedup": 0.0,
            "runtime_us": RUNTIME_SENTINEL_US,
            "ref_runtime_us": RUNTIME_SENTINEL_US,
            "compiled": 0.0,
            "correctness": 0.0,
            "is_valid": 0.0,
        }

