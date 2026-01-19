from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


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
        # Heuristic: our problem dir contains metrics.yaml
        if (pp / "metrics.yaml").exists():
            candidates.append(pp)

    if candidates:
        # Prefer the shallowest path (most likely the actual problem dir)
        candidates.sort(key=lambda x: len(x.parts))
        return candidates[0]

    # Fallback: common location in this workspace
    fallback = Path("/home/sivtsov/kernel_generation")
    return fallback


def build_context() -> dict[str, Any]:
    """
    Context provider for the contextual GigaEvo pipeline.

    We intentionally keep the *evolving programs* free of:
    - `import os` / `import sys` / `import subprocess`
    - env var reads
    - file operations

    Instead, this file provides everything via a context dict loaded from
    `run_config.json` written by the runner script.
    """
    problem_dir = _find_problem_dir()
    cfg_path = problem_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing {cfg_path}. Run "
            "`kernel_generation/scripts/generate_and_eval_single_sample_gigaevo.py` "
            "to create it."
        )

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("run_config.json must contain a JSON object")

    # Minimal schema check
    ref_arch_src = data.get("ref_arch_src")
    if not isinstance(ref_arch_src, str) or not ref_arch_src.strip():
        raise ValueError("run_config.json must contain a non-empty 'ref_arch_src' string")

    return data


