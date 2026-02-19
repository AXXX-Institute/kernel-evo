"""Canonical resolution of problem dir and repo root.

Used by context.py (run_config.json), validate.py (metrics.yaml, KernelBench path),
and evolve.py (resources dir). Single place to change behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sentinel files that identify the problem/resources directory
RUN_CONFIG_JSON = "run_config.json"
METRICS_YAML = "metrics.yaml"
MAX_WALK_UP = 8


def get_resources_dir() -> Path:
    """Package resources dir: kernel_evo/resources (metrics.yaml, validate.py, config, etc.)."""
    # This file is at kernel_evo/resources/paths.py
    return Path(__file__).resolve().parent


def get_repo_root() -> Path:
    """Repo root: walk up from resources dir looking for pyproject.toml or tasks; else resources.parent."""
    p = get_resources_dir()
    for _ in range(MAX_WALK_UP):
        if not p.exists() or p == p.parent:
            break
        if (p / "pyproject.toml").exists() or (p / "tasks").exists():
            return p
        p = p.parent
    return get_resources_dir().parent


def get_problem_dir() -> Path:
    """
    Problem dir: directory containing run_config.json and/or metrics.yaml.

    Resolved in order:
    1. sys.path entries that contain run_config.json (then metrics.yaml)
    2. Walk up from resources dir for run_config.json then metrics.yaml
    3. Fallback: get_resources_dir()

    exec_runner runs code without __file__ but prepends the problem dir to sys.path,
    so we check sys.path first. When running from `kernel_evo evolve`, fallback is used.
    """
    # 1. sys.path: prefer run_config.json (current run), then metrics.yaml
    for p in sys.path:
        try:
            pp = Path(p)
        except Exception:
            continue
        if not pp.exists() or not pp.is_dir():
            continue
        if (pp / RUN_CONFIG_JSON).exists():
            return pp
    candidates_metrics: list[Path] = []
    for p in sys.path:
        try:
            pp = Path(p)
        except Exception:
            continue
        if not pp.exists() or not pp.is_dir():
            continue
        if (pp / METRICS_YAML).exists():
            candidates_metrics.append(pp)
    if candidates_metrics:
        candidates_metrics.sort(key=lambda x: len(x.parts))
        return candidates_metrics[0]

    # 2. Walk up from resources dir (and from cwd for context.py compatibility)
    def walk_up(start: Path) -> Path | None:
        current = start.resolve()
        for _ in range(MAX_WALK_UP):
            if not current.exists() or current == current.parent:
                break
            if (current / RUN_CONFIG_JSON).exists():
                return current
            if (current / METRICS_YAML).exists():
                return current
            current = current.parent
        return None

    try:
        root = walk_up(get_resources_dir())
        if root is not None:
            return root
    except NameError:
        pass
    root = walk_up(Path.cwd())
    if root is not None:
        return root

    # 3. Fallback
    return get_resources_dir()
