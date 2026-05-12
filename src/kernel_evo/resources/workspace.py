from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


_RUNTIME_TEMPLATE_FILES: tuple[str, ...] = (
    "metrics.yaml",
    "validate.py",
    "context.py",
)


@dataclass(slots=True)
class ProblemWorkspace:
    root_dir: Path
    run_config_file: Path
    task_description_file: Path
    initial_programs_dir: Path


def _copy_template_file(*, resources_dir: Path, workspace_root: Path, relative_path: str) -> None:
    source = resources_dir / relative_path
    if not source.exists():
        raise FileNotFoundError(f"Missing runtime template: {source}")

    destination = workspace_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_problem_workspace(*, resources_dir: Path, workspace_root: Path) -> ProblemWorkspace:
    root_dir = workspace_root.expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in _RUNTIME_TEMPLATE_FILES:
        _copy_template_file(
            resources_dir=resources_dir,
            workspace_root=root_dir,
            relative_path=relative_path,
        )

    initial_programs_dir = root_dir / "initial_programs"
    initial_programs_dir.mkdir(parents=True, exist_ok=True)

    return ProblemWorkspace(
        root_dir=root_dir,
        run_config_file=root_dir / "run_config.json",
        task_description_file=root_dir / "task_description.txt",
        initial_programs_dir=initial_programs_dir,
    )
