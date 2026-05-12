from __future__ import annotations

from pathlib import Path

from kernel_evo.resources.prompt_loader import prepare_prompts_for_experiment
from kernel_evo.resources.workspace import prepare_problem_workspace


ROOT = Path(__file__).resolve().parents[1]
RESOURCES_DIR = ROOT / "src" / "kernel_evo" / "resources"


def test_prepare_problem_workspace_copies_runtime_assets(tmp_path: Path) -> None:
    workspace = prepare_problem_workspace(
        resources_dir=RESOURCES_DIR,
        workspace_root=tmp_path / "run" / "problem",
    )

    assert workspace.root_dir == (tmp_path / "run" / "problem").resolve()
    assert workspace.run_config_file == workspace.root_dir / "run_config.json"
    assert workspace.task_description_file == workspace.root_dir / "task_description.txt"
    assert workspace.initial_programs_dir == workspace.root_dir / "initial_programs"
    assert workspace.initial_programs_dir.is_dir()

    for relative_path in ("metrics.yaml", "validate.py", "context.py"):
        copied = workspace.root_dir / relative_path
        source = RESOURCES_DIR / relative_path
        assert copied.exists()
        assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_prepare_prompts_for_experiment_writes_inside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "run" / "problem"
    prompts_dir = prepare_prompts_for_experiment(workspace_root, "triton")

    assert prompts_dir == workspace_root.resolve() / "prompts"
    assert (prompts_dir / "mutation" / "system.txt").exists()
    assert (prompts_dir / "repair" / "system.txt").exists()
