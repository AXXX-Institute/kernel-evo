from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langchain_openai")

from kernel_evo.core.stages.profile.stage import ProfileMutationContextStage  # noqa: E402


def test_run_config_path_prefers_configured_problem_dir(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir(parents=True)
    run_config_path = problem_dir / "run_config.json"
    run_config_path.write_text("{}", encoding="utf-8")

    stage = object.__new__(ProfileMutationContextStage)
    stage.run_config = {"problem_dir": str(problem_dir)}

    resolved = ProfileMutationContextStage._run_config_path(stage, tmp_path / "unused")
    assert resolved == run_config_path


def test_problem_dir_prefers_configured_problem_dir(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir(parents=True)

    stage = object.__new__(ProfileMutationContextStage)
    stage.run_config = {"problem_dir": str(problem_dir)}

    resolved = ProfileMutationContextStage._problem_dir(stage)
    assert resolved == problem_dir.resolve()
