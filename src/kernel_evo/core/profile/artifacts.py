from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gigaevo.programs.program import Program

from kernel_evo.core.profile.contracts import CandidateArtifactLayout


def _artifact_root(run_config: dict[str, Any], problem_dir: Path) -> Path:
    configured = str(run_config.get("profile_artifacts_dir", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    experiment_dir = str(run_config.get("experiment_dir", "") or "").strip()
    if experiment_dir:
        return Path(experiment_dir).expanduser().resolve() / "artifacts"
    return (problem_dir / "profiling_artifacts").resolve()


def prepare_candidate_artifact_layout(
    *,
    run_config: dict[str, Any],
    problem_dir: Path,
    program: Program,
    code: str,
    ref_arch_src: str,
) -> CandidateArtifactLayout:
    root = _artifact_root(run_config, problem_dir) / str(program.generation) / program.id
    root.mkdir(parents=True, exist_ok=True)

    candidate_file = root / "candidate.py"
    candidate_file.write_text(code, encoding="utf-8")

    reference_file = root / "reference.py"
    reference_file.write_text(ref_arch_src, encoding="utf-8")

    return CandidateArtifactLayout(
        root_dir=root,
        candidate_file=candidate_file,
        reference_file=reference_file,
        merged_guidance_file=root / "merged_profile_guidance.json",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

