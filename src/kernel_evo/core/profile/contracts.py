from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_NCU_PATH = "ncu"

PROFILE_SUBPROCESS_TIMEOUT_SECONDS = 600


def run_profile_subprocess(
    cmd: list[str],
    *,
    timeout: float = PROFILE_SUBPROCESS_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run a profiler subprocess with a hard timeout, surfacing timeouts as a failed result."""
    try:
        return subprocess.run(cmd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = (stderr + f"\n[profiler] subprocess timed out after {timeout:.0f}s").lstrip("\n")
        return subprocess.CompletedProcess(cmd, returncode=124, stdout=stdout, stderr=stderr)


@dataclass(slots=True)
class ProfilerRunConfig:
    enabled: bool = False
    runners: tuple[str, ...] = ()
    max_insights: int = 4
    artifacts_dir: str = ""
    torch_warmup_steps: int = 2
    torch_active_steps: int = 3
    ncu_path: str = DEFAULT_NCU_PATH
    ncu_set: str = "full"
    ncu_kernel_name: str = ""
    ncu_extra_args: str = ""
    ncu_min_speedup: float = 1.0

    @classmethod
    def from_run_config(cls, run_config: dict[str, Any]) -> "ProfilerRunConfig":
        runners = tuple(
            str(item).strip()
            for item in run_config.get("profile_runners", []) or []
            if str(item).strip()
        )
        return cls(
            enabled=bool(run_config.get("profile_stage_enabled", False)),
            runners=runners,
            max_insights=int(run_config.get("profile_max_insights", 4)),
            artifacts_dir=str(run_config.get("profile_artifacts_dir", "") or ""),
            torch_warmup_steps=int(run_config.get("profile_torch_warmup_steps", 2)),
            torch_active_steps=int(run_config.get("profile_torch_active_steps", 3)),
            ncu_path=str(run_config.get("profile_ncu_path", DEFAULT_NCU_PATH)),
            ncu_set=str(run_config.get("profile_ncu_set", "full") or "full"),
            ncu_kernel_name=str(run_config.get("profile_ncu_kernel_name", "") or ""),
            ncu_extra_args=str(run_config.get("profile_ncu_extra_args", "") or ""),
            ncu_min_speedup=float(run_config.get("profile_ncu_min_speedup", 1.0)),
        )


@dataclass(slots=True)
class CandidateArtifactLayout:
    root_dir: Path
    candidate_file: Path
    reference_file: Path
    merged_guidance_file: Path
    tool_dirs: dict[str, Path] = field(default_factory=dict)
