from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.stage_registry import StageRegistry
from kernel_evo.core.profile.artifacts import prepare_candidate_artifact_layout, write_json
from kernel_evo.core.profile.contracts import ProfilerRunConfig, run_profile_subprocess
from kernel_evo.core.stages.profile.agent import (
    ProfileExtractAgent,
    ProfileProgramInsight,
)
from kernel_evo.core.stages.profile.prompts import ProfileExtractPrompts
from kernel_evo.core.stages.profile.summary_compaction import summarize_profiler_for_llm
from kernel_evo.resources.paths import get_problem_dir


def _dedupe_insights(items: list[ProfileProgramInsight]) -> list[ProfileProgramInsight]:
    seen: set[str] = set()
    deduped: list[ProfileProgramInsight] = []
    for item in items:
        key = f"{item.type}|{item.insight}|{item.tag}|{item.severity}|{item.source}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_context(insights: list[ProfileProgramInsight]) -> str:
    if not insights:
        return ""
    lines = ["Profiler Guidance:"]
    for item in insights:
        lines.append(
            f"- {item.insight} [source={item.source or 'unknown'} tag={item.tag} severity={item.severity}]"
        )
    return "\n".join(lines)


@StageRegistry.register(description="Optional profiler stage that adds formatted optimization guidance")
class ProfileMutationContextStage(Stage):
    InputsModel = VoidInput
    OutputModel = StringContainer

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        run_config: dict[str, Any],
        prompts_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.run_config = dict(run_config)
        self.profiler_config = ProfilerRunConfig.from_run_config(self.run_config)
        system_prompt = ProfileExtractPrompts.system(prompts_dir=prompts_dir).replace(
            "{max_insights}",
            str(self.profiler_config.max_insights),
        )
        self.profile_extract_agent = ProfileExtractAgent(
            llm=llm,
            system_prompt_template=system_prompt,
            user_prompt_template=ProfileExtractPrompts.user(prompts_dir=prompts_dir),
        )

    def _is_profile_candidate(self, program: Program) -> bool:
        metrics = program.metrics
        return (
            self.profiler_config.enabled
            and float(metrics.get("compiled", 0.0)) > 0.0
            and float(metrics.get("correctness", 0.0)) > 0.0
        )

    def _should_run_ncu(self, program: Program) -> bool:
        speedup = float(program.metrics.get("speedup", 0.0) or 0.0)
        return speedup >= self.profiler_config.ncu_min_speedup

    def _problem_dir(self) -> Path:
        configured = str(self.run_config.get("problem_dir", "") or "").strip()
        if configured:
            candidate = Path(configured).expanduser().resolve()
            if candidate.exists():
                return candidate
        return get_problem_dir()

    def _run_config_path(self, problem_dir: Path) -> Path:
        configured_problem_dir = str(self.run_config.get("problem_dir", "") or "").strip()
        if configured_problem_dir:
            candidate = Path(configured_problem_dir).expanduser().resolve() / "run_config.json"
            if candidate.exists():
                return candidate
        experiment_dir = str(self.run_config.get("experiment_dir", "") or "").strip()
        if experiment_dir:
            candidate = Path(experiment_dir).expanduser().resolve() / "run_config.json"
            if candidate.exists():
                return candidate
        return problem_dir / "run_config.json"

    def _run_tool(self, module_name: str, args: list[str], cwd: Path) -> tuple[int, str, str]:
        proc = run_profile_subprocess(
            [sys.executable, "-m", module_name, *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    async def _extract_profile_insights(
        self,
        *,
        profiler_name: str,
        summary: dict[str, Any],
        code: str,
    ) -> list[ProfileProgramInsight]:
        llm_summary = summarize_profiler_for_llm(profiler_name=profiler_name, summary=summary)
        extracted = await self.profile_extract_agent.arun(
            profiler_name=profiler_name,
            summary_json=json.dumps(llm_summary, ensure_ascii=False, indent=2),
            code=code,
            max_insights=self.profiler_config.max_insights,
        )
        return [
            item.model_copy(update={"source": profiler_name})
            for item in extracted.insights
        ]

    async def compute(self, program: Program) -> StringContainer:
        if not self._is_profile_candidate(program):
            return StringContainer(data="")

        problem_dir = self._problem_dir()
        ref_arch_src = str(self.run_config.get("ref_arch_src", "") or "")
        if not ref_arch_src.strip():
            logger.warning("[ProfileMutationContextStage] Missing ref_arch_src; skipping profiler stage")
            return StringContainer(data="")

        layout = prepare_candidate_artifact_layout(
            run_config=self.run_config,
            problem_dir=problem_dir,
            program=program,
            code=program.code,
            ref_arch_src=ref_arch_src,
        )
        program.metadata["profile_artifact_dir"] = str(layout.root_dir)

        profiler_metadata: dict[str, Any] = {
            "enabled": True,
            "generation": program.generation,
            "program_id": program.id,
            "runners": {},
        }
        combined: list[ProfileProgramInsight] = []

        if "torch" in self.profiler_config.runners:
            torch_dir = layout.root_dir / "torch_profile"
            run_config_path = self._run_config_path(problem_dir)
            returncode, stdout, stderr = self._run_tool(
                "kernel_evo.tools.profile_torch",
                [
                    "--run-config",
                    str(run_config_path),
                    "--candidate-file",
                    str(layout.candidate_file),
                    "--reference-file",
                    str(layout.reference_file),
                    "--out-dir",
                    str(torch_dir),
                ],
                problem_dir,
            )
            torch_summary_path = torch_dir / "summary.json"
            torch_summary: dict[str, Any] = {
                "status": "failed" if returncode else "completed",
                "returncode": returncode,
                "stdout": stdout[-12000:],
                "stderr": stderr[-12000:],
            }
            if torch_summary_path.exists():
                try:
                    torch_summary = json.loads(torch_summary_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    torch_summary["summary_parse_error"] = str(exc)
            (torch_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (torch_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
            profiler_metadata["runners"]["torch"] = torch_summary
            if torch_summary.get("status") == "completed":
                extracted = await self._extract_profile_insights(
                    profiler_name="torch.profiler",
                    summary=torch_summary,
                    code=program.code,
                )
                combined.extend(extracted)

        if "ncu" in self.profiler_config.runners and self._should_run_ncu(program):
            ncu_dir = layout.root_dir / "ncu"
            run_config_path = self._run_config_path(problem_dir)
            returncode, stdout, stderr = self._run_tool(
                "kernel_evo.tools.profile_ncu",
                [
                    "--run-config",
                    str(run_config_path),
                    "--candidate-file",
                    str(layout.candidate_file),
                    "--reference-file",
                    str(layout.reference_file),
                    "--out-dir",
                    str(ncu_dir),
                ],
                problem_dir,
            )
            ncu_summary_path = ncu_dir / "summary.json"
            ncu_summary: dict[str, Any] = {
                "status": "failed" if returncode else "completed",
                "returncode": returncode,
                "stdout": stdout[-12000:],
                "stderr": stderr[-12000:],
            }
            if ncu_summary_path.exists():
                try:
                    ncu_summary = json.loads(ncu_summary_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    ncu_summary["summary_parse_error"] = str(exc)
            (ncu_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (ncu_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
            profiler_metadata["runners"]["ncu"] = ncu_summary
            if ncu_summary.get("status") == "completed":
                extracted = await self._extract_profile_insights(
                    profiler_name="ncu",
                    summary=ncu_summary,
                    code=program.code,
                )
                combined.extend(extracted)
        elif "ncu" in self.profiler_config.runners:
            profiler_metadata["runners"]["ncu"] = {
                "status": "skipped",
                "reason": (
                    f"speedup {float(program.metrics.get('speedup', 0.0) or 0.0):.4f} "
                    f"is below threshold {self.profiler_config.ncu_min_speedup:.4f}"
                ),
            }

        deduped = _dedupe_insights(combined)
        merged_payload = {
            "program_id": program.id,
            "generation": program.generation,
            "insights": [item.model_dump() for item in deduped],
            "formatted_context": _format_context(deduped),
            "profilers": profiler_metadata["runners"],
        }
        write_json(layout.merged_guidance_file, merged_payload)
        program.metadata["profiling"] = profiler_metadata
        return StringContainer(data=merged_payload["formatted_context"])
