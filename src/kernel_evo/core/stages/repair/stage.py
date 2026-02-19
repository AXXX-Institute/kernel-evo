from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from gigaevo.entrypoint.constants import MAX_CODE_LENGTH, MAX_MEMORY_MB, MAX_OUTPUT_SIZE
from kernel_evo.core.stages.repair.agent import RepairAgent
from kernel_evo.core.stages.repair.prompts import RepairPrompts
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.core_types import StageIO, StageState
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer, FloatDictContainer
from gigaevo.programs.stages.python_executors.execution import CallValidatorFunction
from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.programs.stages.validation import ValidateCodeStage

# Default path relative to package root (works for both dev and installed package)
# stage.py is at kernel_evo/core/stages/repair/stage.py -> 3 parents = kernel_evo
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
ERR_MEMORY_DIR = str(_PACKAGE_ROOT / "memory" / "full_err_mem")


class RepairInputs(StageIO):
    payload: AnyContainer
    context: Optional[AnyContainer]


@StageRegistry.register(description="Repair candidate code via LLM when validator fails; retries before giving up")
class RepairStage(Stage):
    """Runs validator; if it fails, invoke a repair LLM loop (up to N times).

    This stage always completes successfully from the DAG perspective, returning
    an empty metrics dict when all repair attempts fail. Failure details are
    recorded in program.metadata['repair_loop'] for downstream analysis.

    Notes:
    - The stage may mutate program.code in-place to the repaired version.
    - The validator is executed in an isolated subprocess (same as CallValidatorFunction).
    """

    InputsModel = RepairInputs
    OutputModel = FloatDictContainer

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        validator_path: Path,
        task_description: str,
        max_repairs: int = 5,
        safe_mode: bool = True,
        max_code_length: int = MAX_CODE_LENGTH,
        max_memory_mb: int | None = MAX_MEMORY_MB,
        max_output_size: int = MAX_OUTPUT_SIZE,
        use_memory_for_errors: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_repairs = int(max_repairs)
        self.use_memory_for_errors = bool(use_memory_for_errors)
        logger.info(f"[RepairStage] use_memory_for_errors: {self.use_memory_for_errors}")
        if self.use_memory_for_errors:
            from evo_memory_agent.shared_memory.memory import AmemGamMemory

            self.memory = AmemGamMemory(checkpoint_path=ERR_MEMORY_DIR, rebuild_interval=10)

        system_template = RepairPrompts.system()
        user_template = RepairPrompts.user()
        system_prompt = system_template.format(task_description=task_description)
        self._repair_agent = RepairAgent(
            llm=llm,
            system_prompt_template=system_prompt,
            user_prompt_template=user_template,
        )

        # Internal stages used as helpers
        self._validator_stage = CallValidatorFunction(
            path=validator_path,
            timeout=self.timeout,
            max_memory_mb=max_memory_mb,
            max_output_size=max_output_size,
        )
        self._validate_code_stage = ValidateCodeStage(
            timeout=min(60.0, float(self.timeout)),
            safe_mode=safe_mode,
            max_code_length=max_code_length,
        )

    def _prepare_payload(self, program: Program) -> Any:
        payload_obj = self.params.payload.data
        if isinstance(payload_obj, dict):
            # Always keep payload['code'] in sync with program.code, when present.
            payload = dict(payload_obj)
            if "code" in payload:
                payload["code"] = program.code
            return payload
        return payload_obj

    async def _run_code_validation(self, program: Program) -> str | None:
        """Run safe-mode validation on program.code; return error string or None."""
        self._validate_code_stage.attach_inputs({})
        res = await self._validate_code_stage.execute(program)
        if res.status == StageState.COMPLETED:
            return None
        if res.error is None:
            return "Code validation failed (no error details)"
        return res.error.pretty(include_traceback=True)

    async def _run_validator(self, program: Program) -> tuple[dict[str, float] | None, str | None]:
        """Run validator on current program code; return (metrics, error)."""
        payload = self._prepare_payload(program)
        inputs = {
            "payload": AnyContainer(data=payload),
            "context": self.params.context,
        }
        self._validator_stage.attach_inputs(inputs)
        res = await self._validator_stage.execute(program)
        if res.status == StageState.COMPLETED and res.output is not None:
            try:
                metrics = res.output.data
                if isinstance(metrics, dict):
                    return {str(k): float(v) for k, v in metrics.items()}, None
            except Exception:
                pass
            return {}, None
        if res.error is None:
            return None, "Validator failed (no error details)"
        return None, res.error.pretty(include_traceback=True)

    async def compute(self, program: Program) -> FloatDictContainer:
        original_code = program.code

        repair_md: dict[str, Any] = {
            "max_repairs": self.max_repairs,
            "attempts": [],
            "succeeded": False,
            "repairs_used": 0,
            "last_error": None,
        }

        last_invalid_code: str | None = None
        last_error_text: str | None = None

        # Total tries = initial try + max_repairs repair attempts
        for attempt in range(0, self.max_repairs + 1):
            # 1) Validate code (syntax + safe_mode) first
            validation_error = await self._run_code_validation(program)
            if validation_error is None:
                # 2) Run validator
                metrics, err = await self._run_validator(program)
                if err is None and metrics is not None:
                    if attempt > 0 and last_invalid_code is not None and last_error_text is not None:
                        # Generate diff between last_invalid_code and program.code
                        diff = difflib.unified_diff(
                            last_invalid_code.splitlines(keepends=True),
                            program.code.splitlines(keepends=True),
                            fromfile="before",
                            tofile="after",
                            lineterm="",
                        )
                        diff_str = "".join(diff)

                        memory_entry = f"Following error was fixed: {last_error_text}\n\nCode diff:\n{diff_str}"  # noqa: F841
                        # logger.info(f"[RepairStage] Saving memory entry: {memory_entry}")

                        # self.memory.save(
                        #     memory_entry
                        # )
                        # self.memory.rebuild()

                    repair_md["succeeded"] = True
                    repair_md["repairs_used"] = attempt
                    repair_md["last_error"] = None
                    program.metadata["repair_loop"] = repair_md
                    return FloatDictContainer(data=metrics)
                error_text = err or "Validator failed"
            else:
                error_text = validation_error

            repair_md["last_error"] = error_text
            last_error_text = error_text

            # Out of budget -> give up
            if attempt >= self.max_repairs:
                logger.info(
                    "[RepairStage] Exhausted repair budget ({}). Leaving code as-is.",
                    self.max_repairs,
                )
                break

            # 3) Call repair LLM
            try:
                if self.use_memory_for_errors:
                    memorized_error_fixes = self.memory.search(f"Relevant error fixes for: {error_text}")
                    error_text = (
                        error_text
                        + "\n\n ===RELEVANT ERROR FIXES===:"
                        + str(memorized_error_fixes)
                        + ";\n\n Use these fixes (if any) to fix the error."
                    )
                repair_out = await self._repair_agent.arun(
                    code=program.code,
                    error=error_text,
                    attempt=attempt + 1,
                    max_attempts=self.max_repairs,
                )
            except Exception as e:
                logger.error("[RepairStage] Repair LLM failed: {}", e)
                repair_md["attempts"].append(
                    {
                        "attempt": attempt + 1,
                        "error": error_text,
                        "llm_error": str(e),
                    }
                )
                break

            repair_md["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "error": error_text,
                    "explanation": getattr(repair_out, "explanation", ""),
                }
            )

            new_code = (getattr(repair_out, "code", "") or "").strip()
            if not new_code:
                logger.warning("[RepairStage] Repair LLM returned empty code; stopping.")
                break

            last_invalid_code = program.code
            program.code = new_code

        # Failed after all attempts: record metadata and return empty metrics.
        program.metadata["repair_loop"] = repair_md

        # If the LLM produced unsafe code and the last attempt errored, it can be
        # useful to restore the original code for safety/debugging.
        if not program.code or (program.code and len(program.code) > MAX_CODE_LENGTH * 4):
            program.code = original_code

        return FloatDictContainer(data={})
