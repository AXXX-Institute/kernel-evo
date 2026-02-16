"""
KernelGeneration-specific pipeline builders.

These are used to tweak the GigaEvo pipeline *without* modifying `gigaevo-core-internal`:
- Pipeline config (`pipeline=...`) can live under `kernel_generation/config/` via Hydra searchpath.
- The pipeline builder target can point at `kernel_generation.*` (ensure repo root is on PYTHONPATH).
"""

import json

from gigaevo.entrypoint.constants import DEFAULT_STAGE_TIMEOUT
from gigaevo.entrypoint.default_pipelines import ContextPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer
from gigaevo.programs.stages.stage_registry import StageRegistry
from kernel_generation.core.stages.repair import RepairStage

MAX_PROGRAM_REPAIRS = 10


@StageRegistry.register(description="Expose Program.code as a payload for downstream stages")
class ProgramCodeAsPayload(Stage):
    """
    Simple utility stage: returns the current Program.code (plus small metadata) as an AnyContainer.

    This enables pipelines where the evolving artifact *is* the final code to evaluate
    (i.e., no `def entrypoint(...)` wrapper function is needed).
    """

    InputsModel = VoidInput
    OutputModel = AnyContainer

    async def compute(self, program: Program) -> AnyContainer:
        # Use a dict payload so validators can access program identity in debug logs.
        # `kernel_generation/validate.py` already supports dict payloads (it extracts "code").
        return AnyContainer(
            data={
                "code": program.code,
                "program_id": program.id,
                "name": program.name,
                "generation": program.generation,
                # In this codebase Lineage.parents is `list[str]` (program ids).
                "parent_ids": list(program.lineage.parents),
            }
        )


class DirectCodeContextPipelineBuilder(ContextPipelineBuilder):
    """
    Context-enabled pipeline variant where the evolving artifact is the final code itself.

    Differences vs ContextPipelineBuilder:
      - Removes `CallProgramFunction(entrypoint)` entirely
      - Feeds Program.code directly into the validator as the payload

    This supports workflows where programs are "fully-contained code" (no entrypoint),
    while still allowing a `context.py` to provide runtime config to the validator.
    """

    def __init__(self, ctx: EvolutionContext):
        super().__init__(ctx)

        # We run syntax/safe-mode checks inside RepairStage so that broken candidates
        # can be repaired instead of being dropped by an early-stage failure.
        self.remove_stage("ValidateCodeStage")

        # Remove the entrypoint call stage and any edges/deps referencing it.
        self.remove_stage("CallProgramFunction")

        # Replace the validator stage with a repair loop that retries validation.
        self.remove_stage("CallValidatorFunction")

        # Feed Program.code directly into the repair/validator loop.
        self.add_stage(
            "ProgramCodeAsPayload",
            lambda: ProgramCodeAsPayload(timeout=DEFAULT_STAGE_TIMEOUT),
        )

        validator_path = ctx.problem_ctx.problem_dir / "validate.py"

        # Read use_memory_for_errors from run_config.json if available
        use_memory_for_errors = False
        run_config_path = ctx.problem_ctx.problem_dir / "run_config.json"
        if run_config_path.exists():
            try:
                run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                use_memory_for_errors = bool(run_config.get("use_memory_for_errors", False))
            except Exception:
                pass  # Use default if config can't be read

        self.add_stage(
            "RepairStage",
            lambda: RepairStage(
                timeout=DEFAULT_STAGE_TIMEOUT,
                llm=ctx.llm_wrapper,
                validator_path=validator_path,
                task_description=ctx.problem_ctx.task_description,
                max_repairs=MAX_PROGRAM_REPAIRS,
                use_memory_for_errors=use_memory_for_errors,
            ),
        )

        # Wire payload+context into RepairStage
        self.add_data_flow_edge("ProgramCodeAsPayload", "RepairStage", "payload")
        self.add_data_flow_edge("AddContext", "RepairStage", "context")

        # Replace upstream edge for metrics merge: validator metrics now come from RepairStage.
        self.add_data_flow_edge("RepairStage", "MergeMetricsStage", "first")


class DirectCodeContextPipelineNoInsightsLineageBuilder(DirectCodeContextPipelineBuilder):
    """
    Variant of `DirectCodeContextPipelineBuilder` that disables the LLM-heavy analysis stages:
      - InsightsStage (structured JSON output, frequently breaks on OpenRouter free models)
      - LineageStage and its derived stages

    MutationContextStage supports these inputs as Optional, so evolution continues with
    a "metrics-only" mutation context (plus any other remaining optional context).

    NOTE: This builder keeps InsightsStage running so that compilation_fix insights
    can be generated from failed programs. These insights flow to MutationContextStage
    and are used by the mutation agent to fix compilation errors.
    """

    def __init__(self, ctx):
        super().__init__(ctx)

        # Keep InsightsStage running to generate compilation_fix insights from failed programs.
        # The insights flow to MutationContextStage and are used by the mutation agent.
        #
        # LineageStage is removed because it requires successful execution to generate
        # transition analysis between parent-child programs.
        self.remove_stage("LineageStage")
        self.remove_stage("LineagesFromAncestors")
        self.remove_stage("LineagesToDescendants")
