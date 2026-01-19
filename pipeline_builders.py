from __future__ import annotations

"""
KernelGeneration-specific pipeline builders.

These are used to tweak the GigaEvo pipeline *without* modifying `gigaevo-core-internal`:
- Pipeline config (`pipeline=...`) can live under `kernel_generation/config/` via Hydra searchpath.
- The pipeline builder target can point at `kernel_generation.*` (ensure repo root is on PYTHONPATH).
"""

from gigaevo.entrypoint.constants import DEFAULT_STAGE_TIMEOUT
from gigaevo.entrypoint.default_pipelines import ContextPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer
from gigaevo.programs.stages.stage_registry import StageRegistry


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

        # Remove the entrypoint call stage and any edges/deps referencing it.
        self.remove_stage("CallProgramFunction")

        # Replace it with a tiny stage that exposes Program.code as the validator payload.
        self.add_stage(
            "ProgramCodeAsPayload",
            lambda: ProgramCodeAsPayload(timeout=DEFAULT_STAGE_TIMEOUT),
        )
        self.add_data_flow_edge(
            "ProgramCodeAsPayload", "CallValidatorFunction", "payload"
        )

        # Ensure we only evaluate if the code passed the (safe-mode) validator.
        self.add_exec_dep(
            "ProgramCodeAsPayload",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )


class DirectCodeContextPipelineNoInsightsLineageBuilder(DirectCodeContextPipelineBuilder):
    """
    Variant of `DirectCodeContextPipelineBuilder` that disables the LLM-heavy analysis stages:
      - InsightsStage (structured JSON output, frequently breaks on OpenRouter free models)
      - LineageStage and its derived stages

    MutationContextStage supports these inputs as Optional, so evolution continues with
    a "metrics-only" mutation context (plus any other remaining optional context).
    """

    def __init__(self, ctx):
        super().__init__(ctx)

        # LLM analysis stages that frequently fail due to strict structured output parsing.
        self.remove_stage("InsightsStage")
        self.remove_stage("LineageStage")
        self.remove_stage("LineagesFromAncestors")
        self.remove_stage("LineagesToDescendants")








