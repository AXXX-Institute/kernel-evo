"""Repair agent for fixing compilation/runtime errors.

This agent is used by RepairStage to iteratively repair a candidate program
based on validator/compilation errors.

All LLM-related logic lives here.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class RepairStructuredOutput(BaseModel):
    """Structured output from the repair LLM."""

    explanation: str = Field(description="1-3 sentences describing the fix")
    code: str = Field(description="The complete repaired Python program code")


class RepairState(TypedDict):
    # Inputs
    code: str
    error: str
    attempt: int
    max_attempts: int

    # LLM interaction
    messages: list[BaseMessage]
    llm_response: Any

    # Output
    structured_output: RepairStructuredOutput

    # Metadata
    metadata: dict[str, Any]


class RepairAgent(LangGraphAgent):
    """Agent that repairs code using validator error messages."""

    StateSchema = RepairState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt_template: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.structured_llm = llm.with_structured_output(RepairStructuredOutput, include_raw=True)
        super().__init__(llm)

    def build_prompt(self, state: RepairState) -> RepairState:
        user_prompt = self.user_prompt_template.format(
            attempt=state["attempt"],
            max_attempts=state["max_attempts"],
            error=state["error"],
            code=state["code"],
        )

        state["messages"] = [
            SystemMessage(content=self.system_prompt_template),
            HumanMessage(content=user_prompt),
        ]
        state.setdefault("metadata", {})["attempt"] = state["attempt"]
        return state

    async def acall_llm(self, state: RepairState) -> RepairState:
        """Call LLM with structured output and robustly parse failures."""
        try:
            result: Any = await self.structured_llm.ainvoke(state["messages"])
        except Exception as e:
            salvaged = self._try_salvage_structured_from_exception(RepairStructuredOutput, e)
            if salvaged is not None:
                state["llm_response"] = salvaged
                md = state.setdefault("metadata", {})
                md["structured_fallback_used"] = True
                md["structured_fallback_reason"] = type(e).__name__
                logger.warning(
                    "[RepairAgent] Structured parsing failed; salvaged structured output from raw httpx.Response"
                )
                return state
            logger.error(f"[RepairAgent] Error calling LLM: {e}")
            raise

        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            raw = result.get("raw")
            parsing_error = result.get("parsing_error")

            if parsed is None:
                raw_text = self._extract_text_from_raw_message(raw)
                parsed = self.parse_llm_structured(RepairStructuredOutput, raw_text)
                md = state.setdefault("metadata", {})
                md["json_repair_used"] = True
                if parsing_error:
                    md["structured_parsing_error"] = str(parsing_error)
                logger.debug("[RepairAgent] Repaired malformed JSON via json_repair")

            state["llm_response"] = parsed
        else:
            state["llm_response"] = result

        state.setdefault("metadata", {})["model_used"] = "llm"
        return state

    def parse_response(self, state: RepairState) -> RepairState:
        resp = state["llm_response"]
        if isinstance(resp, RepairStructuredOutput):
            state["structured_output"] = resp
            return state
        if isinstance(resp, dict):
            state["structured_output"] = RepairStructuredOutput.model_validate(resp)
            return state
        raise TypeError(f"Unexpected repair response type: {type(resp).__name__}")

    async def arun(self, *, code: str, error: str, attempt: int, max_attempts: int) -> RepairStructuredOutput:
        initial_state: RepairState = {
            "code": code,
            "error": error,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "messages": [],
            "llm_response": None,  # type: ignore
            "structured_output": None,  # type: ignore
            "metadata": {},
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["structured_output"]
