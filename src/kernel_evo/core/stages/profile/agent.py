from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class ProfileProgramInsight(BaseModel):
    type: str = Field(description="Insight category")
    insight: str = Field(description="Actionable insight with evidence")
    tag: str = Field(description="Insight tag")
    severity: str = Field(description="Insight severity")
    source: str = Field(default="", description="Profiler source that produced the insight")


class ProfileProgramInsights(BaseModel):
    insights: list[ProfileProgramInsight] = Field(default_factory=list)


class ProfileExtractState(TypedDict):
    profiler_name: str
    summary_json: str
    code: str
    max_insights: int
    messages: list[BaseMessage]
    llm_response: Any
    structured_output: ProfileProgramInsights
    metadata: dict[str, Any]


class ProfileExtractAgent(LangGraphAgent):
    StateSchema = ProfileExtractState

    def __init__(
        self,
        *,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt_template: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.structured_llm = llm.with_structured_output(ProfileProgramInsights, include_raw=True)
        super().__init__(llm)

    def build_prompt(self, state: ProfileExtractState) -> ProfileExtractState:
        user_prompt = self.user_prompt_template.format(
            profiler_name=state["profiler_name"],
            max_insights=state["max_insights"],
            summary_json=state["summary_json"],
            code=state["code"],
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt_template),
            HumanMessage(content=user_prompt),
        ]
        return state

    async def acall_llm(self, state: ProfileExtractState) -> ProfileExtractState:
        result: Any = await self.structured_llm.ainvoke(state["messages"])
        if isinstance(result, dict) and "parsed" in result:
            parsed = result.get("parsed")
            if parsed is None:
                raw = result.get("raw")
                raw_text = self._extract_text_from_raw_message(raw)
                parsed = self.parse_llm_structured(ProfileProgramInsights, raw_text)
            state["llm_response"] = parsed
        else:
            state["llm_response"] = result
        return state

    def parse_response(self, state: ProfileExtractState) -> ProfileExtractState:
        resp = state["llm_response"]
        if isinstance(resp, ProfileProgramInsights):
            state["structured_output"] = resp
            return state
        if isinstance(resp, dict):
            state["structured_output"] = ProfileProgramInsights.model_validate(resp)
            return state
        raise TypeError(f"Unexpected profile extraction response type: {type(resp).__name__}")

    async def arun(
        self,
        *,
        profiler_name: str,
        summary_json: str,
        code: str,
        max_insights: int,
    ) -> ProfileProgramInsights:
        initial_state: ProfileExtractState = {
            "profiler_name": profiler_name,
            "summary_json": summary_json,
            "code": code,
            "max_insights": max_insights,
            "messages": [],
            "llm_response": None,  # type: ignore
            "structured_output": None,  # type: ignore
            "metadata": {},
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["structured_output"]
