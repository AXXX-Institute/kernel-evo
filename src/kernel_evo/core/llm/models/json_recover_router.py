import random
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableConfig
from langfuse.langchain import CallbackHandler

from gigaevo.llm.models import MultiModelRouter, _with_langfuse
from gigaevo.llm.token_tracking import TokenTracker

from loguru import logger

from kernel_evo.core.llm.models.json_recover_utils import (
    extract_text_from_raw_message,
    parse_llm_structured,
    salvage_structured_from_exception,
)


class MultiModelRouterWithJsonRecover(MultiModelRouter):
    """MultiModelRouter subclass that uses JSON-recovery on structured output (include_raw + salvage from exception)."""

    def with_structured_output(self, schema: Any, include_raw: bool = True, **kwargs) -> "_StructuredOutputRouter":
        logger.info(f"MultiModelRouterWithJsonRecover.with_structured_output: {schema}")
        wrapped = [m.with_structured_output(schema, include_raw=True, **kwargs) for m in self.models]
        return _StructuredOutputRouter(
            wrapped,
            self.model_names,
            self.probabilities,
            self._langfuse,
            self._tracker,
            schema,
        )


class _StructuredOutputRouter(Runnable):
    """Structured output router: include_raw, recover from raw or from exception."""

    def __init__(
        self,
        models: list,
        model_names: list[str],
        probabilities: list[float],
        langfuse: CallbackHandler | None,
        tracker: TokenTracker,
        schema: Any,
    ):
        self._models = models
        self._names = model_names
        self._probs = probabilities
        self._langfuse = langfuse
        self._tracker = tracker
        self._schema = schema

    def _select(self) -> tuple[Any, str]:
        idx = random.choices(range(len(self._models)), weights=self._probs)[0]
        return self._models[idx], self._names[idx]

    def _config(self, config: RunnableConfig | None, model_name: str) -> RunnableConfig:
        return _with_langfuse(config, self._langfuse, model_name)

    def _process(self, response: Any, name: str) -> Any:
        if not isinstance(response, dict):
            return response
        raw = response.get("raw")
        if raw is not None:
            self._tracker.track(raw, name)
        parsed = response.get("parsed")
        if parsed is not None and response.get("parsing_error") is None:
            return parsed
        raw_text = extract_text_from_raw_message(raw)
        if not raw_text.strip():
            return parsed
        return parse_llm_structured(self._schema, raw_text)

    def invoke(self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs) -> Any:
        model, name = self._select()
        try:
            out = model.invoke(input, self._config(config, name), **kwargs)
            return self._process(out, name)
        except Exception as e:
            salvaged = salvage_structured_from_exception(self._schema, e)
            if salvaged is not None:
                return salvaged
            raise

    async def ainvoke(self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs) -> Any:
        model, name = self._select()
        try:
            out = await model.ainvoke(input, self._config(config, name), **kwargs)
            return self._process(out, name)
        except Exception as e:
            salvaged = salvage_structured_from_exception(self._schema, e)
            if salvaged is not None:
                return salvaged
            raise
