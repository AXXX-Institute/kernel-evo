"""Recover LLM JSON output: extract text from messages/response, parse and normalize into schema."""

import json
from typing import Any

import json_repair
from loguru import logger

from kernel_generation.core.llm.models.extract_schema_utils import (
    normalize_and_validate_single_list_schema,
)


def extract_text_from_raw_message(raw_message: Any) -> str:
    """Best-effort extraction of JSON-like text from a LangChain message.

    When structured outputs use function/tool calling, the useful JSON often
    lives in function_call.arguments or tool_calls[*].function.arguments.
    """
    if raw_message is None:
        return ""
    if isinstance(raw_message, str):
        return raw_message

    content = getattr(raw_message, "content", None)
    if isinstance(content, str) and content.strip():
        return content

    additional_kwargs = getattr(raw_message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        function_call = additional_kwargs.get("function_call")
        if isinstance(function_call, dict):
            arguments = function_call.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments
        tool_calls = additional_kwargs.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first = tool_calls[0]
            if isinstance(first, dict):
                fn = first.get("function")
                if isinstance(fn, dict):
                    arguments = fn.get("arguments")
                    if isinstance(arguments, str) and arguments.strip():
                        return arguments

    tool_calls_attr = getattr(raw_message, "tool_calls", None)
    if isinstance(tool_calls_attr, list) and tool_calls_attr:
        first = tool_calls_attr[0]
        if isinstance(first, dict):
            args = first.get("args")
            if isinstance(args, (dict, list)):
                return json.dumps(args)
            if isinstance(args, str) and args.strip():
                return args
            arguments = first.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                return arguments
            fn = first.get("function")
            if isinstance(fn, dict):
                arguments = fn.get("arguments")
                if isinstance(arguments, str) and arguments.strip():
                    return arguments

    return ""


def extract_text_from_openai_response_payload(payload: Any) -> str:
    """Extract model content/arguments from an OpenAI-style chat completion JSON payload."""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not (isinstance(choices, list) and choices):
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content
    function_call = msg.get("function_call")
    if isinstance(function_call, dict):
        arguments = function_call.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return arguments
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        tc0 = tool_calls[0]
        if isinstance(tc0, dict):
            fn = tc0.get("function")
            if isinstance(fn, dict):
                arguments = fn.get("arguments")
                if isinstance(arguments, str) and arguments.strip():
                    return arguments
    return ""


def parse_llm_structured(schema: Any, llm_output: str) -> Any:
    """Parse and validate LLM output into a Pydantic model using json_repair.

    Uses extract_schema_utils.normalize_and_validate_single_list_schema for the
    single top-level list field recover (list -> {field: list}, single dict -> {field: [dict]},
    comma-separated wrap, optional 3–5 clamp).
    """
    obj = json_repair.loads(llm_output)
    while isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]

    try:
        return schema.model_validate(obj)
    except Exception:
        pass

    recovered = normalize_and_validate_single_list_schema(schema, obj, llm_output)
    if recovered is not None:
        return recovered

    return schema.model_validate(obj)


def salvage_structured_from_exception(schema: Any, e: Exception) -> Any | None:
    """When the client raises before returning include_raw payload, extract from e.response and parse."""
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    try:
        data = resp.json()
    except Exception:
        try:
            data = json.loads(getattr(resp, "text", "") or "")
        except Exception:
            logger.debug("Failed to load JSON from exception response: {}", e)
            return None
    text = extract_text_from_openai_response_payload(data)
    if not text or not text.strip():
        return None
    try:
        return parse_llm_structured(schema, text)
    except Exception:
        logger.debug("Failed to parse salvaged structured output: {}", e)
        return None
