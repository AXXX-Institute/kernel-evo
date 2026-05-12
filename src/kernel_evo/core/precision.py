from __future__ import annotations


VALID_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16", "fp8")
VALID_RUNTIME_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16")


def normalize_precision_string(value: str | None, *, default: str = "fp32") -> str:
    precision = str(value or "").strip().lower()
    return precision or default


def resolve_runtime_precision_string(
    requested_precision: str | None,
    runtime_precision: str | None = None,
) -> str:
    explicit = normalize_precision_string(runtime_precision, default="")
    if explicit:
        return explicit

    requested = normalize_precision_string(requested_precision)
    if requested == "fp8":
        return "bf16"
    return requested
