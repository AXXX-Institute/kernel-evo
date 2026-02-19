import types
from typing import Annotated, Any, get_args, get_origin

import json_repair


def _unwrap_annotated(tp: Any) -> Any:
    # If something is Annotated[T, ...], unwrap to T
    if get_origin(tp) is Annotated:
        return get_args(tp)[0]
    return tp


def _extract_list_item_type(tp: Any) -> Any | None:
    """
    Return the element type if tp is list[T] (or Optional[list[T]]), else None.
    """
    tp = _unwrap_annotated(tp)
    origin = get_origin(tp)

    # list[T]
    if origin is list:
        args = get_args(tp)
        return args[0] if args else Any

    # Optional[list[T]] / Union[list[T], None]
    if origin in (types.UnionType, getattr(types, "UnionType", object), None) or origin is None:
        pass
    if origin in (types.UnionType, getattr(types, "UnionType", object)) or origin is getattr(
        __import__("typing"), "Union", object
    ):
        for a in get_args(tp):
            item = _extract_list_item_type(a)
            if item is not None:
                return item

    return None


def single_top_level_list_field(model_cls: type) -> tuple[str, Any] | None:
    """
    If model_cls has exactly one field and it is list[T], return (field_name, T).
    Otherwise return None.
    """
    fields = getattr(model_cls, "model_fields", None)  # pydantic v2
    if fields is None:
        fields = getattr(model_cls, "__fields__", None)  # pydantic v1
    if not fields:
        return None

    if len(fields) != 1:
        return None

    name, field = next(iter(fields.items()))

    # v2: FieldInfo.annotation ; v1: ModelField.outer_type_ (or .type_)
    ann = getattr(field, "annotation", None)
    if ann is None:
        ann = getattr(field, "outer_type_", None) or getattr(field, "type_", None)

    item_type = _extract_list_item_type(ann)
    if item_type is None:
        return None

    return name, item_type


def _validate_list_payload(
    schema: Any,
    field_name: str,
    items: Any,
    min_len: int = 0,
    max_len: int = 5,
) -> Any | None:
    """Validate schema with a single list field; clamp length if needed (e.g. 3–5 for lineage)."""
    if not isinstance(items, list):
        return None
    try:
        return schema.model_validate({field_name: items})
    except Exception:
        pass
    if not items:
        return None
    clamped = list(items)[:max_len]
    if len(clamped) < min_len and clamped:
        last = clamped[-1]
        clamped = clamped + [last] * (min_len - len(clamped))
    elif len(clamped) < min_len:
        return None
    try:
        return schema.model_validate({field_name: clamped})
    except Exception:
        return None


def normalize_and_validate_single_list_schema(schema: Any, obj: Any, llm_output: str) -> Any | None:
    """Recover when schema has a single top-level list field (broad replacement for insights workaround).

    Normalizes: raw list -> {field: list}, single dict -> {field: [dict]},
    and comma-separated objects without leading '[' -> wrap in '[...]' then validate.
    Tries optional clamp to [3, 5] when schema rejects (e.g. TransitionInsights).
    Returns validated model or None.
    """
    top = single_top_level_list_field(schema)
    if top is None:
        return None
    field_name, _ = top

    def validate(items: Any, min_len: int = 0) -> Any | None:
        return _validate_list_payload(schema, field_name, items, min_len=min_len, max_len=5)

    if isinstance(obj, list):
        v = validate(obj)
        if v is not None:
            return v
        v = validate(obj, min_len=3)
        if v is not None:
            return v
    if isinstance(obj, dict) and field_name not in obj:
        v = validate([obj])
        if v is not None:
            return v
        v = validate([obj], min_len=3)
        if v is not None:
            return v

    s = (llm_output or "").strip()
    if s and not s.lstrip().startswith("["):
        inner = s.rstrip()[:-1].rstrip() if s.rstrip().endswith("]") else s
        wrapped = "[" + inner + "]"
        try:
            obj2 = json_repair.loads(wrapped)
        except Exception:
            obj2 = None
        if isinstance(obj2, list):
            v = validate(obj2)
            if v is not None:
                return v
            v = validate(obj2, min_len=3)
            if v is not None:
                return v
        if isinstance(obj2, dict):
            v = validate([obj2])
            if v is not None:
                return v
    return None
