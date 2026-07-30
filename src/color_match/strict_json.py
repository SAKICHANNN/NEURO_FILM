"""Strict JSON decoding shared by persisted reference-match contracts."""

from __future__ import annotations

import json
import math
from typing import Any


class _StrictJsonValueError(ValueError):
    pass


_MAX_JSON_NESTING_DEPTH = 64


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _StrictJsonValueError(f"non-standard JSON constant: {value}")


def _validate_decoded_value(value: Any, *, depth: int = 0) -> None:
    if depth > _MAX_JSON_NESTING_DEPTH:
        raise _StrictJsonValueError(
            "decoded JSON exceeds the nesting depth limit"
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _StrictJsonValueError(
                "decoded JSON number is not finite"
            )
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise _StrictJsonValueError(
                "decoded JSON string is not Unicode scalar text"
            ) from exc
        return
    if isinstance(value, list):
        for child in value:
            _validate_decoded_value(child, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_decoded_value(key, depth=depth + 1)
            _validate_decoded_value(child, depth=depth + 1)


def _decoded_document(document: object) -> str:
    if isinstance(document, str):
        return document
    if isinstance(document, (bytes, bytearray)):
        try:
            return bytes(document).decode("utf-8")
        except UnicodeDecodeError:
            return ""
    return ""


def strict_json_loads(document: str | bytes | bytearray) -> Any:
    """Decode RFC-compatible JSON while rejecting ambiguous object keys."""

    try:
        value = json.loads(
            document,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        _validate_decoded_value(value)
        return value
    except (
        _StrictJsonValueError,
        RecursionError,
        UnicodeDecodeError,
    ) as exc:
        raise json.JSONDecodeError(
            str(exc),
            _decoded_document(document),
            0,
        ) from exc


__all__ = ["strict_json_loads"]
