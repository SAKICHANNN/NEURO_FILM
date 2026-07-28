"""Strict JSON decoding shared by persisted reference-match contracts."""

from __future__ import annotations

import json
from typing import Any


class _StrictJsonValueError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _StrictJsonValueError(f"non-standard JSON constant: {value}")


def strict_json_loads(document: str | bytes | bytearray) -> Any:
    """Decode RFC-compatible JSON while rejecting ambiguous object keys."""

    try:
        return json.loads(
            document,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except _StrictJsonValueError as exc:
        if isinstance(document, str):
            decoded = document
        else:
            try:
                decoded = bytes(document).decode("utf-8")
            except UnicodeDecodeError:
                decoded = ""
        raise json.JSONDecodeError(str(exc), decoded, 0) from exc


__all__ = ["strict_json_loads"]
