from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.color_match.strict_json import strict_json_loads


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "encoded",
    (
        '{"a":1,"a":1}',
        '{"outer":{"a":1,"a":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":"\\ud800"}',
        '{"\\udfff":"value"}',
    ),
)
def test_strict_json_rejects_ambiguous_or_nonstandard_input(
    encoded: str,
) -> None:
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads(encoded)


def test_strict_json_retains_order_independent_standard_values() -> None:
    assert strict_json_loads(
        b'{"nested":{"ok":true},"items":[1,2,null]}'
    ) == {
        "nested": {"ok": True},
        "items": [1, 2, None],
    }


def test_strict_json_wraps_invalid_utf8_and_excessive_nesting() -> None:
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads(b'{"value":"\xff"}')
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads("[" * 2000 + "]" * 2000)
    with pytest.raises(json.JSONDecodeError, match="nesting depth"):
        strict_json_loads("[" * 65 + "0" + "]" * 65)
    assert strict_json_loads("[" * 64 + "0" + "]" * 64)


def test_persisted_color_match_modules_do_not_bypass_strict_json() -> None:
    bypasses = []
    for path in sorted((ROOT / "src" / "color_match").rglob("*.py")):
        if path.name == "strict_json.py":
            continue
        if "json.loads(" in path.read_text(encoding="utf-8"):
            bypasses.append(path.name)
    assert bypasses == []


def test_strict_json_calls_do_not_override_the_shared_policy() -> None:
    invalid_calls: list[tuple[str, int]] = []
    for path in sorted((ROOT / "src" / "color_match").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "strict_json_loads"
                and (len(node.args) != 1 or node.keywords)
            ):
                invalid_calls.append((path.name, node.lineno))
    assert invalid_calls == []
