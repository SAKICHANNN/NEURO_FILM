from __future__ import annotations

import ast
import json
from pathlib import Path

from src.color_match import MAX_REFERENCE_MATCH_BATCH_SOURCES


ROOT = Path(__file__).resolve().parents[1]
COLOR_MATCH = ROOT / "src" / "color_match"
SCHEMAS = ROOT / "configs" / "schemas"


def _reference_batch_schemas() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(SCHEMAS.glob("reference_*.schema.json"))
        if '"source_count"' in path.read_text(encoding="utf-8")
    )


def _source_index_contracts(value: object) -> tuple[dict[str, object], ...]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "source_index" and isinstance(child, dict):
                found.append(child)
            else:
                found.extend(_source_index_contracts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_source_index_contracts(child))
    return tuple(found)


def test_every_persisted_batch_schema_uses_the_single_source_limit() -> None:
    schemas = _reference_batch_schemas()
    assert len(schemas) == 29
    for path in schemas:
        payload = json.loads(path.read_text(encoding="utf-8"))
        properties = payload["properties"]
        assert properties["source_count"]["maximum"] == (
            MAX_REFERENCE_MATCH_BATCH_SOURCES
        ), path
        for collection in ("sources", "outputs"):
            if collection in properties:
                assert properties[collection]["maxItems"] == (
                    MAX_REFERENCE_MATCH_BATCH_SOURCES
                ), path
        for source_index in _source_index_contracts(payload):
            assert source_index["maximum"] == (
                MAX_REFERENCE_MATCH_BATCH_SOURCES - 1
            ), path


def test_legacy_file_reports_use_the_same_source_limit() -> None:
    for name in (
        "reference_match_report_v1.schema.json",
        "reference_match_replay_report_v1.schema.json",
    ):
        payload = json.loads(
            (SCHEMAS / name).read_text(encoding="utf-8")
        )
        assert payload["properties"]["outputs"]["maxItems"] == (
            MAX_REFERENCE_MATCH_BATCH_SOURCES
        )


def test_every_direct_python_batch_validator_uses_the_single_limit() -> None:
    checked: list[Path] = []
    for path in sorted(COLOR_MATCH.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "value.source_count <= 0" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "batch_limits"
            and any(
                alias.name == "MAX_REFERENCE_MATCH_BATCH_SOURCES"
                for alias in node.names
            )
            for node in tree.body
        )
        assert imported, path
        assert (
            "value.source_count > MAX_REFERENCE_MATCH_BATCH_SOURCES"
            in source
        ), path
        checked.append(path)
    assert len(checked) == 27
