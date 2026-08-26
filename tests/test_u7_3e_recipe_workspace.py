from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from src.inference.recipe_workspace import (
    RecipeWorkspaceError,
    build_offline_recipe_workspace,
    materialize_offline_recipe_workspace,
)


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))


def test_ready_workspace_reuses_history_and_previews(tmp_path: Path) -> None:
    source = Path("outputs/eval/u7_2_three_stock_24mp_smoke")
    first = build_offline_recipe_workspace(source)
    second = build_offline_recipe_workspace(source)
    assert first == second
    assert set(first) == {
        "index.html",
        "history.html",
        "previews.html",
        "workspace_receipt.json",
    }
    receipt = json.loads(first["workspace_receipt.json"])
    assert receipt["catalog_counts"] == {"discovered": 3, "valid": 3, "invalid": 0}
    for name in ("index.html", "history.html", "previews.html"):
        assert receipt["pages"][name]["sha256"] == hashlib.sha256(first[name]).hexdigest()
    parser = _Parser()
    parser.feed(first["index.html"].decode())
    assert "main" in parser.tags and "nav" in parser.tags and "h1" in parser.tags
    assert parser.links == ["history.html", "previews.html"]
    assert b"http://" not in first["index.html"] and b"https://" not in first["index.html"]

    destination = tmp_path / "workspace"
    materialized = materialize_offline_recipe_workspace(source, destination)
    assert materialized == receipt
    assert {path.name for path in destination.iterdir()} == set(first)
    with pytest.raises(RecipeWorkspaceError, match="already exists"):
        materialize_offline_recipe_workspace(source, destination)


def test_empty_workspace_has_explicit_nonpreview_state(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    files = build_offline_recipe_workspace(source)
    receipt = json.loads(files["workspace_receipt.json"])
    assert receipt["catalog_status"] == "empty"
    assert b"No verified previews available" in files["previews.html"]
    assert b"No recipes found" in files["history.html"]
