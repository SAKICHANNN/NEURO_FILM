from __future__ import annotations

import copy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from src.inference import (
    RecipeHistoryHtmlError,
    build_render_recipe_history,
    render_recipe_history_html,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke"


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.ids: set[str] = set()
        self.labels: list[dict[str, str | None]] = []
        self.external_sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "label":
            self.labels.append(values)
        for key in ("src", "href"):
            value = values.get(key)
            if value:
                self.external_sources.append(value)


def _catalog() -> dict:
    return build_render_recipe_history(HISTORY_ROOT)


def test_history_page_is_deterministic_accessible_and_offline() -> None:
    first = render_recipe_history_html(_catalog())
    second = render_recipe_history_html(_catalog())
    assert first == second
    text = first.decode("utf-8")
    parser = _StructureParser()
    parser.feed(text)
    assert text.startswith("<!doctype html>")
    assert "main" in parser.tags
    assert "h1" in parser.tags
    assert "recipe-search" in parser.ids
    assert "result-status" in parser.ids
    assert any(label.get("for") == "recipe-search" for label in parser.labels)
    assert parser.external_sources == []
    assert 'aria-live="polite"' in text
    assert "prefers-reduced-motion:reduce" in text
    assert "word-break:break-all" in text
    assert "width:min(calc(100vw - 20px),1120px)" in text
    assert "overflow-x:clip" in text
    assert "connect-src &#x27;none&#x27;" not in text
    assert "connect-src 'none'" in text
    assert text.count('<article class="recipe-card"') == 3
    assert "film-inspired" in text
    assert "look-approximation" in text


def test_catalog_text_is_escaped_and_never_embedded_as_script() -> None:
    catalog = _catalog()
    malicious = copy.deepcopy(catalog)
    malicious["entries"][0]["input_path"] = '</code><script id="owned">x</script>'
    rendered = render_recipe_history_html(malicious).decode("utf-8")
    assert '<script id="owned">' not in rendered
    assert "&lt;/code&gt;&lt;script" in rendered
    assert "inline_catalog_json" not in rendered


def test_empty_and_invalid_states_render_explicitly(tmp_path: Path) -> None:
    empty = render_recipe_history_html(build_render_recipe_history(tmp_path)).decode()
    assert "No recipes found" in empty
    assert empty.count('<article class="recipe-card"') == 0
    (tmp_path / "bad.recipe.json").write_text("{", encoding="utf-8")
    invalid = render_recipe_history_html(build_render_recipe_history(tmp_path)).decode()
    assert "fail-closed" in invalid
    assert "recipe_invalid_json" in invalid
    assert 'role="alert"' in invalid


def test_catalog_shape_drift_fails_closed() -> None:
    catalog = _catalog()
    catalog["surprise"] = True
    with pytest.raises(RecipeHistoryHtmlError, match="catalog keys differ"):
        render_recipe_history_html(catalog)


def test_catalog_count_or_status_drift_fails_closed() -> None:
    count_drift = _catalog()
    count_drift["counts"]["valid"] = 2
    with pytest.raises(RecipeHistoryHtmlError, match="valid/invalid counts differ"):
        render_recipe_history_html(count_drift)
    status_drift = _catalog()
    status_drift["status"] = "partial"
    with pytest.raises(RecipeHistoryHtmlError, match="aggregate status differs"):
        render_recipe_history_html(status_drift)
