from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.jstage_film_granularity_source import (
    JstageFilmGranularitySourceError,
    analyze_article_html,
    analyze_machine_text,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4al_jstage_film_granularity_source_v1.json"


def test_contract_mutation_fails_closed(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["acquisition"]["redistribution_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(JstageFilmGranularitySourceError, match="contract drift"):
        load_contract(path)


def test_article_identity_and_rights_labels_are_independent() -> None:
    source = json.loads(CONTRACT.read_text(encoding="utf-8"))["source"]
    raw = (
        "Granularity of Photographic Film Masao Takano 10.3169/itej1954.23.13 "
        "1969 Volume 23 Issue 1 Pages 13-23 JOURNAL FREE ACCESS "
        "The Institute of Image Information and Television Engineers"
    ).encode()
    assert all(analyze_article_html(raw, source).values())
    without_rights = raw.replace(b"JOURNAL FREE ACCESS", b"SUBSCRIPTION")
    assert not analyze_article_html(without_rights, source)["free_access"]


def test_machine_text_uses_exact_frozen_terms() -> None:
    result = analyze_machine_text("写真の粒状性と濃度、自己相関。", ["粒状性", "写真", "濃度", "自己相関", "ウィーナー"])
    assert not all(result["term_results"].values())
    assert result["japanese_characters"] > 0
