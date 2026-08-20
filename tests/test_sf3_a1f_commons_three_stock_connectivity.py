from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.commons_three_stock_connectivity import (
    CONTRACT_SCHEMA,
    INPUT_SCHEMA,
    CommonsThreeStockConnectivityError,
    evaluate,
)

STOCKS = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]


def _fixture(tmp_path: Path, *, authors: int = 2) -> Path:
    blocks = []
    for stock_index, stock in enumerate(STOCKS):
        candidates = []
        for index in range(authors):
            candidates.append(
                {
                    "page_id": 100 * stock_index + index,
                    "author_raw_html": (
                        f'<a href="//commons.wikimedia.org/wiki/User:Author_{index}">'
                        f"Author {index}</a>"
                    ),
                }
            )
        blocks.append({"film_stock_id": stock, "eligible_candidates": candidates})
    source = {
        "schema": INPUT_SCHEMA,
        "stable_evidence_id": "a" * 64,
        "stock_results": blocks,
        "image_payload_downloads": 0,
        "pixel_decodes": 0,
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")
    contract = {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": "test",
        "input_report": "input.json",
        "input_report_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_stable_evidence_id": "a" * 64,
        "stock_ids": STOCKS,
        "identity_alias_groups": [],
        "metadata_gates": {
            "minimum_shared_authors_per_pair": 2,
            "minimum_shared_authors_all_three_stocks": 2,
            "minimum_rows_per_stock_in_connected_pool": 2,
        },
        "decision_if_pass": "pass",
        "decision_if_fail": "fail",
        "claim_ceiling": "metadata only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path


def test_connected_triangle_passes_without_pixels(tmp_path: Path) -> None:
    report = evaluate(_fixture(tmp_path), root=tmp_path)
    assert report["automatic_pass"] is True
    assert report["triple_shared_author_count"] == 2
    assert report["network_requests"] == report["pixel_decodes"] == 0
    assert report["operator_fits"] == 0


def test_alias_group_collapses_cross_platform_identity(tmp_path: Path) -> None:
    contract_path = _fixture(tmp_path)
    source_path = tmp_path / "input.json"
    source = json.loads(source_path.read_text())
    source["stock_results"][0]["eligible_candidates"][0]["author_raw_html"] = (
        '<a href="https://500px.com/person">Person</a>'
    )
    source["stock_results"][1]["eligible_candidates"][0]["author_raw_html"] = (
        '<a href="https://www.flickr.com/people/123@N00">Person</a>'
    )
    source["stock_results"][2]["eligible_candidates"][0]["author_raw_html"] = (
        '<a href="https://500px.com/person">Person</a>'
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")
    contract = json.loads(contract_path.read_text())
    contract["input_report_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    contract["identity_alias_groups"] = [["500px:person", "flickr:123@n00"]]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    report = evaluate(contract_path, root=tmp_path)
    assert report["automatic_pass"] is True
    assert report["triple_shared_author_count"] == 2


def test_insufficient_triangle_fails_before_pixels(tmp_path: Path) -> None:
    contract_path = _fixture(tmp_path, authors=1)
    report = evaluate(contract_path, root=tmp_path)
    assert report["automatic_pass"] is False
    assert report["decision"] == "fail"
    assert report["image_payload_download_allowed"] is False


def test_input_hash_drift_fails_closed(tmp_path: Path) -> None:
    contract_path = _fixture(tmp_path)
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CommonsThreeStockConnectivityError, match="hash drifted"):
        evaluate(contract_path, root=tmp_path)
