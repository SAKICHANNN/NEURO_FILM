from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.yfcc_three_stock_triangle import (
    CONTRACT_SCHEMA,
    YfccThreeStockTriangleError,
    evaluate,
)


def _write_fixture(tmp_path: Path) -> Path:
    stocks = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    rows = []
    for uid_index in range(2):
        for stock_index, stock in enumerate(stocks):
            rows.append(
                {
                    "photoid": 100 * uid_index + stock_index,
                    "uid": f"uid-{uid_index}",
                    "film_stock_id": stock,
                    "title": "ordinary photograph",
                    "description": "",
                    "usertags": "film",
                    "pageurl": f"https://www.flickr.com/photos/u/{100 * uid_index + stock_index}/",
                    "downloadurl": f"https://live.staticflickr.com/a/{uid_index}-{stock_index}.jpg",
                    "licenseurl": "http://creativecommons.org/licenses/by/2.0/",
                }
            )
    report = {
        "candidate_results": {
            stock: {"rows": 2, "author_uids": 2} for stock in stocks
        },
        "matches": rows,
        "ambiguous_multi_stock_rows": [],
        "image_payloads_downloaded_or_decoded": False,
    }
    report_path = tmp_path / "input.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    contract = {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": "test",
        "input_report": "input.json",
        "input_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "stock_ids": stocks,
        "expected_stock_support": {
            stock: {"rows": 2, "author_uids": 2} for stock in stocks
        },
        "expected_pair_shared_uids": {
            "fujifilm_velvia_50__kodak_portra_400": 2,
            "fujifilm_velvia_50__kodak_ektar_100": 2,
            "kodak_portra_400__kodak_ektar_100": 2,
        },
        "metadata_gates": {
            "minimum_rows_per_stock": 2,
            "minimum_author_uids_per_stock": 2,
            "minimum_shared_uids_per_pair": 2,
            "minimum_triple_shared_uids": 2,
            "minimum_eligible_triple_uids": 2,
            "maximum_candidates_per_stock_per_uid": 1,
            "maximum_page_candidates": 6,
        },
        "required_snapshot_license_url": "http://creativecommons.org/licenses/by/2.0/",
        "process_contamination_exclusions": ["multiple[ -]?exposure"],
        "decision_if_pass": "pass",
        "decision_if_fail": "fail",
        "claim_ceiling": "metadata only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path


def test_complete_triangle_passes_without_network_or_pixels(tmp_path: Path) -> None:
    report = evaluate(_write_fixture(tmp_path), root=tmp_path)
    assert report["automatic_pass"] is True
    assert report["eligible_triple_uid_count"] == 2
    assert report["page_candidate_count"] == 6
    assert report["network_requests"] == report["pixel_decodes"] == 0
    assert report["operator_fits"] == 0


def test_contaminated_arm_fails_eligibility(tmp_path: Path) -> None:
    contract = _write_fixture(tmp_path)
    input_path = tmp_path / "input.json"
    payload = json.loads(input_path.read_text())
    payload["matches"][0]["title"] = "multiple exposure"
    input_path.write_text(json.dumps(payload))
    config = json.loads(contract.read_text())
    config["input_report_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    contract.write_text(json.dumps(config))
    report = evaluate(contract, root=tmp_path)
    assert report["automatic_pass"] is False
    assert report["gates"]["eligible_triple_connectivity"] is False


def test_input_hash_drift_fails_closed(tmp_path: Path) -> None:
    contract = _write_fixture(tmp_path)
    (tmp_path / "input.json").write_text("{}")
    with pytest.raises(YfccThreeStockTriangleError, match="hash drifted"):
        evaluate(contract, root=tmp_path)


def test_ambiguous_row_cannot_enter_stock_matches(tmp_path: Path) -> None:
    contract = _write_fixture(tmp_path)
    input_path = tmp_path / "input.json"
    payload = json.loads(input_path.read_text())
    payload["ambiguous_multi_stock_rows"] = [{"photoid": payload["matches"][0]["photoid"]}]
    input_path.write_text(json.dumps(payload))
    config = json.loads(contract.read_text())
    config["input_report_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    contract.write_text(json.dumps(config))
    with pytest.raises(YfccThreeStockTriangleError, match="ambiguous"):
        evaluate(contract, root=tmp_path)
