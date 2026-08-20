from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

from src.real_film.yfcc_three_stock_rights import (
    CONTRACT_SCHEMA,
    YfccThreeStockRightsError,
    load_candidate_matrix,
    run,
)


class _Session(requests.Session):
    def __init__(self, *, licensed: bool = True) -> None:
        super().__init__()
        self.licensed = licensed

    def get(self, url: str, **_: object) -> _Response:
        body = b'https://creativecommons.org/licenses/by/2.0/' if self.licensed else b"no licence"
        return _Response(url, body)


class _Response:
    status_code = 200
    encoding = "utf-8"

    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.headers = {"Content-Type": "text/html; charset=utf-8"}
        self._body = body

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self._body]

    def close(self) -> None:
        return None


def _fixture(tmp_path: Path) -> Path:
    stocks = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    matrix = {}
    for uid_index in range(2):
        uid = f"uid-{uid_index}"
        matrix[uid] = {}
        for stock_index, stock in enumerate(stocks):
            photoid = uid_index * 10 + stock_index
            matrix[uid][stock] = [{
                "photoid": photoid,
                "uid": uid,
                "film_stock_id": stock,
                "pageurl": f"https://www.flickr.com/photos/u/{photoid}/",
                "downloadurl": f"https://live.staticflickr.com/a/{photoid}.jpg",
                "licenseurl": "http://creativecommons.org/licenses/by/2.0/",
            }]
    source = {
        "stable_evidence_id": "stable",
        "decision": "open",
        "image_payload_download_allowed": False,
        "candidate_matrix": matrix,
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    contract = {
        "schema": CONTRACT_SCHEMA,
        "experiment_id": "test",
        "input_report": "source.json",
        "input_report_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "input_stable_evidence_id": "stable",
        "required_input_decision": "open",
        "stock_ids": stocks,
        "expected_eligible_uids": ["uid-0", "uid-1"],
        "expected_page_candidates": 6,
        "selection": {"maximum_page_requests": 6},
        "live_rights": {
            "required_page_regex": "creativecommons\\.org/licenses/by/2\\.0/",
            "required_content_type_prefix": "text/html",
            "maximum_html_bytes": 1024,
        },
        "request_limits": {
            "timeout_seconds": 1,
            "request_retries": 1,
            "retry_backoff_seconds": 0,
            "request_interval_seconds": 0,
        },
        "user_agent": "test",
        "minimum_usable_three_stock_uids": 2,
        "decision_if_pass": "pass",
        "decision_if_fail": "fail",
        "claim_ceiling": "page only",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return contract_path


def test_all_three_stock_arms_pass(tmp_path: Path) -> None:
    report = run(_fixture(tmp_path), root=tmp_path, session=_Session(), sleep_fn=lambda _: None)
    assert report["automatic_pass"] is True
    assert report["usable_three_stock_uid_count"] == 2
    assert report["page_requests"] == 6
    assert report["image_payload_download_allowed"] is False


def test_rights_failure_closes_before_pixels(tmp_path: Path) -> None:
    report = run(
        _fixture(tmp_path), root=tmp_path, session=_Session(licensed=False), sleep_fn=lambda _: None
    )
    assert report["automatic_pass"] is False
    assert report["decision"] == "fail"
    assert report["operator_fitting_allowed"] is False


def test_candidate_count_drift_is_invalid(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    payload = json.loads(contract.read_text())
    payload["expected_page_candidates"] = 7
    contract.write_text(json.dumps(payload))
    with pytest.raises(YfccThreeStockRightsError, match="page count drifted"):
        load_candidate_matrix(contract, root=tmp_path)
