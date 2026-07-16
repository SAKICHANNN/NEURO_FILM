from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.yfcc_shared_author_rights import (
    YfccSharedAuthorRightsError,
    run_shared_author_rights_preflight,
    select_shared_author_candidates,
)


def _config() -> dict:
    return {
        "preflight_id": "test",
        "passing_gate_id": "pair",
        "left_stock_id": "left",
        "right_stock_id": "right",
        "expected_support": {"shared_author_uids": 2, "exclusive_candidate_rows": 5},
        "selection": {"maximum_candidates_per_stock_per_author": 1, "maximum_page_requests": 4},
        "process_contamination_exclusions": ["hdr"],
        "live_rights": {
            "required_snapshot_license_url": "http://creativecommons.org/licenses/by/2.0/",
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
        "decision_gates": {"minimum_usable_shared_authors": 1},
        "input_report_sha256": "a" * 64,
        "input_decision_sha256": "b" * 64,
        "claim_ceiling": "test",
    }


def _row(photoid: int, uid: str, stock: str, title: str = "film") -> dict:
    return {
        "photoid": photoid,
        "uid": uid,
        "film_stock_id": stock,
        "title": title,
        "description": "",
        "usertags": "",
        "licenseurl": "http://creativecommons.org/licenses/by/2.0/",
        "pageurl": f"https://www.flickr.com/photos/{uid}/{photoid}",
        "downloadurl": "https://example.invalid/image.jpg",
    }


def _report() -> dict:
    rows = [
        _row(1, "u1", "left"),
        _row(2, "u1", "right"),
        _row(3, "u2", "left", "HDR"),
        _row(4, "u2", "left"),
        _row(5, "u2", "right"),
    ]
    return {
        "matches": rows,
        "shared_author_results": {
            "pair": {"metadata_gate_passed": True, "shared_uids": ["u1", "u2"]}
        },
    }


def _decision() -> dict:
    return {
        "decision": "open_bounded_live_rights_preflight",
        "passing_gate_ids": ["pair"],
        "pixel_download_allowed": False,
        "operator_fitting_allowed": False,
    }


def test_selection_is_bounded_and_applies_prospective_exclusion() -> None:
    matrix = select_shared_author_candidates(_report(), _decision(), _config())
    assert matrix["u1"]["left"][0]["photoid"] == 1
    assert matrix["u2"]["left"][0]["photoid"] == 4
    assert sum(len(rows) for stocks in matrix.values() for rows in stocks.values()) == 4
    bad = _decision()
    bad["pixel_download_allowed"] = True
    with pytest.raises(YfccSharedAuthorRightsError, match="no-pixel"):
        select_shared_author_candidates(_report(), bad, _config())


class _Response:
    def __init__(self, url: str, body: bytes, *, status: int = 200, content_type: str = "text/html") -> None:
        self.url = url
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index:index + chunk_size]

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, bodies: dict[int, bytes]) -> None:
        self.headers: dict[str, str] = {}
        self.bodies = bodies
        self.requested: list[str] = []
        self.responses: list[_Response] = []

    def get(self, url: str, **kwargs) -> _Response:
        del kwargs
        self.requested.append(url)
        photoid = int(url.rstrip("/").split("/")[-1])
        response = _Response(url, self.bodies.get(photoid, b"no licence"))
        self.responses.append(response)
        return response


def test_preflight_passes_only_authors_with_both_live_stock_pages() -> None:
    matrix = select_shared_author_candidates(_report(), _decision(), _config())
    licence = b'<a href="https://creativecommons.org/licenses/by/2.0/">CC BY</a>'
    session = _Session({1: licence, 2: licence, 4: licence})
    result = run_shared_author_rights_preflight(matrix, _config(), session=session, sleep_fn=lambda _: None)
    assert result["decision"] == "pass_live_rights_feasibility"
    assert result["usable_shared_author_uids"] == ["u1"]
    assert result["image_payload_download_allowed"] is False
    assert all("example.invalid" not in url for url in session.requested)
    assert all(response.closed for response in session.responses)


def test_config_fixture_is_valid_json() -> None:
    path = Path("configs/real_film_yfcc_shared_author_rights_v1.json")
    assert json.loads(path.read_text(encoding="utf-8"))["image_payload_download_allowed"] is False
