from __future__ import annotations

import json
from pathlib import Path

from src.real_film.apollo_metadata_feasibility import (
    build_sample,
    decide_metadata_feasibility,
    parse_photo_page,
    run_metadata_feasibility_audit,
)


def _config() -> dict:
    return {
        "audit_id": "test",
        "source": {
            "mission": "AS07",
            "photo_page_template": "https://eol.jsc.nasa.gov/SearchPhotos/photo.pl?mission={mission}&roll={roll}&frame={frame}",
        },
        "magazines": [
            {"physical_magazine": "A", "photo_roll": "1", "frame_start": 1, "frame_end": 3, "film_stock_id": "left", "film_code": "SO368", "filter": "none"},
            {"physical_magazine": "B", "photo_roll": "2", "frame_start": 4, "frame_end": 6, "film_stock_id": "left", "film_code": "SO368", "filter": "none"},
            {"physical_magazine": "C", "photo_roll": "3", "frame_start": 7, "frame_end": 9, "film_stock_id": "right", "film_code": "SO121", "filter": "none"},
            {"physical_magazine": "D", "photo_roll": "4", "frame_start": 10, "frame_end": 12, "film_stock_id": "right", "film_code": "SO121", "filter": "none"},
        ],
        "selection": {"frames_per_magazine": 3, "expected_page_requests": 12, "maximum_page_requests": 12, "image_urls_requested": False, "html_bodies_retained": False},
        "page_contract": {"required_status": 200, "required_content_type_prefix": "text/html", "maximum_html_bytes": 4096},
        "content_tags": {"ocean_water": ["ocean"], "cloud_weather": ["cloud"]},
        "request_limits": {"timeout_seconds": 1, "request_retries": 1, "retry_backoff_seconds": 0, "request_interval_seconds": 0, "user_agent": "test"},
        "decision_gates": {
            "minimum_valid_pages_per_stock": 4,
            "minimum_valid_pages_per_magazine": 2,
            "minimum_independent_magazines_per_stock": 2,
            "minimum_shared_content_tags": 2,
            "minimum_rows_per_stock_per_shared_tag": 2,
            "minimum_magazines_per_stock_per_shared_tag": 2,
            "minimum_filter_free_rows_per_stock": 2,
            "minimum_distinct_reported_exposure_states": 2,
        },
        "allowed_decisions": ["metadata_connectivity_candidate", "insufficient_connectivity", "stock_filter_confounded", "content_confounded", "source_unavailable", "metadata_mismatch"],
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": "test",
    }


def _html(photo_id: str, film: str, exposure: str = "Normal") -> bytes:
    return f"""
    <html><body><div>NASA Photo ID {photo_id}</div>
    <table>
      <tr><td><b>Country or Geographic Name:</b></td><td>ATLANTIC OCEAN</td></tr>
      <tr><td><b>Features:</b></td><td>CLOUD, OCEAN</td></tr>
      <tr><td><b>Format:</b></td><td>{film}: Kodak Ektachrome</td></tr>
      <tr><td><b>Film Exposure:</b></td><td>{exposure}</td></tr>
    </table>
    <div><em><b>Image Caption</b></em>: Ocean and cloud scene.</div>
    <a href="/DatabaseImages/ISD/highres/{photo_id}.JPG">image</a>
    </body></html>
    """.encode()


class _Response:
    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.status_code = 200
        self.headers = {"Content-Type": "text/html; charset=utf-8"}
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index:index + chunk_size]

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, mismatch: bool = False) -> None:
        self.headers: dict[str, str] = {}
        self.requested: list[str] = []
        self.responses: list[_Response] = []
        self.mismatch = mismatch

    def get(self, url: str, **kwargs) -> _Response:
        del kwargs
        self.requested.append(url)
        query = dict(item.split("=", 1) for item in url.split("?", 1)[1].split("&"))
        photo_id = f"{query['mission']}-{query['roll']}-{query['frame']}"
        if self.mismatch and query["frame"] == "1":
            photo_id = "AS07-1-999"
        film = "SO368" if query["roll"] in {"1", "2"} else "SO121"
        exposure = "Normal" if int(query["frame"]) % 2 else "Under Exposed"
        response = _Response(url, _html(photo_id, film, exposure))
        self.responses.append(response)
        return response


def test_frozen_config_builds_exact_unique_sample() -> None:
    config = json.loads(Path("configs/real_film_apollo7_metadata_feasibility_v1.json").read_text(encoding="utf-8"))
    sample = build_sample(config)
    assert len(sample) == 63
    assert len({row["page_url"] for row in sample}) == 63
    assert {row["physical_magazine"] for row in sample} == {"M", "N", "Q", "O", "S", "R", "P"}


def test_parser_records_offered_images_without_requesting_them() -> None:
    config = _config()
    row = build_sample(config)[0]
    parsed = parse_photo_page(_html(row["nasa_photo_id"], "SO368"), row, row["page_url"], config)
    assert parsed["photo_id_match"] is True
    assert parsed["film_code_match"] is True
    assert parsed["content_tags"] == ["cloud_weather", "ocean_water"]
    assert parsed["offered_image_metadata_only"][0]["url"].endswith(".JPG")


def test_audit_passes_connectivity_without_image_requests() -> None:
    session = _Session()
    report, decision = run_metadata_feasibility_audit(_config(), session=session, sleep_fn=lambda _: None)
    assert report["page_requests"] == 12
    assert decision["decision"] == "metadata_connectivity_candidate"
    assert decision["image_payload_download_allowed"] is False
    assert all("photo.pl?" in url for url in session.requested)
    assert all(response.closed for response in session.responses)


def test_metadata_mismatch_fails_closed() -> None:
    _, decision = run_metadata_feasibility_audit(_config(), session=_Session(mismatch=True), sleep_fn=lambda _: None)
    assert decision["decision"] == "metadata_mismatch"
    assert decision["metadata_mismatch_photo_ids"] == ["AS07-1-1"]


def test_filter_confounding_has_explicit_branch() -> None:
    config = _config()
    records = []
    for row in build_sample(config):
        records.append({**row, "valid_metadata_page": True, "page_status": 200, "photo_id_match": True, "film_code_match": True, "film_exposure": "Normal" if row["frame"] % 2 else "Under Exposed", "content_tags": ["ocean_water", "cloud_weather"]})
    for row in records:
        if row["film_stock_id"] == "right":
            row["filter"] = "wratten_2a"
    decision = decide_metadata_feasibility(records, config)
    assert decision["decision"] == "stock_filter_confounded"
