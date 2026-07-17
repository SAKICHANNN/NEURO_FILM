from __future__ import annotations

import json
from pathlib import Path

from src.real_film.nasa_sts_stock_snapshot import parse_result_table, run_snapshot


def _config() -> dict:
    return {
        "snapshot_id": "test",
        "source": {
            "query_endpoint": "https://eol.jsc.nasa.gov/SearchPhotos/Technical.pl",
            "result_base_url": "https://eol.jsc.nasa.gov/SearchPhotos/",
            "target_mission": "STS098",
        },
        "stocks": [
            {"film_code": "A", "film_stock_id": "stock_a", "minimum_target_rows": 2, "minimum_target_rolls": 2},
            {"film_code": "B", "film_stock_id": "stock_b", "minimum_target_rows": 2, "minimum_target_rolls": 2},
        ],
        "query": {
            "form_fields": {"mode": "db"},
            "one_exact_film_code_per_query": True,
            "maximum_post_requests": 2,
            "maximum_result_get_requests": 2,
            "maximum_total_requests": 4,
            "photo_page_requests_allowed": False,
            "image_payload_requests_allowed": False,
            "raw_html_retained": False,
        },
        "result_contract": {
            "required_status": 200,
            "required_content_type_prefix": "text/html",
            "maximum_result_bytes": 65536,
            "required_headers": ["Photo ID", "Photo Date", "Lat", "Lon", "Geographic Name", "Features Identified Manually", "Features Identified by Machine Learning", "Focal Length (mm)", "Record Type"],
            "photo_id_regex": "^[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+$",
            "target_mission_prefix": "STS098-",
            "cross_stock_photo_id_overlap_must_be_zero": True,
        },
        "request_limits": {"timeout_seconds": 1, "request_retries": 1, "retry_backoff_seconds": 0, "request_interval_seconds": 0, "user_agent": "test"},
        "allowed_decisions": ["snapshot_complete_open_connectivity_design", "insufficient_stock_roll_support", "cross_stock_id_overlap", "query_contract_mismatch", "source_unavailable"],
        "image_payload_download_allowed": False,
        "photo_page_access_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": "test",
    }


def _table(ids: list[str]) -> bytes:
    headers = _config()["result_contract"]["required_headers"]
    rows = ["<tr>" + "".join(f"<th>{value}</th>" for value in headers) + "</tr>"]
    for photo_id in ids:
        values = [photo_id, "200102__", "1.0", "2.0", "EARTH", "OCEAN", "", "100", "Cataloged With Center Point"]
        rows.append("<tr>" + "".join(f"<td><a>{value}</a></td>" for value in values) + "</tr>")
    return ("<html><table>" + "".join(rows) + "</table></html>").encode()


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
    def __init__(self, overlap: bool = False) -> None:
        self.headers: dict[str, str] = {}
        self.requests: list[tuple[str, str, dict | None]] = []
        self.responses: list[_Response] = []
        self.codes: dict[str, str] = {}
        self.overlap = overlap

    def post(self, url: str, data: dict, **kwargs) -> _Response:
        del kwargs
        self.requests.append(("POST", url, data))
        token = "1" if data["film"] == "A" else "2"
        self.codes[token] = data["film"]
        response = _Response(url, f'<script>window.location.href="ShowQueryResults-TextTable.pl?results={token}"</script>'.encode())
        self.responses.append(response)
        return response

    def get(self, url: str, **kwargs) -> _Response:
        del kwargs
        self.requests.append(("GET", url, None))
        token = url.rsplit("=", 1)[-1]
        code = self.codes[token]
        if code == "A":
            ids = ["STS098-1-1", "STS098-2-2", "STS099-9-9"]
        else:
            ids = ["STS098-3-3", "STS098-4-4"]
            if self.overlap:
                ids[0] = "STS098-1-1"
        response = _Response(url, _table(ids))
        self.responses.append(response)
        return response


def test_tracked_config_is_fail_closed() -> None:
    config = json.loads(Path("configs/real_film_nasa_sts098_stock_snapshot_v1.json").read_text(encoding="utf-8"))
    assert config["query"]["maximum_total_requests"] == 6
    assert config["photo_page_access_allowed"] is False
    assert config["image_payload_download_allowed"] is False


def test_result_parser_finds_only_photo_rows() -> None:
    headers, rows = parse_result_table(_table(["STS098-1-1", "STS098-2-2"]), _config())
    assert headers[0] == "Photo ID"
    assert [row["Photo ID"] for row in rows] == ["STS098-1-1", "STS098-2-2"]


def test_snapshot_passes_without_photo_or_image_requests() -> None:
    session = _Session()
    report, decision = run_snapshot(_config(), session=session, sleep_fn=lambda _: None)
    assert decision["decision"] == "snapshot_complete_open_connectivity_design"
    assert report["requests_made"] == 4
    assert len(report["records"]) == 4
    assert all("photo.pl" not in url.casefold() for _, url, _ in session.requests)
    assert all("DatabaseImages" not in url for _, url, _ in session.requests)
    assert all(response.closed for response in session.responses)


def test_cross_stock_id_overlap_fails_closed() -> None:
    _, decision = run_snapshot(_config(), session=_Session(overlap=True), sleep_fn=lambda _: None)
    assert decision["decision"] == "cross_stock_id_overlap"
    assert decision["cross_stock_photo_id_overlap"] == ["STS098-1-1"]
