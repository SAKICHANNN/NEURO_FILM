from __future__ import annotations

import json
from pathlib import Path

from src.real_film.nasa_sts_stock_snapshot import (
    decide_cross_mission_aggregates,
    run_cross_mission_census,
)


def _config() -> dict:
    config = json.loads(
        Path("configs/real_film_nasa_cross_mission_connectivity_v1.json").read_text(encoding="utf-8")
    )
    config["request_limits"].update(
        {"timeout_seconds": 1, "request_retries": 1, "retry_backoff_seconds": 0, "request_interval_seconds": 0}
    )
    return config


def _ids(mission: str, rolls: str, rows_per_roll: int) -> list[str]:
    return [
        f"{mission}-{roll}-{frame:03d}"
        for roll in rolls
        for frame in range(1, rows_per_roll + 1)
    ]


def _table(ids: list[str], config: dict) -> bytes:
    headers = config["result_contract"]["required_headers"]
    rows = ["<tr>" + "".join(f"<th>{value}</th>" for value in headers) + "</tr>"]
    for photo_id in ids:
        values = [
            photo_id,
            "200101__",
            "1.0",
            "2.0",
            "EARTH",
            "OCEAN",
            "",
            "100",
            "Cataloged With Center Point",
        ]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
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
            yield self._body[index : index + chunk_size]

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, rows_by_code: dict[str, list[str]]) -> None:
        self.headers: dict[str, str] = {}
        self.requests: list[tuple[str, str, dict | None]] = []
        self.responses: list[_Response] = []
        self.rows_by_code = rows_by_code
        self.tokens: dict[str, str] = {}

    def post(self, url: str, data: dict, **kwargs) -> _Response:
        del kwargs
        self.requests.append(("POST", url, data))
        token = str(len(self.tokens) + 1)
        self.tokens[token] = data["film"]
        body = f'<script>window.location.href="ShowQueryResults-TextTable.pl?results={token}"</script>'.encode()
        response = _Response(url, body)
        self.responses.append(response)
        return response

    def get(self, url: str, **kwargs) -> _Response:
        del kwargs
        self.requests.append(("GET", url, None))
        code = self.tokens[url.rsplit("=", 1)[-1]]
        response = _Response(url, _table(self.rows_by_code[code], _config()))
        self.responses.append(response)
        return response


def _eligible_rows() -> dict[str, list[str]]:
    return {
        "VELVI": _ids("STS777", "ABCD", 8) + _ids("STS111", "A", 3),
        "5775": _ids("STS777", "EFGH", 8),
        "5776": _ids("STS777", "I", 2),
    }


def test_tracked_contract_is_bounded_and_fail_closed() -> None:
    config = _config()
    assert config["query"]["maximum_total_requests"] == 6
    assert config["query"]["frame_rows_retained"] is False
    assert config["query"]["raw_html_retained"] is False
    assert config["photo_page_access_allowed"] is False
    assert config["image_payload_download_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
    assert config["latent_mode_study_allowed"] is False


def test_census_passes_and_is_replayable_without_frame_rows() -> None:
    config = _config()
    session = _Session(_eligible_rows())
    report, decision = run_cross_mission_census(config, session=session, sleep_fn=lambda _: None)
    assert report["requests_made"] == 6
    assert decision["decision"] == "candidate_mission_found_open_metadata_connectivity"
    assert decision["selected_mission"] == "STS777"
    assert report["frame_rows_retained"] is False
    assert "records" not in report
    serialized = json.dumps(report)
    assert "STS777-A-001" not in serialized
    assert '"frame"' not in serialized
    replay = decide_cross_mission_aggregates(
        report["query_evidence"],
        report["mission_aggregates"],
        decision["cross_stock_photo_id_overlap"],
        config,
    )
    assert replay == decision
    assert all("photo.pl" not in url.casefold() for _, url, _ in session.requests)
    assert all("databaseimages" not in url.casefold() for _, url, _ in session.requests)
    assert all(response.closed for response in session.responses)


def test_census_closes_when_primary_support_is_insufficient() -> None:
    rows = _eligible_rows()
    rows["5775"] = _ids("STS777", "E", 8)
    _, decision = run_cross_mission_census(_config(), session=_Session(rows), sleep_fn=lambda _: None)
    assert decision["decision"] == "no_candidate_mission"
    assert decision["selected_mission"] is None


def test_cross_stock_photo_id_overlap_fails_closed() -> None:
    rows = _eligible_rows()
    rows["5775"][0] = rows["VELVI"][0]
    _, decision = run_cross_mission_census(_config(), session=_Session(rows), sleep_fn=lambda _: None)
    assert decision["decision"] == "cross_stock_id_overlap"
    assert decision["cross_stock_photo_id_overlap"] == [rows["VELVI"][0]]
