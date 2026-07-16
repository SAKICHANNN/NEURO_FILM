from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import requests

from src.real_film.yfcc_full_index import (
    YfccFullIndexError,
    audit_candidate_rows,
    download_full_index,
    scan_full_index,
    validate_source_headers,
)


def _config() -> dict:
    return {
        "dataset_id": "test", "claim_ceiling": "test",
        "source": {"expected_bytes": 10, "expected_etag": "etag", "expected_last_modified": "date"},
        "rights_filter": {"allowed_license_urls": ["cc-by"]},
        "stock_patterns": [
            {"film_stock_id": "ektar", "exact_regex": "(^|[^a-z0-9])ektar 100([^a-z0-9]|$)"},
            {"film_stock_id": "velvia", "exact_regex": "(^|[^a-z0-9])velvia 50([^a-z0-9]|$)"},
        ],
        "shared_author_gates": [
            {"gate_id": "pair", "left_stock_id": "ektar", "right_stock_id": "velvia", "minimum_rows_each_stock": 1, "minimum_uids_each_stock": 1, "minimum_shared_uids": 1}
        ],
        "scan": {
            "required_columns": ["photoid", "uid", "unickname", "title", "description", "usertags", "pageurl", "downloadurl", "licensename", "licenseurl", "serverid", "farmid", "secret", "secretoriginal", "ext", "marker"],
            "broad_prefilter_terms": ["ektar", "velvia"],
        },
    }


def test_source_headers_fail_closed_on_etag_drift() -> None:
    headers = {"Content-Length": "10", "ETag": '"etag"', "Last-Modified": "date"}
    assert validate_source_headers(headers, _config())["etag"] == "etag"
    headers["ETag"] = '"other"'
    with pytest.raises(YfccFullIndexError, match="ETag"):
        validate_source_headers(headers, _config())


def test_exact_rows_and_shared_uid_gate() -> None:
    base = {"unickname": "", "description": "", "pageurl": "p", "downloadurl": "d", "licensename": "by", "licenseurl": "cc-by", "serverid": 1, "farmid": 1, "secret": "s", "secretoriginal": "o", "ext": "jpg", "marker": 0}
    rows = [
        {**base, "photoid": 1, "uid": "shared", "title": "Kodak Ektar-100", "usertags": "film"},
        {**base, "photoid": 2, "uid": "shared", "title": "", "usertags": "Fuji+Velvia+50"},
    ]
    report = audit_candidate_rows(rows, _config())
    assert report["shared_author_results"]["pair"]["metadata_gate_passed"] is True


def test_sqlite_scan_applies_licence_and_photo_filter(tmp_path: Path) -> None:
    path = tmp_path / "tiny.sqlite"
    columns = _config()["scan"]["required_columns"]
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE yfcc100m_dataset (photoid INTEGER, uid TEXT, unickname TEXT, title TEXT, description TEXT, usertags TEXT, pageurl TEXT, downloadurl TEXT, licensename TEXT, licenseurl TEXT, serverid INTEGER, farmid INTEGER, secret TEXT, secretoriginal TEXT, ext TEXT, marker INTEGER)")
        rows = [
            (1, "shared", "", "Ektar 100", "", "", "p", "d", "by", "cc-by", 1, 1, "s", "o", "jpg", 0),
            (2, "shared", "", "Velvia 50", "", "", "p", "d", "by", "cc-by", 1, 1, "s", "o", "jpg", 0),
            (3, "bad", "", "Velvia 50", "", "", "p", "d", "nc", "other", 1, 1, "s", "o", "jpg", 0),
        ]
        connection.executemany(f"INSERT INTO yfcc100m_dataset ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", rows)
    report = scan_full_index(path, _config())
    assert len(report["matches"]) == 2
    assert report["any_shared_author_gate_passed"] is True


class _FakeResponse:
    def __init__(self, status_code: int, *, headers: dict[str, str] | None = None, chunks: list[bytes | Exception] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int):
        del chunk_size
        for item in self._chunks:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self) -> None:
        self.closed = True


class _InterruptedSession:
    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self.headers: dict[str, str] = {}
        self._payload = payload
        self._source_headers = headers
        self.responses: list[_FakeResponse] = []
        self.get_calls = 0

    def head(self, url: str, **kwargs) -> _FakeResponse:
        del url, kwargs
        response = _FakeResponse(200, headers=self._source_headers)
        self.responses.append(response)
        return response

    def get(self, url: str, *, headers: dict[str, str], **kwargs) -> _FakeResponse:
        del url, kwargs
        start = int(headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        self.get_calls += 1
        if self.get_calls == 1:
            response = _FakeResponse(
                200,
                chunks=[self._payload[:128], requests.ConnectionError("interrupted")],
            )
        else:
            response = _FakeResponse(206, chunks=[self._payload[start:]])
        self.responses.append(response)
        return response


def test_download_resumes_after_stream_interruption_and_closes_responses(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE yfcc100m_dataset (photoid INTEGER, uid TEXT)")
    payload = source.read_bytes()
    headers = {"Content-Length": str(len(payload)), "ETag": '"etag"', "Last-Modified": "date"}
    config = {
        "dataset_id": "test", "claim_ceiling": "test", "user_agent": "test",
        "source": {"url": "https://example.invalid/index", "expected_bytes": len(payload), "expected_etag": "etag", "expected_last_modified": "date"},
        "download_limits": {
            "minimum_free_space_after_download_bytes": 0, "chunk_bytes": 64,
            "progress_checkpoint_bytes": len(payload), "connect_timeout_seconds": 1,
            "read_timeout_seconds": 1, "request_retries": 2, "retry_backoff_seconds": 0,
        },
        "scan": {"required_columns": ["photoid", "uid"]},
    }
    session = _InterruptedSession(payload, headers)
    destination = tmp_path / "download.sqlite"
    manifest = download_full_index(config, destination, session=session)
    assert destination.read_bytes() == payload
    assert manifest["bytes"] == len(payload)
    assert session.get_calls == 2
    assert all(response.closed for response in session.responses)
