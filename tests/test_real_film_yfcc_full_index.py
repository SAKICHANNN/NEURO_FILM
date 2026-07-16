from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.real_film.yfcc_full_index import (
    YfccFullIndexError,
    audit_candidate_rows,
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
