from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.luminant_portra_source import (
    LuminantPortraSourceError,
    run_luminant_portra_source_audit,
)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _fixture(tmp_path: Path, *, admitted_groups: bool = False):
    base = "https://fixture.test"
    about = (
        b"All images captured by human operator using photosensitive chemical process.\n"
        b"Digital domain transfer through optical reimaging method.\n"
        b"Creative Commons Attribution 4.0 International"
    )
    inventory = [
        {
            "id": "00001",
            "date": "2026/01",
            "film": "Kodak Portra 400",
            "place": "A",
            "full_url": f"{base}/img/00001/00001_full.jpg",
        },
        {
            "id": "00002",
            "date": "2026/02",
            "film": "Kodak Portra 400 +2",
            "place": "B",
            "full_url": f"{base}/img/00002/00002_full.jpg",
        },
    ]
    selected = sorted(
        inventory, key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest()
    )
    selected_ids = [item["id"] for item in selected]
    head_entries = [
        {
            "id": item["id"],
            "status": 200,
            "content_length": 100 + int(item["id"]),
            "content_type": "image/jpeg",
            "etag": f'"{item["id"]}"',
            "last_modified": "date",
            "accept_ranges": "bytes",
        }
        for item in sorted(inventory, key=lambda item: item["id"])
    ]
    group_count = 3 if admitted_groups else 0
    groups = {
        "independent_author_count": group_count,
        "explicit_roll_group_count": group_count,
        "explicit_process_group_count": group_count,
        "explicit_scanner_group_count": group_count,
        "same_scene_neutral_film_pair_count": group_count,
        "sealed_confirmation_group_count": group_count,
    }
    config = {
        "experiment_id": "fixture",
        "source": {
            "base_url": base,
            "about_url": f"{base}/about/",
            "page_url_template": f"{base}/page/{{page:04d}}/",
            "first_page": 1,
            "last_page": 2,
            "author_identity": "fixture",
            "required_about_phrases": about.decode().splitlines(),
            "license_id": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "target": {
            "accepted_labels": ["Kodak Portra 400", "Kodak Portra 400 +2"],
            "expected_inventory_count": 2,
            "expected_label_counts": {
                "Kodak Portra 400": 1,
                "Kodak Portra 400 +2": 1,
            },
            "expected_inventory_sha256": _json_sha256(inventory),
        },
        "head_sample": {
            "selection": "rank",
            "count": 2,
            "expected_ids": selected_ids,
            "expected_total_bytes": sum(
                entry["content_length"] for entry in head_entries
            ),
            "expected_metadata_sha256": _json_sha256(head_entries),
            "required_content_type": "image/jpeg",
            "required_accept_ranges": "bytes",
        },
        "group_evidence": groups,
        "admission_minimums": {key: 1 for key in groups},
        "operation_limits": {
            "html_get_requests": 3,
            "image_head_requests": 2,
            "image_get_requests": 0,
            "image_range_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "fixture",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    page = {
        1: '<img alt="#00001 - 2026/01 - Kodak Portra 400 - A">',
        2: '<img alt="#00002 - 2026/02 - Kodak Portra 400 +2 - B">',
    }

    def fetcher(url: str, method: str):
        if url.endswith("/about/"):
            return 200, {}, about
        if "/page/" in url:
            number = int(url.rstrip("/").rsplit("/", 1)[1])
            return 200, {}, page[number].encode()
        image_id = url.split("/")[-2]
        entry = next(item for item in head_entries if item["id"] == image_id)
        return (
            200,
            {
                "Content-Length": str(entry["content_length"]),
                "Content-Type": entry["content_type"],
                "ETag": entry["etag"],
                "Last-Modified": entry["last_modified"],
                "Accept-Ranges": entry["accept_ranges"],
            },
            b"",
        )

    return config_path, fetcher


def test_single_source_groups_fail_closed(tmp_path: Path) -> None:
    config, fetcher = _fixture(tmp_path)
    report = run_luminant_portra_source_audit(config, fetcher=fetcher)
    assert report["decision"] == "FAIL"
    assert all(report["audit_gates"].values())
    assert report["admission_gates"]["independent_author_count"] is False
    assert report["operation_counts"]["image_get_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_complete_group_evidence_would_open_separate_preflight(tmp_path: Path) -> None:
    config, fetcher = _fixture(tmp_path, admitted_groups=True)
    report = run_luminant_portra_source_audit(config, fetcher=fetcher)
    assert report["decision"] == "PASS"


def test_forward_reverse_transport_order_is_exact(tmp_path: Path) -> None:
    config, fetcher = _fixture(tmp_path)
    forward = run_luminant_portra_source_audit(config, fetcher=fetcher)
    reverse = run_luminant_portra_source_audit(config, reverse=True, fetcher=fetcher)
    assert json.dumps(forward, sort_keys=True) == json.dumps(reverse, sort_keys=True)


def test_inventory_drift_fails_gate(tmp_path: Path) -> None:
    config, fetcher = _fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["target"]["expected_inventory_count"] = 3
    config.write_text(json.dumps(payload), encoding="utf-8")
    report = run_luminant_portra_source_audit(config, fetcher=fetcher)
    assert report["audit_gates"]["inventory_count_exact"] is False
    assert report["decision"] == "FAIL"


def test_head_body_rejects(tmp_path: Path) -> None:
    config, fetcher = _fixture(tmp_path)

    def bad_fetcher(url: str, method: str):
        status, headers, body = fetcher(url, method)
        if method == "HEAD":
            body = b"unexpected"
        return status, headers, body

    with pytest.raises(LuminantPortraSourceError, match="unexpectedly returned a body"):
        run_luminant_portra_source_audit(config, fetcher=bad_fetcher)
