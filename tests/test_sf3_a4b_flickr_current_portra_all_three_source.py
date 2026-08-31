from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.flickr_current_portra_all_three_source import (
    FlickrCurrentPortraSourceError,
    audit_metadata_payloads,
    canonical_bytes,
    classify_target_row,
)


def _row(photo_id: str, description: str, *, license_id: int = 4) -> dict[str, object]:
    return {
        "id": photo_id,
        "owner": "owner-1",
        "ownername": "Author",
        "license": str(license_id),
        "media": "photo",
        "title": "",
        "description": {"_content": description},
        "tags": "film",
        "datetaken": "2011-01-01 00:00:00",
    }


def _fixture(
    *, altered_velvia: bool, complete_groups: bool
) -> tuple[list[dict[str, object]], bytes, dict[str, object]]:
    rows = [
        _row("1", "Kodak Ektar 100"),
        _row("2", "Kodak Ektar 100"),
        _row("3", "New Kodak Portra 400"),
        _row("4", "Portra 400 (new)"),
        _row(
            "5",
            "Fujifilm Velvia 50 desaturated"
            if altered_velvia
            else "Fujifilm Velvia 50",
        ),
        _row("6", "Fuji Velvia 50"),
    ]
    config: dict[str, object] = {
        "experiment_id": "test",
        "source": {
            "author_name": "Author",
            "owner_nsid": "owner-1",
            "api_pages": 2,
            "expected_public_photo_count": 6,
            "allowed_license_ids": [4],
            "license_label": "CC BY 2.0",
            "current_portra_indicators": [
                "new portra",
                "new kodak portra",
                "portra 400 (new)",
                "newportra",
            ],
            "disqualifying_alteration_terms": ["desaturated", "de-saturated"],
            "expected_target_counts": {
                "fujifilm_velvia_50": 2,
                "kodak_ektar_100": 2,
                "kodak_portra_400_current": 2,
            },
        },
        "baseline": {
            "bytes": 0,
            "sha256": "",
            "required_qualifying_author_count": 4,
            "required_total_after_addition": 5,
        },
        "group_evidence": {
            "explicit_cross_stock_camera_group_count": 1 if complete_groups else 0,
            "explicit_roll_group_count": 3 if complete_groups else 0,
            "explicit_process_group_count": 1 if complete_groups else 0,
            "explicit_scanner_group_count": 1 if complete_groups else 0,
            "sealed_confirmation_group_count": 1 if complete_groups else 0,
        },
        "admission_minimums": {
            "unaltered_rows_per_stock": 1,
            "explicit_cross_stock_camera_group_count": 1,
            "explicit_roll_group_count": 3,
            "explicit_process_group_count": 1,
            "explicit_scanner_group_count": 1,
            "sealed_confirmation_group_count": 1,
        },
        "operation_limits": {
            "author_html_get_requests": 1,
            "api_metadata_get_requests": 2,
            "image_url_requests": 0,
            "image_head_requests": 0,
            "image_range_requests": 0,
            "image_body_requests": 0,
            "exif_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pixel_admission_passes": "PASS",
        "decision_if_pixel_admission_fails": "FAIL",
        "claim_ceiling": "test",
    }
    target_rows = [classify_target_row(row, config) for row in rows]
    targets = sorted(
        (row for row in target_rows if row is not None),
        key=lambda row: row["photo_id"],
    )
    config["source"]["expected_target_manifest_sha256"] = hashlib.sha256(
        canonical_bytes(targets)
    ).hexdigest()
    baseline = {
        "portra_generation": {
            "qualifying_current_portra_three_stock_author_uids": [
                "author-a",
                "author-b",
                "author-c",
                "author-d",
            ]
        }
    }
    baseline_payload = json.dumps(baseline, separators=(",", ":")).encode()
    config["baseline"]["bytes"] = len(baseline_payload)
    config["baseline"]["sha256"] = hashlib.sha256(baseline_payload).hexdigest()
    payloads: list[dict[str, object]] = []
    for page, page_rows in ((1, rows[:3]), (2, rows[3:])):
        payloads.append(
            {
                "stat": "ok",
                "photos": {"page": page, "total": 6, "photo": page_rows},
            }
        )
    return payloads, baseline_payload, config


def test_complete_source_is_exact_across_page_order() -> None:
    payloads, baseline, config = _fixture(altered_velvia=False, complete_groups=True)
    forward = audit_metadata_payloads(payloads, baseline, config)
    reverse = audit_metadata_payloads(payloads, baseline, config, reverse=True)
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["source_gates"].values())
    assert all(forward["connectivity_gates"].values())
    assert all(forward["pixel_admission_gates"].values())
    assert forward["connectivity"]["augmented_qualifying_author_count"] == 5


def test_explicit_velvia_alteration_fails_pixel_gate_but_not_connectivity() -> None:
    payloads, baseline, config = _fixture(altered_velvia=True, complete_groups=True)
    config["admission_minimums"]["unaltered_rows_per_stock"] = 2
    report = audit_metadata_payloads(payloads, baseline, config)
    assert report["decision"] == "FAIL"
    assert all(report["source_gates"].values())
    assert all(report["connectivity_gates"].values())
    assert not report["pixel_admission_gates"]["unaltered_rows_per_stock"]
    assert (
        report["target_manifest"]["unaltered_stock_counts"]["fujifilm_velvia_50"] == 1
    )


def test_plain_portra_is_generation_ambiguous_and_excluded() -> None:
    _, _, config = _fixture(altered_velvia=False, complete_groups=True)
    assert classify_target_row(_row("7", "Kodak Portra 400"), config) is None
    current = classify_target_row(_row("8", "Kodak New Portra 400"), config)
    assert current is not None
    assert current["stock_id"] == "kodak_portra_400_current"


def test_wrong_license_cannot_satisfy_frozen_manifest() -> None:
    payloads, baseline, config = _fixture(altered_velvia=False, complete_groups=True)
    changed = copy.deepcopy(payloads)
    changed[0]["photos"]["photo"][0]["license"] = "2"
    report = audit_metadata_payloads(changed, baseline, config)
    assert report["decision"] == "FAIL"
    assert not report["source_gates"]["target_manifest_sha256_exact"]
    assert not report["source_gates"]["target_stock_counts_exact"]


def test_duplicate_photo_identity_rejects_before_report() -> None:
    payloads, baseline, config = _fixture(altered_velvia=False, complete_groups=True)
    changed = copy.deepcopy(payloads)
    changed[1]["photos"]["photo"][0]["id"] = "1"
    with pytest.raises(FlickrCurrentPortraSourceError, match="duplicate"):
        audit_metadata_payloads(changed, baseline, config)


def test_project_contract_forbids_media_urls_and_pixels() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root / "configs/sf3_a4b_flickr_current_portra_all_three_source_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["source"]["expected_public_photo_count"] == 3285
    assert config["source"]["api_pages"] == 7
    assert config["source"]["expected_target_counts"] == {
        "fujifilm_velvia_50": 1,
        "kodak_ektar_100": 47,
        "kodak_portra_400_current": 29,
    }
    for key in (
        "image_url_requests",
        "image_head_requests",
        "image_range_requests",
        "image_body_requests",
        "exif_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert config["operation_limits"][key] == 0
