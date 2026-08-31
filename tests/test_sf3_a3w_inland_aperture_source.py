from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.inland_aperture_source import (
    InlandApertureSourceError,
    run_inland_aperture_source_audit,
)


def _photo_page(row: dict[str, object], *, author_url: str, license_url: str) -> bytes:
    photo_url = f'{author_url}/{row["id"]}'
    model = {
        "data": {
            "_flickrModelRegistry": "photo-models",
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "license": 11,
            "oWidth": row["width"],
            "oHeight": row["height"],
        },
        "exportMetaType": "model",
    }
    stats = {
        "data": {
            "_flickrModelRegistry": "photo-stats-models",
            "id": row["id"],
            "dateTaken": row["date_taken"],
            "datePosted": row["date_posted"],
        },
        "exportMetaType": "model",
    }
    rights = {
        "@context": "https://schema.org/",
        "@graph": [
            {
                "@type": "ImageObject",
                "contentUrl": f'https://media.invalid/{row["id"]}.jpg',
                "license": license_url,
                "acquireLicensePage": photo_url,
                "author": {"@type": "Person", "name": "Matt", "url": author_url},
            }
        ],
    }
    return (
        "<html><script>window.model="
        + json.dumps(model, separators=(",", ":"))
        + ";window.stats="
        + json.dumps(stats, separators=(",", ":"))
        + ";</script><script type=\"application/ld+json\">"
        + json.dumps(rights)
        + "</script></html>"
    ).encode()


def _fixture(tmp_path: Path, *, complete_groups: bool = False) -> tuple[Path, dict[str, bytes]]:
    rows = [
        {
            "id": f"{index}",
            "stock": "portra_400" if index < 4 else "ektar_100",
            "title": f"title-{index}",
            "description": f"Canon EOS1 stock-{index}",
            "width": 100 + index,
            "height": 200 + index,
            "date_taken": f"2026-08-{index + 1:02d} 00:00:00",
            "date_posted": str(1000 + index),
        }
        for index in range(7)
    ]
    import hashlib

    expected_rows = sorted(
        ({**row, "license": 11} for row in rows), key=lambda row: row["id"]
    )
    manifest_sha = hashlib.sha256(
        json.dumps(
            expected_rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    groups = {
        "target_stock_count": 2,
        "observed_photo_count": 7,
        "same_author_camera_cross_stock_control_count": 1,
        "independent_author_source_count": 1,
        "explicit_roll_group_count": 0,
        "explicit_process_group_count": 0,
        "explicit_scanner_group_count": 0,
        "same_scene_neutral_film_pair_count": 0,
        "sealed_confirmation_group_count": 0,
    }
    minimums = dict(groups)
    if not complete_groups:
        minimums["independent_author_source_count"] = 2
    config = {
        "experiment_id": "test",
        "source": {
            "author_name": "Matt",
            "author_slug": "inlandaperture",
            "author_url": "https://example.test/photos/inlandaperture",
            "license_code": 11,
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "required_camera_pattern": r"canon\s+eos\s*1",
            "expected_manifest_sha256": manifest_sha,
            "rows": rows,
        },
        "group_evidence": groups,
        "admission_minimums": minimums,
        "operation_limits": {
            "photo_html_get_requests": 7,
            "image_head_requests": 0,
            "image_range_requests": 0,
            "image_body_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    payloads = {
        f'{config["source"]["author_url"]}/{row["id"]}/': _photo_page(
            row,
            author_url=config["source"]["author_url"],
            license_url=config["source"]["license_url"],
        )
        for row in rows
    }
    return config_path, payloads


def test_complete_mock_source_is_exact_across_request_order(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path, complete_groups=True)
    requests: list[str] = []

    def fetcher(url: str) -> bytes:
        requests.append(url)
        assert "media.invalid" not in url
        return payloads[url]

    forward = run_inland_aperture_source_audit(config, fetcher=fetcher)
    reverse = run_inland_aperture_source_audit(config, reverse=True, fetcher=fetcher)
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["audit_gates"].values())
    assert len(requests) == 14


def test_single_source_groups_fail_without_media(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    report = run_inland_aperture_source_audit(config, fetcher=lambda url: payloads[url])
    assert report["decision"] == "FAIL"
    assert all(report["audit_gates"].values())
    assert not report["admission_gates"]["independent_author_source_count"]
    assert report["operation_counts"]["image_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_wrong_rights_record_fails_source_gate(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    url = next(iter(payloads))
    payloads[url] = payloads[url].replace(b"licenses/by/4.0", b"licenses/by-nc/4.0")
    report = run_inland_aperture_source_audit(config, fetcher=lambda item: payloads[item])
    assert not report["audit_gates"]["per_work_cc_by_4_author_exact"]
    assert report["decision"] == "FAIL"


def test_missing_photo_model_fails_closed(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    url = next(iter(payloads))
    payloads[url] = b"<html></html>"
    with pytest.raises(InlandApertureSourceError):
        run_inland_aperture_source_audit(config, fetcher=lambda item: payloads[item])


def test_project_contract_forbids_media_and_pixel_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/sf3_a3w_inland_aperture_portra_ektar_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["source"]["license_code"] == 11
    assert len(config["source"]["rows"]) == 7
    for key in (
        "image_head_requests",
        "image_range_requests",
        "image_body_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert config["operation_limits"][key] == 0
