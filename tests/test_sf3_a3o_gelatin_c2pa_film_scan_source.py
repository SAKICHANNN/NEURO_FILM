from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.gelatin_c2pa_film_scan_source import run_source_audit


def _config(tmp_path: Path) -> Path:
    payload = {
        "schema": "test",
        "experiment_id": "test",
        "sources": {
            "credentials": {
                "url": "https://example/credentials",
                "training_forbidden_phrase": "not allowed",
                "required_phrases": ["C2PA", "physical film"],
            },
            "terms": {
                "url": "https://example/terms",
                "customer_rights_phrase": "customer owns",
                "required_phrases": ["customer owns"],
            },
        },
        "target_stocks": [
            {"film_stock_id": "velvia", "public_label": "Velvia 50"},
            {"film_stock_id": "portra", "public_label": "Portra 400"},
            {"film_stock_id": "ektar", "public_label": "Ektar 100"},
        ],
        "admission_requirements": {
            "required_public_artifacts": ["assets", "manifest", "groups", "rights"]
        },
        "network_limits": {
            "html_requests": 2,
            "image_requests": 0,
            "verifier_uploads": 0,
            "fingerprint_queries": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reader(*, complete: bool):
    credentials = "C2PA physical film not allowed Portra 400"
    terms = "customer owns"
    if complete:
        credentials = (
            "C2PA physical film Velvia 50 Portra 400 Ektar 100 "
            "assets manifest groups rights"
        )
    payloads = {
        "https://example/credentials": credentials.encode(),
        "https://example/terms": terms.encode(),
    }
    return lambda url: payloads[url]


def test_signed_but_unlicensed_source_fails_before_images(tmp_path: Path) -> None:
    report = run_source_audit(_config(tmp_path), html_reader=_reader(complete=False))
    assert report["decision"] == "FAIL"
    assert report["admission"]["observed_target_stock_ids"] == ["portra"]
    assert not report["gates"]["complete_target_stock_coverage_present"]
    assert not report["gates"]["training_mining_authorized"]
    assert report["operation_counts"]["image_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_complete_authorized_source_can_only_open_next_preflight(tmp_path: Path) -> None:
    report = run_source_audit(
        _config(tmp_path),
        html_reader=_reader(complete=True),
        request_order=("terms", "credentials"),
    )
    assert report["decision"] == "PASS"
    assert all(report["gates"].values())


def test_request_order_must_be_exact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="credentials and terms"):
        run_source_audit(
            _config(tmp_path),
            html_reader=_reader(complete=False),
            request_order=("credentials", "credentials"),
        )


def test_project_contract_is_zero_media_and_target_stock_first() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/sf3_a3o_gelatin_c2pa_film_scan_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["film_stock_id"] for row in config["target_stocks"]] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert all(
        config["network_limits"][key] == 0
        for key in config["network_limits"]
        if key != "html_requests"
    )
