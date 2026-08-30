from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.filmmatch_portra_source import run_source_audit


def _config(tmp_path: Path, payloads: dict[str, bytes], *, admitted: bool) -> Path:
    roles = ("portramatch", "learnmore", "shooting_charts")
    config = {
        "experiment_id": "test",
        "sources": {
            role: {
                "url": f"https://example/{role}",
                "bytes": len(payloads[role]),
                "sha256": hashlib.sha256(payloads[role]).hexdigest(),
                "required_phrases": [f"method {role}"],
            }
            for role in roles
        },
        "public_links": {
            "known_practice_charts_folder_id": "practice",
            "known_profiling_tools_folder_id": "tools",
            "portra_specific_dataset_folder_ids": ["portra-data"] if admitted else [],
        },
        "target_stock": {"film_stock_id": "portra", "public_label": "Portra 400"},
        "admission_markers": {
            "dataset_rights": ["dataset license"],
            "manifest": ["sha256 manifest"],
            "group_roles": ["roll_id sealed confirmation"],
        },
        "operation_limits": {
            "html_requests": 3,
            "image_requests": 0,
            "drive_metadata_requests": 0,
            "drive_body_requests": 0,
            "product_downloads": 0,
            "chart_media_requests": 0,
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
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _payloads(*, admitted: bool) -> dict[str, bytes]:
    suffix = (
        ' dataset license sha256 manifest roll_id sealed confirmation '
        '<a href="https://drive.google.com/drive/folders/portra-data">data</a>'
        if admitted
        else ""
    )
    return {
        role: f"<html>method {role}{suffix}</html>".encode()
        for role in ("portramatch", "learnmore", "shooting_charts")
    }


def test_controlled_but_private_portra_source_fails_before_media(tmp_path: Path) -> None:
    payloads = _payloads(admitted=False)
    config = _config(tmp_path, payloads, admitted=False)
    report = run_source_audit(config, html_reader=lambda url: payloads[url.rsplit("/", 1)[-1]])
    assert report["decision"] == "FAIL"
    assert report["gates"]["controlled_portra_method_facts_present"]
    assert not report["gates"]["public_portra_specific_observation_payload_present"]
    assert report["operation_counts"]["drive_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_complete_mock_source_can_only_open_next_audit(tmp_path: Path) -> None:
    payloads = _payloads(admitted=True)
    config = _config(tmp_path, payloads, admitted=True)
    reader = lambda url: payloads[url.rsplit("/", 1)[-1]]
    forward = run_source_audit(config, html_reader=reader)
    reverse = run_source_audit(
        config,
        html_reader=reader,
        request_order=("shooting_charts", "learnmore", "portramatch"),
    )
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["gates"].values())


def test_request_order_must_be_exact(tmp_path: Path) -> None:
    payloads = _payloads(admitted=False)
    config = _config(tmp_path, payloads, admitted=False)
    with pytest.raises(ValueError, match="each frozen source"):
        run_source_audit(
            config,
            html_reader=lambda url: payloads[url.rsplit("/", 1)[-1]],
            request_order=("portramatch", "portramatch", "shooting_charts"),
        )


def test_project_contract_forbids_media_and_drive_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/sf3_a3r_filmmatch_portra_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["target_stock"]["film_stock_id"] == "kodak_portra_400"
    assert config["public_links"]["portra_specific_dataset_folder_ids"] == []
    assert all(
        config["operation_limits"][key] == 0
        for key in config["operation_limits"]
        if key != "html_requests"
    )
