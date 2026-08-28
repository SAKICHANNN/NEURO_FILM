from __future__ import annotations

import json
from pathlib import Path

from src.real_film.parvec_controlled_film_source import run_source_audit


def _config(tmp_path: Path) -> Path:
    payload = {
        "schema": "test",
        "experiment_id": "test",
        "project": {
            "url": "https://example/project",
            "required_phrases": ["controlled", "manual", "email", "unreleased"],
        },
        "video": {
            "watch_url": "https://example/watch",
            "oembed_url": "https://example/oembed",
            "required_title": "Film test",
            "required_author": "PARVEC",
            "required_chapters": [
                {"film_stock_id": "portra", "title": "Portra 400"},
                {"film_stock_id": "ektar", "title": "Ektar 100"},
            ],
            "required_missing_stock": {
                "film_stock_id": "velvia",
                "title": "Velvia 50",
            },
            "required_protocol_chapter": "PROTOCOL",
        },
        "admission_requirements": {
            "required_stock_ids": ["velvia", "portra", "ektar"],
            "required_public_artifacts": ["measurements", "manifest", "groups", "rights"],
        },
        "network_limits": {
            "html_requests": 3,
            "video_stream_requests": 0,
            "thumbnail_requests": 0,
            "preset_requests": 0,
            "form_submissions": 0,
            "image_requests": 0,
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
    project = "controlled manual email unreleased"
    if complete:
        project += " measurements manifest groups rights"
    watch = "PROTOCOL Portra 400 Ektar 100" + (" Velvia 50" if complete else "")
    payloads = {
        "https://example/project": project.encode(),
        "https://example/watch": watch.encode(),
        "https://example/oembed": json.dumps(
            {
                "title": "Film test",
                "author_name": "PARVEC",
                "author_url": "https://www.youtube.com/@parvec",
            }
        ).encode(),
    }
    return lambda url: payloads[url]


def test_two_stock_unreleased_source_fails_before_media(tmp_path: Path) -> None:
    report = run_source_audit(_config(tmp_path), html_reader=_reader(complete=False))
    assert report["decision"] == "FAIL"
    assert report["gates"]["portra_ektar_chapters_present"]
    assert not report["gates"]["complete_three_stock_coverage_present"]
    assert not report["gates"][
        "public_machine_readable_measurements_and_rights_present"
    ]
    assert report["operation_counts"]["preset_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_complete_authorized_source_can_only_open_next_preflight(tmp_path: Path) -> None:
    report = run_source_audit(
        _config(tmp_path),
        html_reader=_reader(complete=True),
        request_order=("oembed", "watch", "project"),
    )
    assert report["decision"] == "PASS"
    assert all(report["gates"].values())


def test_request_order_must_be_exact(tmp_path: Path) -> None:
    try:
        run_source_audit(
            _config(tmp_path),
            html_reader=_reader(complete=False),
            request_order=("project", "watch", "watch"),
        )
    except ValueError as exc:
        assert "project, watch and oembed" in str(exc)
    else:
        raise AssertionError("invalid request order did not fail")


def test_project_contract_is_zero_media_and_stock_first() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/sf3_a3n_parvec_controlled_film_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["admission_requirements"]["required_stock_ids"] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert all(
        config["network_limits"][key] == 0
        for key in (
            "video_stream_requests",
            "thumbnail_requests",
            "preset_requests",
            "form_submissions",
            "image_requests",
            "pixel_decodes",
            "fit_calls",
            "render_calls",
            "score_calls",
        )
    )
