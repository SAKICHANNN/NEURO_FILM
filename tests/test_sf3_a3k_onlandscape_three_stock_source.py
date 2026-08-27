from __future__ import annotations

import json
from pathlib import Path

from src.real_film.onlandscape_three_stock_source import run_source_audit


def _config(tmp_path: Path) -> Path:
    config = {
        "schema": "test",
        "experiment_id": "test",
        "article": {
            "url": "https://example/article",
            "required_phrases": ["same scene", "same lab", "same scanner"],
            "required_group": "all",
            "required_stocks": [
                {
                    "film_stock_id": "velvia",
                    "label": "Velvia 50",
                    "accepted_asset_names": ["Velvia 50.jpg"],
                },
                {
                    "film_stock_id": "portra",
                    "label": "Portra 400",
                    "accepted_asset_names": ["Portra 400 default.jpg"],
                },
                {
                    "film_stock_id": "ektar",
                    "label": "Ektar",
                    "accepted_asset_names": ["Ektar default.jpg"],
                },
            ],
        },
        "terms": {
            "url": "https://example/terms",
            "required_restriction_phrases": ["single copy"],
            "required_explicit_permissions": ["fit", "release"],
        },
        "network_limits": {
            "html_requests": 2,
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
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _article() -> bytes:
    items = "".join(
        f'<li class="swapper" img="{asset}">{label}</li>'
        for asset, label in (
            ("Velvia 50.jpg", "Velvia 50"),
            ("Portra 400 default.jpg", "Portra 400"),
            ("Ektar default.jpg", "Ektar"),
        )
    )
    return (
        "same <a href='lab'>scene same lab</a> same scanner"
        '<ul urlbase="https://example/film-comparison/all">' + items + "</ul>"
    ).encode()


def test_private_research_terms_fail_before_any_pixel_action(tmp_path: Path) -> None:
    payloads = {
        "https://example/article": _article(),
        "https://example/terms": b"single copy only",
    }
    report = run_source_audit(_config(tmp_path), html_reader=lambda url: payloads[url])
    assert report["decision"] == "FAIL"
    assert report["gates"]["same_group_three_stock_inventory_present"]
    assert not report["gates"][
        "explicit_fitting_release_commercial_permissions_present"
    ]
    assert report["image_requests"] == 0
    assert report["pixel_decodes"] == 0
    assert report["fit_calls"] == 0


def test_explicit_permissions_can_open_only_the_next_preflight(tmp_path: Path) -> None:
    payloads = {
        "https://example/article": _article(),
        "https://example/terms": b"single copy fit release",
    }
    report = run_source_audit(
        _config(tmp_path),
        html_reader=lambda url: payloads[url],
        request_order=("terms", "article"),
    )
    assert report["decision"] == "PASS"
    assert all(report["gates"].values())


def test_missing_stock_closes_structure(tmp_path: Path) -> None:
    article = _article().replace(
        b'<li class="swapper" img="Ektar default.jpg">Ektar</li>', b""
    )
    payloads = {
        "https://example/article": article,
        "https://example/terms": b"single copy fit release",
    }
    report = run_source_audit(_config(tmp_path), html_reader=lambda url: payloads[url])
    assert report["decision"] == "FAIL"
    assert not report["article"]["required_stock_results"]["ektar"]


def test_project_contract_is_zero_pixel_and_three_stock() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/sf3_a3k_onlandscape_three_stock_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["network_limits"] == {
        "html_requests": 2,
        "image_requests": 0,
        "pixel_decodes": 0,
        "fit_calls": 0,
        "render_calls": 0,
        "score_calls": 0,
    }
    assert {row["film_stock_id"] for row in config["article"]["required_stocks"]} == {
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    }
