from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.yfcc_three_stock_generation_connectivity import (
    YfccThreeStockConnectivityError,
    run_connectivity_audit,
)

STOCKS = ["kodak_ektar_100", "fujifilm_velvia_50", "kodak_portra_400"]


def _write_fixture(tmp_path: Path, portra_titles: dict[str, str]) -> Path:
    matches: list[dict[str, object]] = []
    photoid = 1
    for uid, title in portra_titles.items():
        for stock in STOCKS:
            matches.append(
                {
                    "uid": uid,
                    "photoid": photoid,
                    "film_stock_id": stock,
                    "title": title if stock == "kodak_portra_400" else stock,
                    "description": "",
                    "usertags": "",
                }
            )
            photoid += 1
    source = {
        "dataset_id": "fixture-yfcc",
        "image_payloads_downloaded_or_decoded": False,
        "matches": matches,
    }
    source_path = tmp_path / "source.json"
    source_payload = json.dumps(source, sort_keys=True).encode("utf-8")
    source_path.write_bytes(source_payload)
    config = {
        "experiment_id": "SF3.A3Q_FIXTURE",
        "source_report": {
            "path": str(source_path),
            "bytes": len(source_payload),
            "sha256": hashlib.sha256(source_payload).hexdigest(),
            "required_dataset_id": "fixture-yfcc",
        },
        "target_stock_ids": STOCKS,
        "portra_generation": {
            "current_indicators": [
                "new portra",
                "new kodak portra",
                "portra 400 (new)",
            ],
            "legacy_indicators": [
                "portra 400 nc",
                "portra 400 vc",
                "portra 400-nc",
                "portra 400-vc",
                "portra 400nc",
                "portra 400vc",
            ],
            "text_fields": ["title", "description", "usertags"],
        },
        "gates": {"minimum_current_portra_three_stock_author_uids": 5},
        "operation_limits": {
            "sqlite_reads": 0,
            "html_requests": 0,
            "image_requests": 0,
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
    return config_path


def test_five_current_authors_pass(tmp_path: Path) -> None:
    config = _write_fixture(
        tmp_path,
        {f"author-{index}": "New Kodak Portra 400" for index in range(5)},
    )
    report = run_connectivity_audit(config)
    assert report["decision"] == "PASS"
    assert report["connectivity"]["all_three_author_uid_count"] == 5
    assert (
        report["portra_generation"][
            "qualifying_current_portra_three_stock_author_uid_count"
        ]
        == 5
    )


def test_generation_filter_fails_and_current_indicator_takes_precedence(
    tmp_path: Path,
) -> None:
    titles = {f"current-{index}": "New Portra 400" for index in range(4)}
    titles["comparison"] = "New Portra compared with Portra 400 VC"
    titles["legacy"] = "Portra 400 NC"
    titles["ambiguous"] = "Portra 400"
    config = _write_fixture(tmp_path, titles)
    report = run_connectivity_audit(config)
    assert report["decision"] == "PASS"
    assert report["portra_generation"]["classification_row_counts"] == {
        "current_explicit": 5,
        "generation_ambiguous": 1,
        "legacy_explicit": 1,
    }
    assert report["operation_counts"]["image_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_four_current_authors_fail_closed(tmp_path: Path) -> None:
    config = _write_fixture(
        tmp_path,
        {
            **{f"current-{index}": "Portra 400 (new)" for index in range(4)},
            "legacy": "Portra 400VC",
            "ambiguous": "Portra 400",
        },
    )
    report = run_connectivity_audit(config)
    assert report["decision"] == "FAIL"
    assert report["gates"]["minimum_current_portra_three_stock_author_uids"] is False


def test_forward_reverse_stock_order_is_byte_exact(tmp_path: Path) -> None:
    config = _write_fixture(
        tmp_path,
        {f"author-{index}": "New Portra" for index in range(5)},
    )
    forward = run_connectivity_audit(config, stock_order=STOCKS)
    reverse = run_connectivity_audit(config, stock_order=list(reversed(STOCKS)))
    assert json.dumps(forward, sort_keys=True) == json.dumps(reverse, sort_keys=True)


def test_source_identity_drift_rejects_before_adjudication(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, {"author": "New Portra"})
    source_path = Path(
        json.loads(config.read_text(encoding="utf-8"))["source_report"]["path"]
    )
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(YfccThreeStockConnectivityError, match="byte size drifted"):
        run_connectivity_audit(config)


def test_invalid_stock_order_rejects(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, {"author": "New Portra"})
    with pytest.raises(YfccThreeStockConnectivityError, match="stock_order"):
        run_connectivity_audit(config, stock_order=[STOCKS[0], STOCKS[1]])
