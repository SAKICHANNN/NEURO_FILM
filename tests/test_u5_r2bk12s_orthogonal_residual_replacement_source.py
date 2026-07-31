from __future__ import annotations

import json
from pathlib import Path
import re

from src.eval.rawpixls_confirmation_preflight import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2bk12s_orthogonal_residual_replacement_source_v1.json"
)
DECISION = (
    ROOT
    / "configs/u5_r2bk12s_orthogonal_residual_replacement_source_decision_v1.json"
)


def _repository_ids(payload: object) -> set[int]:
    return {
        int(value)
        for value in re.findall(
            r"raw\.pixls\.us/getfile\.php/(\d+)/(?:nice|exif)/",
            json.dumps(payload),
        )
    }


def test_bk12s_contract_is_frozen_before_replacement_pixel_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    rows = config["candidates"]
    assert config["status"] == "contract_frozen_acquisition_ready"
    assert len(rows) == 12
    assert len({row["make"].casefold() for row in rows}) == 12
    assert len(config["retained_from_failed_pool"]["ids"]) == 7
    assert len(config["replacement_ids"]) == 5
    assert config["selection"]["retained_candidate_count"] == 7
    assert config["selection"]["replacement_candidate_count"] == 5
    assert sum(row["bytes_reported_mb"] for row in rows) == 252.31
    assert config["maximum_download_bytes"] == 270 * 1024 * 1024
    assert not config["retained_from_failed_pool"][
        "operator_outputs_inspected"
    ]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]


def test_bk12s_retained_rows_and_replacements_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(
        (
            ROOT
            / config["retained_from_failed_pool"]["decision"]
        ).read_text(encoding="utf-8")
    )
    assert set(config["retained_from_failed_pool"]["ids"]) == set(
        parent["result"]["retained_rows"]
    )
    rows = {row["id"]: row for row in config["candidates"]}
    assert set(rows) == (
        set(config["retained_from_failed_pool"]["ids"])
        | set(config["replacement_ids"])
    )


def test_bk12s_replacements_are_disjoint_from_prior_tracked_pools() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    replacements = {
        row["id"]: row
        for row in config["candidates"]
        if row["id"] in config["replacement_ids"]
    }
    current_ids = {row["repository_id"] for row in replacements.values()}
    current_hashes = {row["sha256"] for row in replacements.values()}
    current_models = {
        (row["make"].casefold(), row["model"].casefold())
        for row in replacements.values()
    }
    current_makes = {row["make"].casefold() for row in replacements.values()}
    prior_ids: set[int] = set()
    prior_hashes: set[str] = set()
    prior_models: set[tuple[str, str]] = set()
    prior_makes: set[str] = set()
    for path in ROOT.glob("configs/*.json"):
        if path == CONFIG:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        prior_ids.update(_repository_ids(payload))
        rows = payload.get("candidates", []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("sha256")
            if isinstance(value, str) and len(value) == 64:
                prior_hashes.add(value)
            if "make" in row and "model" in row:
                make = row["make"].casefold()
                prior_makes.add(make)
                prior_models.add((make, row["model"].casefold()))
    assert current_ids.isdisjoint(prior_ids)
    assert current_hashes.isdisjoint(prior_hashes)
    assert current_models.isdisjoint(prior_models)
    assert current_makes.isdisjoint(prior_makes)


def test_bk12s_selection_excludes_nonphotographic_metadata() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = re.compile(r"\b(scanner|chart|flat|target|film|buggy)\b", re.I)
    for row in config["candidates"]:
        text = " ".join(
            str(row[key]) for key in ("make", "model", "mode", "filename")
        )
        assert forbidden.search(text) is None


def test_bk12s_decision_preserves_automatic_failure() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["decision"] == (
        "close_bk12s_open_corrected_replacement_freeze"
    )
    assert not result["automatic_pass"]
    assert not result["visual_review_allowed"]
    assert not result["visual_review_performed"]
    assert result["decoded_rows"] == 11
    assert result["decoded_camera_makes"] == 11
    assert result["failed_gate"]["name"] == (
        "maximum_largest_make_fraction"
    )
    assert result["failed_gate"]["observed"] == 1 / 11
    assert result["failed_gate"]["required_maximum"] == 1 / 12
    assert not result["operator_outputs_inspected"]
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
