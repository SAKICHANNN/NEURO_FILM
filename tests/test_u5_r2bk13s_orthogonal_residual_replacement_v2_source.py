from __future__ import annotations

import json
from pathlib import Path
import re

from src.eval.rawpixls_confirmation_preflight import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk13s_orthogonal_residual_replacement_v2_source_v1.json"
BK12 = ROOT / "configs/u5_r2bk12s_orthogonal_residual_replacement_source_v1.json"
DECISION = ROOT / "configs/u5_r2bk12s_orthogonal_residual_replacement_source_decision_v1.json"


def _repository_ids(payload: object) -> set[int]:
    return {
        int(value)
        for value in re.findall(
            r"raw\.pixls\.us/getfile\.php/(\d+)/(?:nice|exif)/",
            json.dumps(payload),
        )
    }


def test_bk13s_contract_is_frozen_before_replacement_pixel_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    rows = config["candidates"]
    assert config["status"] == "contract_frozen_acquisition_ready"
    assert len(rows) == 12
    assert len({row["make"].casefold() for row in rows}) == 12
    assert config["selection"]["carried_candidate_count"] == 11
    assert config["selection"]["replacement_candidate_count"] == 1
    assert abs(sum(row["bytes_reported_mb"] for row in rows) - 246.36) < 1e-9
    assert config["maximum_download_bytes"] == 265 * 1024 * 1024
    assert config["preflight"]["maximum_largest_make_fraction"] == 1 / 9
    assert not config["carried_from_failed_pool"]["source_contact_sheet_inspected"]
    assert not config["carried_from_failed_pool"]["operator_outputs_inspected"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]


def test_bk13s_carries_only_decoded_bk12_rows_exactly() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bk12 = json.loads(BK12.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    carried = set(config["carried_from_failed_pool"]["ids"])
    assert carried == {row["id"] for row in bk12["candidates"]} - {"xiro_xplorer_v"}
    assert decision["result"]["fixed_decode_failures"][0]["id"] == "xiro_xplorer_v"
    expected = {row["id"]: row for row in bk12["candidates"] if row["id"] in carried}
    actual = {row["id"]: row for row in config["candidates"] if row["id"] in carried}
    assert actual == expected


def test_bk13s_only_new_row_is_disjoint_from_prior_tracked_pools() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    replacement = next(
        row for row in config["candidates"] if row["id"] == "kandao_qoocam"
    )
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
    key = (replacement["make"].casefold(), replacement["model"].casefold())
    assert replacement["repository_id"] not in prior_ids
    assert replacement["sha256"] not in prior_hashes
    assert key not in prior_models
    assert replacement["make"].casefold() not in prior_makes


def test_bk13s_does_not_add_failed_pool_to_cross_comparisons() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    paths = {row["path"] for row in config["preflight"]["comparison_manifests"]}
    assert all("bk12s" not in path.casefold() for path in paths)
    assert all("bk11s" not in path.casefold() for path in paths)


def test_bk13s_preserves_fixed_algorithm_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["parent_decision"]["required_decision"] == (
        "close_bk12s_open_corrected_replacement_freeze"
    )
    assert config["next_leaf"]["fixed_arms"] == [
        "fixed_bk10_safe_base_orthogonal_residual",
        "fixed_bk7_smooth_perceptual_hue_density",
        "fixed_ao6_colour_only_t15_c35",
        "safe_rich_velvia_50",
    ]
    assert not config["next_leaf"]["strength_retuning_allowed"]
    assert not config["next_leaf"]["router_training_allowed"]
