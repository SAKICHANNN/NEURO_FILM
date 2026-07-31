from __future__ import annotations

import json
from pathlib import Path
import re

from src.eval.rawpixls_confirmation_preflight import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bk17s_factorized_ao6_sixth_fresh_source_v1.json"
)
DECISION = (
    ROOT
    / "configs/u5_r2bk17s_factorized_ao6_sixth_fresh_source_decision_v1.json"
)


def _walk_rows(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, dict):
        if {"make", "model", "sha256"} <= value.keys():
            rows.append(value)
        for child in value.values():
            rows.extend(_walk_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_rows(child))
    return rows


def test_bk17s_contract_is_frozen_before_candidate_pixel_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    rows = config["candidates"]
    selection = config["selection"]
    assert config["status"] == "contract_frozen_acquisition_ready"
    assert len(rows) == 12
    assert len({row["make"].casefold() for row in rows}) == 12
    assert selection["new_make_count"] == 5
    assert selection["expected_download_bytes_upper_bound"] == 305020273
    assert config["maximum_download_bytes"] == 320 * 1024 * 1024
    assert not selection["candidate_pixels_inspected_before_freeze"]
    assert not selection["bk16_outputs_inspected_before_source_eligibility"]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]


def test_bk17s_candidates_are_disjoint_from_all_prior_tracked_contracts() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = config["candidates"]
    prior_ids: set[int] = set()
    prior_hashes: set[str] = set()
    prior_models: set[tuple[str, str]] = set()
    prior_makes: set[str] = set()
    for path in ROOT.glob("configs/*.json"):
        if path == CONFIG:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        prior_ids.update(
            int(value)
            for value in re.findall(
                r"raw\.pixls\.us/getfile\.php/(\d+)/(?:nice|exif)/",
                json.dumps(payload),
            )
        )
        for row in _walk_rows(payload):
            prior_hashes.add(str(row["sha256"]).casefold())
            make = str(row["make"]).casefold()
            prior_makes.add(make)
            prior_models.add((make, str(row["model"]).casefold()))
    assert not ({int(row["repository_id"]) for row in rows} & prior_ids)
    assert not ({str(row["sha256"]).casefold() for row in rows} & prior_hashes)
    assert not (
        {
            (str(row["make"]).casefold(), str(row["model"]).casefold())
            for row in rows
        }
        & prior_models
    )
    assert sum(str(row["make"]).casefold() not in prior_makes for row in rows) == 5


def test_bk17s_cross_pool_manifest_set_includes_fifth_fresh_population() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    paths = {
        row["path"] for row in config["preflight"]["comparison_manifests"]
    }
    assert (
        "outputs/u5_r2bk13s_orthogonal_residual_replacement_v2_source_v1/"
        "run_a_retry/manifest.json"
    ) in paths
    assert any("r2bh1s" in path for path in paths)
    assert any("r2bk8s" in path for path in paths)
    assert any("color_baseline" in path for path in paths)


def test_bk17s_preserves_fixed_algorithm_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["parent_decision"]["required_decision"] == (
        "retain_fixed_bk16_open_sixth_fresh_source_preflight"
    )
    assert config["next_leaf"]["fixed_arms"] == [
        "fixed_bk16_safe_base_factorized_ao6_residual",
        "fixed_bk7_smooth_perceptual_hue_density",
        "fixed_ao6_colour_only_t15_c35",
        "safe_rich_velvia_50",
    ]
    assert not config["next_leaf"]["strength_retuning_allowed"]
    assert not config["next_leaf"]["router_training_allowed"]
    assert not config["next_leaf"]["operator_output_access_before_source_decision"]


def test_bk17s_decision_opens_only_fixed_sixth_fresh_confirmation() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["status"] == "source_eligible_fixed_comparison_open"
    assert decision["decision"] == "open_fixed_bk16_sixth_fresh_confirmation"
    assert result["automatic_pass"]
    assert result["visual_pass"]
    assert result["eligible_rows"] == 12
    assert result["eligible_camera_makes"] == 12
    assert result["confirmed_severe_source_artifact_count"] == 0
    assert result["within_exact_pairs"] == 0
    assert result["within_dhash_pairs_le_4"] == 0
    assert result["cross_exact_pairs"] == 0
    assert result["cross_dhash_pairs_le_4"] == 0
    assert not result["operator_outputs_inspected"]
    assert decision["evidence"]["repeat_manifest_sha256_exact"]
    assert decision["evidence"]["repeat_report_sha256_exact"]
    assert decision["evidence"]["repeat_contact_sheet_sha256_exact"]
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
    assert not decision["selector_training_allowed"]
    assert not decision["strength_retuning_allowed"]
