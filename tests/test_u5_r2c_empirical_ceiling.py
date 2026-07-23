from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.eval.empirical_ceiling import (
    IDENTITY_ID,
    EmpiricalCeilingError,
    all_scene_tie_score,
    annotation_workload,
    load_empirical_ceiling_contract,
    validate_b1_manifest,
    validate_complete_policy,
    validate_panel_isolation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u5_r2c_empirical_ceiling_v1.json"


def _lineage_row(record_id: str, split: str, suffix: str, origin: str) -> dict:
    return {
        "record_id": record_id,
        "split": split,
        "evidence_origin": origin,
        "parent_scene_id": f"scene-{suffix}",
        "source_id": f"source-{suffix}",
        "uploader_or_creator_group": f"creator-{suffix}",
        "camera_group": f"camera-{suffix}",
        "roll_group": f"roll-{suffix}",
        "exact_hash": (suffix[0] if suffix[0] in "abcdef" else "a") * 64,
        "perceptual_hash": f"phash-{suffix}",
        "transform_family": f"transform-{suffix}",
        "failure_family": f"failure-{suffix}",
    }


def test_contract_freezes_exact_bank_fallback_and_authority() -> None:
    config = load_empirical_ceiling_contract(CONFIG)
    assert config["candidate_budget_k"] == 7
    assert config["global_comparator"] == "anchor56_chroma_margin4_challenger"
    assert config["identity_counts_toward_k"] is False
    assert config["planning_endpoint"]["binding_sample_size"] is None
    mutated = copy.deepcopy(config)
    mutated["pixel_acquisition_authorized"] = True
    path = ROOT / "outputs" / "u5_r2c_empirical_ceiling_v1" / "invalid_config_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(mutated), encoding="utf-8")
    try:
        with pytest.raises(EmpiricalCeilingError, match="must remain false"):
            load_empirical_ceiling_contract(path)
    finally:
        path.unlink(missing_ok=True)


def test_workload_matches_frozen_54n_and_78n_counts() -> None:
    config = load_empirical_ceiling_contract(CONFIG)
    base = annotation_workload(config, scene_count=600)
    maximum = annotation_workload(
        config,
        scene_count=600,
        candidate_severity_escalation_fraction=1.0,
        deployed_severity_escalation_fraction=1.0,
    )
    assert base["counts"]["base_total"] == 32_400
    assert base["counts"]["total_with_escalation"] == 32_400
    assert maximum["counts"]["total_with_escalation"] == 46_800
    assert maximum["binding_sample_size"] is None
    with pytest.raises(EmpiricalCeilingError, match="positive integer"):
        annotation_workload(config, scene_count=0)


def test_panel_isolation_rejects_overlap_and_duplicates() -> None:
    report = validate_panel_isolation(["a" * 64], ["b" * 64])
    assert report["panels_disjoint"] is True
    with pytest.raises(EmpiricalCeilingError, match="overlap"):
        validate_panel_isolation(["a" * 64], ["a" * 64])
    with pytest.raises(EmpiricalCeilingError, match="duplicate"):
        validate_panel_isolation(["a" * 64, "a" * 64], ["b" * 64])


def test_b1_manifest_rejects_forbidden_origin_and_existing_lineage_overlap() -> None:
    config = load_empirical_ceiling_contract(CONFIG)
    existing = [_lineage_row("a0", "A0", "a0", "synthetic_a0")]
    proposed = [_lineage_row("b1", "B1", "b1", "new_rights_cleared_b1")]
    report = validate_b1_manifest(
        proposed,
        existing_rows=existing,
        forbidden_origins=config["forbidden_b1_evidence_assignments"],
    )
    assert report["b1_manifest_eligible"] is True
    forbidden = [dict(proposed[0], evidence_origin="u5_r2b_render_bank")]
    with pytest.raises(EmpiricalCeilingError, match="forbidden"):
        validate_b1_manifest(
            forbidden,
            existing_rows=existing,
            forbidden_origins=config["forbidden_b1_evidence_assignments"],
        )
    leaked = copy.deepcopy(proposed)
    leaked[0]["parent_scene_id"] = existing[0]["parent_scene_id"]
    with pytest.raises(EmpiricalCeilingError, match="lineage leakage"):
        validate_b1_manifest(
            leaked,
            existing_rows=existing,
            forbidden_origins=config["forbidden_b1_evidence_assignments"],
        )


def test_complete_policy_requires_every_scene_and_justified_identity() -> None:
    config = load_empirical_ceiling_contract(CONFIG)
    rows = [
        {
            "parent_scene_id": "s1",
            "eligible_candidate_ids": ["anchor09_color_only"],
            "selected_candidate_id": "anchor09_color_only",
        },
        {
            "parent_scene_id": "s2",
            "eligible_candidate_ids": [],
            "selected_candidate_id": IDENTITY_ID,
        },
    ]
    report = validate_complete_policy(
        ["s1", "s2"], rows, candidate_ids=config["candidate_ids"]
    )
    assert report["identity_fallback_count"] == 1
    assert report["all_scenes_in_denominator"] is True
    invalid = copy.deepcopy(rows)
    invalid[1]["selected_candidate_id"] = config["global_comparator"]
    with pytest.raises(EmpiricalCeilingError, match="fall back to identity"):
        validate_complete_policy(
            ["s1", "s2"], invalid, candidate_ids=config["candidate_ids"]
        )


def test_tie_score_is_all_scene_and_descriptive_only() -> None:
    report = all_scene_tie_score(
        ["s1", "s2", "s3"],
        [
            {"parent_scene_id": "s1", "outcome": "win"},
            {"parent_scene_id": "s2", "outcome": "tie"},
            {"parent_scene_id": "s3", "outcome": "loss"},
        ],
    )
    assert report["all_scene_tie_score"] == pytest.approx(0.5)
    assert report["one_sided_lower_bound"] is None
    assert report["binding_decision_allowed"] is False
    with pytest.raises(EmpiricalCeilingError, match="missing scenes"):
        all_scene_tie_score(
            ["s1", "s2"],
            [{"parent_scene_id": "s1", "outcome": "win"}],
        )
