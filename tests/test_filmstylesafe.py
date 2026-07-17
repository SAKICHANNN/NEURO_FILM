from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.filmstylesafe import (
    SCHEMA_ID,
    FilmStyleSafeContractError,
    aggregate_scene_severity,
    load_contract,
    validate_annotation,
    validate_split_manifest,
    zero_event_power_worksheet,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "filmstylesafe_r1a_contract_v1.json"
SCHEMA = ROOT / "configs" / "schemas" / "filmstylesafe_annotation_v1.schema.json"


def _rating(
    reviewer: int,
    label: str = "safe",
    *,
    reviewer_kind: str = "human_initial",
    split: str = "A0",
    evidence_origin: str = "synthetic_a0_fixture",
) -> dict:
    severe = label == "severe"
    return {
        "schema_id": SCHEMA_ID,
        "record_id": f"rating-{reviewer}-{reviewer_kind}-{label}",
        "record_kind": "severity_rating",
        "split": split,
        "evidence_origin": evidence_origin,
        "parent_scene_id": "scene-1",
        "candidate_id": "candidate-1",
        "input_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "transform_family": "global_lut",
        "failure_family": "none_or_fixture",
        "reviewer": {
            "reviewer_id_hash": f"{reviewer:064x}",
            "reviewer_kind": reviewer_kind,
            "panel_id": "panel-1" if reviewer_kind == "human_initial" else "panel-senior",
            "blind": True,
            "qualification_version": "qualification-v1",
            "primary_evidence": reviewer_kind != "autonomous_vlm_secondary",
        },
        "viewing": {
            "full_resolution": True,
            "zoom_percent": 100,
            "input_visible": True,
            "effects_disabled": True,
            "display_protocol_id": "display-v1",
            "colour_state": "display_srgb",
        },
        "severity": {
            "label": label,
            "categories": ["neon_chroma_island_or_highlight_speckle"] if severe else [],
            "regions": [
                {
                    "region_id": "region-1",
                    "kind": "mask",
                    "artifact_category": "neon_chroma_island_or_highlight_speckle",
                    "asset_path": "masks/scene-1.png",
                    "asset_sha256": "c" * 64,
                }
            ] if severe else [],
            "notes": "fixture",
        },
        "style": None,
    }


def _split_row(record_id: str, split: str, suffix: str, *, origin: str = "new_manifest") -> dict:
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


def test_contract_and_json_schema_freeze_strict_vocabularies() -> None:
    contract = load_contract(CONTRACT)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen_before_schema_implementation"
    assert contract["current_evidence_assignment"]["rf2_c0_spektrafilm_outputs"] == "A0_only"
    assert contract["preregistration_planning_values"]["binding_sample_size"] is None
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_id"]["const"] == SCHEMA_ID


def test_annotation_rejects_unknown_keys_vlm_escalation_and_seen_hidden_evidence() -> None:
    assert validate_annotation(_rating(1))["record_kind"] == "severity_rating"
    unknown = _rating(1)
    unknown["surprise"] = True
    with pytest.raises(FilmStyleSafeContractError, match="keys mismatch"):
        validate_annotation(unknown)
    vlm = _rating(20, reviewer_kind="autonomous_vlm_secondary")
    vlm["reviewer"]["primary_evidence"] = True
    with pytest.raises(FilmStyleSafeContractError, match="cannot be primary"):
        validate_annotation(vlm)
    hidden = _rating(1, split="A1", evidence_origin="rf2_c0_spektrafilm_outputs")
    with pytest.raises(FilmStyleSafeContractError, match="A0-only"):
        validate_annotation(hidden)


def test_severe_requires_category_and_region_and_style_requires_traceability() -> None:
    severe = _rating(1, "severe")
    severe["severity"]["regions"] = []
    with pytest.raises(FilmStyleSafeContractError, match="category and region"):
        validate_annotation(severe)
    style = _rating(1)
    style["record_kind"] = "style_rating"
    style["severity"] = None
    style["style"] = {
        "look_id": "film_stock:ektar100",
        "reference_board_sha256": "d" * 64,
        "film_origin_traceable": False,
        "look_adherence": "pass",
        "style_strength": 4,
        "keep_preference": "tie",
        "comparator_id": "safe-rich",
    }
    with pytest.raises(FilmStyleSafeContractError, match="cannot name a film stock"):
        validate_annotation(style)


def test_scene_aggregation_initial_pass_and_conservative_missing_panel() -> None:
    records = [_rating(1), _rating(2, "minor"), _rating(3)]
    report = aggregate_scene_severity(records, parent_scene_id="scene-1", candidate_id="candidate-1")
    assert report["protocol_adjudicated_severe"] == 0
    assert report["status"] == "initial_unanimous_nonsevere"
    escalated = [_rating(1), _rating(2, "uncertain"), _rating(3)]
    report = aggregate_scene_severity(escalated, parent_scene_id="scene-1", candidate_id="candidate-1")
    assert report["protocol_adjudicated_severe"] == 1
    assert report["status"] == "missing_or_incomplete_senior_panel"


def test_scene_aggregation_senior_majority_and_vlm_is_ignored() -> None:
    records = [_rating(1), _rating(2, "severe"), _rating(3)]
    records += [
        _rating(11, "safe", reviewer_kind="human_senior"),
        _rating(12, "minor", reviewer_kind="human_senior"),
        _rating(13, "severe", reviewer_kind="human_senior"),
        _rating(21, "severe", reviewer_kind="autonomous_vlm_secondary"),
    ]
    report = aggregate_scene_severity(records, parent_scene_id="scene-1", candidate_id="candidate-1")
    assert report["protocol_adjudicated_severe"] == 0
    assert report["status"] == "senior_majority_nonsevere"
    assert report["autonomous_records_ignored"] == 1


def test_split_manifest_rejects_lineage_seen_evidence_and_a1_family_leakage() -> None:
    valid = [_split_row("r-a0", "A0", "a0"), _split_row("r-a1", "A1", "b1")]
    assert validate_split_manifest(valid)["hidden_split_eligible"] is True
    lineage = copy.deepcopy(valid)
    lineage[1]["parent_scene_id"] = lineage[0]["parent_scene_id"]
    with pytest.raises(FilmStyleSafeContractError, match="lineage leakage"):
        validate_split_manifest(lineage)
    seen = [_split_row("r-seen", "A1", "c1", origin="known_id11_red_speckle")]
    with pytest.raises(FilmStyleSafeContractError, match="cannot enter hidden"):
        validate_split_manifest(seen)
    family = copy.deepcopy(valid)
    family[1]["transform_family"] = family[0]["transform_family"]
    with pytest.raises(FilmStyleSafeContractError, match="unseen-family"):
        validate_split_manifest(family)


def test_zero_event_power_is_planning_only_and_matches_one_percent_scale() -> None:
    worksheet = zero_event_power_worksheet(epsilon=0.01, expected_coverage=0.5)
    assert worksheet["zero_event_independent_accepted_scenes"] == 299
    assert worksheet["total_independent_scenes_before_adjustments"] == 598
    assert worksheet["binding_sample_size"] is None
    with pytest.raises(FilmStyleSafeContractError, match="probabilities"):
        zero_event_power_worksheet(epsilon=0.0)
