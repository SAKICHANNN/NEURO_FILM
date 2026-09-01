from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.eval.current_three_look_style_appeal import (
    canonical_bytes,
    contact_sheet_bytes,
    evaluate_observations,
    load_parent_rows,
    materialize_blind_package,
    sha256_bytes,
)
from src.filmcase.vision_audit import build_blind_audit
from src.inference.product_desktop import product_output_format

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u4_2a_current_three_look_style_appeal_v1.json").read_text(
        encoding="utf-8"
    )
)


def _observations(plan, *, weak_look: str | None = None):
    rows = []
    for mapping in plan.mapping:
        for label, candidate in mapping["label_to_candidate"].items():
            identity = candidate == "identity_product_path"
            weak = candidate == weak_look
            rows.append(
                {
                    "round": mapping["round"],
                    "sample_id": mapping["sample_id"],
                    "label": label,
                    "severe": "no",
                    "style_strength": 1 if identity or weak else 4,
                    "appeal": 3 if identity or weak else 4,
                }
            )
    return {
        "schema": "neuro_film.u4_2a_blind_observations.v1",
        "status": "OBSERVATIONS_FROZEN_BEFORE_MAPPING_REVEAL",
        "mapping_read_or_reconstructed_before_freeze": False,
        "reviewer_class": "autonomous_visual_evidence",
        "score_rubric_sha256": sha256_bytes(canonical_bytes(CONFIG["score_rubric"])),
        "public_manifest_sha256": "0" * 64,
        "build_scientific_identity": "1" * 64,
        "review_media_scope": "public_review_sheets_only",
        "review_sheet_read_count": 27,
        "review_sheet_sha256s": [str(index) * 64 for index in range(1, 4)],
        "non_review_media_reads_before_freeze": {
            "candidate_images": 0,
            "identity_images": 0,
            "source_images": 0,
            "recipes": 0,
            "build_report": 0,
            "private_mapping": 0,
        },
        "observations": rows,
    }


def test_parent_report_and_final_adjudication_are_separately_locked() -> None:
    rows, evidence = load_parent_rows(CONFIG, ROOT)
    assert len(rows) == 27
    assert evidence["status"] == CONFIG["parents"]["u4_1a_evidence"]["required_status"]
    assert CONFIG["parents"]["u4_1a_report"]["required_status"] != evidence["status"]


def test_contact_sheet_is_deterministic_and_contains_no_candidate_names() -> None:
    images = [
        (label, np.full((9, 13, 3), index * 40, dtype=np.uint8))
        for index, label in enumerate("ABCD", 1)
    ]
    first = contact_sheet_bytes(round_index=1, sample_id="01", labeled_images=images)
    second = contact_sheet_bytes(round_index=1, sample_id="01", labeled_images=images)
    assert first == second
    assert first.startswith(b"\x89PNG")
    assert b"velvia" not in first and b"portra" not in first and b"ektar" not in first


def test_all_three_looks_must_independently_pass() -> None:
    plan = build_blind_audit(
        CONFIG["population"]["sample_ids"],
        CONFIG["arms"],
        seed=CONFIG["blind_protocol"]["seed"],
    )
    passed = evaluate_observations(CONFIG, plan, _observations(plan))
    assert passed["portfolio_pass"] is True
    assert passed["passing_named_looks"] == 3

    failed = evaluate_observations(
        CONFIG, plan, _observations(plan, weak_look="portra_400")
    )
    assert failed["portfolio_pass"] is False
    assert failed["passing_named_looks"] == 2
    assert failed["look_results"]["portra_400"]["pass"] is False


def test_observation_freeze_and_media_scope_are_mandatory() -> None:
    plan = build_blind_audit(
        CONFIG["population"]["sample_ids"],
        CONFIG["arms"],
        seed=CONFIG["blind_protocol"]["seed"],
    )
    invalid = _observations(plan)
    invalid["mapping_read_or_reconstructed_before_freeze"] = True
    try:
        evaluate_observations(CONFIG, plan, invalid)
    except ValueError as exc:
        assert "mapping access" in str(exc)
    else:
        raise AssertionError("mapping access before freeze must fail closed")

    invalid = _observations(plan)
    invalid["non_review_media_reads_before_freeze"]["identity_images"] = 1
    try:
        evaluate_observations(CONFIG, plan, invalid)
    except ValueError as exc:
        assert "media read boundary" in str(exc)
    else:
        raise AssertionError("non-review media access must fail closed")


def test_observation_provenance_expectations_come_from_frozen_config() -> None:
    plan = build_blind_audit(
        CONFIG["population"]["sample_ids"],
        CONFIG["arms"],
        seed=CONFIG["blind_protocol"]["seed"],
    )
    drifted = deepcopy(CONFIG)
    drifted["blind_protocol"]["reviewer_class"] = "different-reviewer-class"
    try:
        evaluate_observations(drifted, plan, _observations(plan))
    except ValueError as exc:
        assert "reviewer class" in str(exc)
    else:
        raise AssertionError("config-driven reviewer-class drift must fail closed")


def test_config_has_frozen_no_rescue_claim_boundary() -> None:
    assert CONFIG["blind_protocol"]["expected_observations"] == 108
    assert CONFIG["gates"]["required_passing_named_looks"] == 3
    assert "look_amount_or_parameter_tuning" in CONFIG["forbidden_rescue"]
    assert CONFIG["training_allowed"] is False
    assert CONFIG["operator_fitting_allowed"] is False
    assert CONFIG["score_rubric"]["style_strength"]["3"].startswith("clearly visible")
    assert CONFIG["score_rubric"]["appeal"]["3"].startswith("acceptable")
    assert product_output_format("png16").recipe_format == "PNG"
    assert sha256_bytes(b"stable") == sha256_bytes(b"stable")


def test_builder_contract_distinguishes_in_memory_mapping_from_reviewer_access() -> (
    None
):
    source = Path(materialize_blind_package.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    assert '"builder_mapping_used_in_memory": True' in source
    assert '"mapping_persisted": False' in source
