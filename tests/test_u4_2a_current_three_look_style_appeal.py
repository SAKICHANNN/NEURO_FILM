from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.current_three_look_style_appeal import (
    contact_sheet_bytes,
    evaluate_observations,
    load_parent_rows,
    sha256_bytes,
)
from src.filmcase.vision_audit import build_blind_audit

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


def test_config_has_frozen_no_rescue_claim_boundary() -> None:
    assert CONFIG["blind_protocol"]["expected_observations"] == 108
    assert CONFIG["gates"]["required_passing_named_looks"] == 3
    assert "look_amount_or_parameter_tuning" in CONFIG["forbidden_rescue"]
    assert CONFIG["training_allowed"] is False
    assert CONFIG["operator_fitting_allowed"] is False
    assert sha256_bytes(b"stable") == sha256_bytes(b"stable")
