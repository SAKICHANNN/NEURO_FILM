"""Adjudicate BL11 automatic and autonomous visual evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.filmmatch_strict_interior_fresh_confirmation import AO6_ARM
from src.eval.global_frontier import sha256_file


CANDIDATE = "ao6_lightness_bl5_chroma_analytical_safe"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def adjudicate_luma_preserving_chroma(
    *,
    config: dict[str, Any],
    observations_path: Path,
    run_a: Path,
    run_b: Path,
    blind_dir: Path,
) -> dict[str, Any]:
    if (run_a / "report.json").read_bytes() != (run_b / "report.json").read_bytes():
        raise ValueError("BL11 two-run report identity failed")
    report = json.loads((run_a / "report.json").read_text(encoding="utf-8"))
    if not report.get("automatic_gate_pass") or not report.get("visual_review_opened"):
        raise ValueError("BL11 automatic gate did not open visual review")
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if (
        observations.get("mapping_revealed_before_choices_fixed") is not False
        or observations.get("population_preference_claimed")
        or observations.get("product_promotion_allowed")
    ):
        raise ValueError("BL11 observation boundary drift")
    mapping_path = blind_dir / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_order = observations["source_order"]
    choices = observations["choices"]
    if (
        [row["source_id"] for row in mapping] != source_order
        or len(choices) != len(mapping)
        or not set(choices) <= {"A", "B"}
        or len(report["rows"]) != config["automatic_gate"]["expected_outputs"]
    ):
        raise ValueError("BL11 visual population drift")
    for row in report["rows"]:
        expected = row["output_sha256"]
        if (
            sha256_file(run_a / row["output"]) != expected
            or sha256_file(run_b / row["output"]) != expected
        ):
            raise ValueError("BL11 output identity drift")
    decoded: list[dict[str, str]] = []
    candidate_choices = 0
    for row, choice in zip(mapping, choices, strict=True):
        selected = row[choice]
        if selected not in {AO6_ARM, CANDIDATE}:
            raise ValueError("unexpected BL11 visual arm")
        candidate_choices += int(selected == CANDIDATE)
        decoded.append(
            {
                "source_id": row["source_id"],
                "anonymous_choice": choice,
                "selected_arm": selected,
            }
        )
    severe = int(observations["confirmed_severe_artifact_count"])
    candidate_majority = candidate_choices > len(mapping) - candidate_choices
    core = {
        "schema": "neuro_film.u5_r2bl11_filmmatch_luma_preserving_chroma_transplant_decision.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": report["config_sha256"],
        "report_sha256": sha256_file(run_a / "report.json"),
        "report_stable_evidence_id": report["stable_evidence_id"],
        "blind_mapping_sha256": sha256_file(mapping_path),
        "blind_observations_sha256": sha256_file(observations_path),
        "automatic_gate_pass": True,
        "two_run_identity": True,
        "confirmed_severe_artifact_count": severe,
        "severe_artifact_veto_pass": severe == 0,
        "candidate_choices": candidate_choices,
        "ao6_choices": len(mapping) - candidate_choices,
        "choice_denominator": len(mapping),
        "candidate_won_descriptive_majority": candidate_majority,
        "visual_evidence_grade": "autonomous_development_non_precommitted",
        "decoded_choices": decoded,
        "decision": (
            "retain_bl11_as_development_challenger_require_new_population"
            if severe == 0 and candidate_majority
            else "close_bl11_preference_retain_ao6"
        ),
        "interpretation": (
            "BL11 cleanly removes the BL5 tone failure and preserves its chroma request, "
            "but the transplant loses the descriptive autonomous preference comparison "
            "because it usually reduces AO6's salient cyan-magenta colour separation."
        ),
        "threshold_retuning_allowed": False,
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


__all__ = ["adjudicate_luma_preserving_chroma"]
