"""Resolve the frozen BL6 two-arm blind comparison without refitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CANDIDATE = "fixed_bl5_strict_interior_sigmoid"
REFERENCE = "fixed_ao6_colour_only_t15_c35"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adjudicate_strict_interior_ood(
    *,
    config: dict[str, Any],
    observations_path: Path,
    run_a: Path,
    run_b: Path,
) -> dict[str, Any]:
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    if observations.get("revealed_mapping_at_capture") is not False:
        raise ValueError("blind observations were not sealed before reveal")
    if int(observations.get("severe_artifact_observations", -1)) != 0:
        raise ValueError("blind severe-artifact veto failed")

    report_a_path = run_a / "report.json"
    report_b_path = run_b / "report.json"
    if report_a_path.read_bytes() != report_b_path.read_bytes():
        raise ValueError("two-run report identity failed")
    report = json.loads(report_a_path.read_text(encoding="utf-8"))
    if not report.get("automatic_gate_pass"):
        raise ValueError("automatic gate did not open visual adjudication")

    expected_sources = list(observations["source_order"])
    report_by_source = {row["source_id"]: row for row in report["rows"]}
    if list(report_by_source) != expected_sources:
        raise ValueError("report population or order drift")
    output_hashes: dict[str, str] = {}
    for source_id in expected_sources:
        expected = report_by_source[source_id]["output_sha256"]
        relative = Path(report_by_source[source_id]["output"])
        actual_a = _sha256(run_a / relative)
        actual_b = _sha256(run_b / relative)
        if actual_a != expected or actual_b != expected:
            raise ValueError(f"render identity drift: {source_id}")
        output_hashes[source_id] = expected

    candidate_choices = 0
    candidate_round_wins = 0
    decoded_rounds = []
    mapping_hashes = []
    for round_row in observations["rounds"]:
        round_index = int(round_row["round"])
        mapping_name = f"blind_round_{round_index}_mapping.json"
        mapping_a_path = run_a / "blind" / mapping_name
        mapping_b_path = run_b / "blind" / mapping_name
        if mapping_a_path.read_bytes() != mapping_b_path.read_bytes():
            raise ValueError("two-run mapping identity failed")
        mapping = json.loads(mapping_a_path.read_text(encoding="utf-8"))
        if [row["source_id"] for row in mapping] != expected_sources:
            raise ValueError("blind mapping population or order drift")
        choices = list(round_row["choices"])
        if len(choices) != len(mapping) or not set(choices) <= {"A", "B"}:
            raise ValueError("invalid blind choice inventory")
        sheet_hashes = []
        for part_index, expected_hash in enumerate(
            round_row["sheet_sha256"], start=1
        ):
            sheet_name = f"blind_round_{round_index}_part_{part_index}.png"
            sheet_a = run_a / "blind" / sheet_name
            sheet_b = run_b / "blind" / sheet_name
            actual_a = _sha256(sheet_a)
            actual_b = _sha256(sheet_b)
            if actual_a != expected_hash or actual_b != expected_hash:
                raise ValueError("blind sheet identity drift")
            sheet_hashes.append(expected_hash)
        decoded = []
        candidate_count = 0
        for row, choice in zip(mapping, choices, strict=True):
            selected = row[choice]
            if selected not in {CANDIDATE, REFERENCE}:
                raise ValueError("unexpected blind arm")
            candidate_count += int(selected == CANDIDATE)
            decoded.append(
                {
                    "source_id": row["source_id"],
                    "anonymous_choice": choice,
                    "selected_arm": selected,
                }
            )
        candidate_choices += candidate_count
        round_pass = candidate_count > len(mapping) - candidate_count
        candidate_round_wins += int(round_pass)
        mapping_hashes.append(_sha256(mapping_a_path))
        decoded_rounds.append(
            {
                "round": round_index,
                "candidate_choices": candidate_count,
                "reference_choices": len(mapping) - candidate_count,
                "candidate_won_round": round_pass,
                "sheet_sha256": sheet_hashes,
                "decoded": decoded,
            }
        )

    protocol = config["blind_protocol"]
    gate_checks = {
        "automatic_gate": True,
        "two_run_identity": True,
        "severe_artifact_veto": True,
        "minimum_candidate_round_wins": candidate_round_wins
        >= int(protocol["minimum_candidate_round_wins"]),
        "minimum_candidate_aggregate_choices": candidate_choices
        >= int(protocol["minimum_candidate_aggregate_choices"]),
    }
    passed = all(gate_checks.values())
    core = {
        "schema": "neuro_film.u5_r2bl6_filmmatch_strict_interior_ood_decision.v1",
        "experiment_id": config["experiment_id"],
        "report_sha256": _sha256(report_a_path),
        "report_stable_evidence_id": report["stable_evidence_id"],
        "blind_observations_sha256": _sha256(observations_path),
        "blind_mapping_sha256": mapping_hashes,
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "candidate_round_wins": candidate_round_wins,
        "candidate_aggregate_choices": candidate_choices,
        "aggregate_choice_denominator": int(
            protocol["aggregate_choice_denominator"]
        ),
        "confirmed_severe_artifact_count": 0,
        "gates": gate_checks,
        "passed": passed,
        "decision": (
            "retain_strict_interior_as_high_priority_internal_challenger"
            if passed
            else "retain_ao6_incumbent"
        ),
        "product_integration_opened": False,
        "independent_confirmation_opened": passed,
        "claim_ceiling": config["claim_ceiling"],
        "output_sha256": output_hashes,
        "rounds": decoded_rounds,
    }
    core["stable_evidence_id"] = _canonical_sha256(core)
    return core


__all__ = ["adjudicate_strict_interior_ood"]
