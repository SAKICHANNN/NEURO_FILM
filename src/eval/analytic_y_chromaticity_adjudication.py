"""CB50 autonomous severe-review and blind development adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json

SCHEMA = "neuro_film.u5_r2cb50_analytic_y_chromaticity_adjudication_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb50_analytic_y_chromaticity_adjudication_report.v1"
EXPERIMENT_ID = "U5.R2CB50A"


class AnalyticYChromaticityAdjudicationError(RuntimeError):
    pass


def _load_exact(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise AnalyticYChromaticityAdjudicationError(f"hash drift: {path}")
    return json.loads(payload)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise AnalyticYChromaticityAdjudicationError("CB50A contract drift")
    return payload


def adjudicate(config: dict[str, Any], root: Path, report_path: Path) -> dict[str, Any]:
    report = _load_exact(report_path, config["evidence"]["report_sha256"])
    observations = _load_exact(
        root / config["evidence"]["observations_path"],
        config["evidence"]["observations_sha256"],
    )
    if (
        not report.get("automatic_pass")
        or report.get("stable_evidence_id") != config["evidence"]["stable_evidence_id"]
        or observations.get("report_sha256") != config["evidence"]["report_sha256"]
        or observations.get("mapping_opened") is not False
        or observations["source_order"] != [row["id"] for row in report["rows"]]
    ):
        raise AnalyticYChromaticityAdjudicationError("CB50 evidence binding drift")
    round_candidate_choices: list[int] = []
    source_candidate_choices = {
        source_id: 0 for source_id in observations["source_order"]
    }
    for round_row in observations["rounds"]:
        round_key = str(round_row["round"])
        expected_sheet = next(
            row["sha256"]
            for row in report["blind_sheets"]
            if row["round"] == round_row["round"]
        )
        if round_row["sheet_sha256"] != expected_sheet:
            raise AnalyticYChromaticityAdjudicationError("CB50 sheet binding drift")
        candidate_choices = 0
        for source_id, choice in zip(
            observations["source_order"], round_row["choices"], strict=True
        ):
            mapping = report["sealed_mappings"][round_key][source_id]
            picked = mapping[0 if choice == "A" else 1]
            if picked == "candidate":
                candidate_choices += 1
                source_candidate_choices[source_id] += 1
        round_candidate_choices.append(candidate_choices)
    gates = config["gates"]
    round_wins = sum(
        value > len(observations["source_order"]) / 2
        for value in round_candidate_choices
    )
    aggregate = sum(round_candidate_choices)
    source_majorities = sum(value >= 2 for value in source_candidate_choices.values())
    severe = observations["severe_review"]["confirmed_severe_artifact_count"]
    checks = {
        "automatic": True,
        "severe_artifact_veto": severe
        <= gates["maximum_confirmed_severe_artifact_count"],
        "round_wins": round_wins >= gates["minimum_candidate_round_wins"],
        "aggregate_choices": aggregate >= gates["minimum_candidate_aggregate_choices"],
        "source_majorities": source_majorities
        >= gates["minimum_candidate_source_majorities"],
    }
    passed = all(checks.values())
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "automatic_report_sha256": config["evidence"]["report_sha256"],
        "automatic_stable_evidence_id": config["evidence"]["stable_evidence_id"],
        "observations_sha256": config["evidence"]["observations_sha256"],
        "round_candidate_choices": round_candidate_choices,
        "candidate_round_wins": round_wins,
        "candidate_aggregate_choices": aggregate,
        "candidate_source_majorities": source_majorities,
        "source_candidate_choices": source_candidate_choices,
        "confirmed_severe_artifact_count": severe,
        "checks": checks,
        "pass": passed,
        "decision": (
            "pass_cb50_development_open_source_disjoint_confirmation"
            if passed
            else "close_cb50_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    result["stable_evidence_id"] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


__all__ = ["adjudicate", "load_contract"]
