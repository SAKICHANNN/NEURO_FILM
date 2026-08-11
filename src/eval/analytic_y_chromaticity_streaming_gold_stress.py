"""CB55 exact streamed replay of the frozen CB52 gold/stress evidence."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any

from src.eval.analytic_y_chromaticity_gold_stress import evaluate_with_selector
from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file

SCHEMA = "neuro_film.u5_r2cb55_analytic_y_chromaticity_streaming_gold_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb55_analytic_y_chromaticity_streaming_gold_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB55"


class AnalyticYChromaticityStreamingGoldStressError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise AnalyticYChromaticityStreamingGoldStressError("CB55 contract drift")
    return payload


def evaluate(
    contract: dict[str, Any],
    root: Path,
    output_dir: Path,
    *,
    baseline_report_path: Path,
) -> dict[str, Any]:
    parents = contract["parents"]
    for path_key, sha_key in (
        ("cb54_decision_path", "cb54_decision_sha256"),
        ("cb52_contract_path", "cb52_contract_sha256"),
    ):
        if hash_file(root / parents[path_key]) != parents[sha_key]:
            raise AnalyticYChromaticityStreamingGoldStressError(
                f"CB55 parent drift: {path_key}"
            )
    decision = json.loads(
        (root / parents["cb54_decision_path"]).read_text(encoding="utf-8")
    )
    if decision.get("decision") != parents["cb54_required_decision"]:
        raise AnalyticYChromaticityStreamingGoldStressError("CB54 decision drift")
    if hash_file(baseline_report_path) != parents["cb52_baseline_report_sha256"]:
        raise AnalyticYChromaticityStreamingGoldStressError("CB52 baseline report drift")
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if baseline.get("stable_evidence_id") != parents["cb52_baseline_stable_evidence_id"]:
        raise AnalyticYChromaticityStreamingGoldStressError("CB52 baseline identity drift")
    cb52_config = json.loads(
        (root / parents["cb52_contract_path"]).read_text(encoding="utf-8")
    )
    scratch_root = output_dir.parent / (output_dir.name + "-cb55-scratch")
    scratch_root.mkdir(parents=True, exist_ok=False)
    try:
        report = evaluate_with_selector(
            cb52_config,
            root,
            output_dir,
            selector=partial(
                select_analytic_y_chromaticity_candidate_streamed,
                row_chunk=int(contract["execution"]["row_chunk"]),
                scratch_root=scratch_root,
            ),
            report_schema=REPORT_SCHEMA,
            experiment_id=EXPERIMENT_ID,
            contract_filename="u5_r2cb52_analytic_y_chromaticity_gold_stress_v1.json",
        )
        residue = sorted(path.name for path in scratch_root.iterdir())
    finally:
        if scratch_root.exists() and not any(scratch_root.iterdir()):
            scratch_root.rmdir()
    baseline_rows = {row["id"]: row for row in baseline["rows"]}
    current_rows = {row["id"]: row for row in report["rows"]}
    fact_keys = (
        "characteristic_strength",
        "fraction_gamut_scale_below_0p8",
        "global_dose",
        "maximum_luminance_error",
        "median_gamut_scale",
        "selected_gradient_ratio",
        "selected_lstar_inversion_fraction",
        "selected_new_boundary_fraction",
    )
    matching_hashes = sum(
        current_rows[key]["output_sha256"] == baseline_rows[key]["output_sha256"]
        for key in sorted(current_rows)
    )
    matching_facts = sum(
        all(current_rows[key][field] == baseline_rows[key][field] for field in fact_keys)
        for key in sorted(current_rows)
    )
    metrics_exact = report["metrics"] == baseline["metrics"]
    checks_exact = report["checks"] == baseline["checks"]
    execution = contract["execution"]
    replay_checks = {
        "source_count": len(current_rows) == int(execution["required_source_count"]),
        "output_hashes": matching_hashes
        == int(execution["required_output_hash_match_count"]),
        "row_facts": matching_facts
        == int(execution["required_row_fact_match_count"]),
        "metrics": metrics_exact,
        "checks": checks_exact,
        "scratch_cleanup": not residue and not scratch_root.exists(),
    }
    report["cb55_replay"] = {
        "baseline_report_sha256": hash_file(baseline_report_path),
        "matching_output_hash_count": matching_hashes,
        "matching_row_fact_count": matching_facts,
        "metrics_exact": metrics_exact,
        "checks_exact": checks_exact,
        "scratch_residue": residue,
    }
    report["checks"].update({f"cb55_{key}": value for key, value in replay_checks.items()})
    report["automatic_pass"] = all(report["checks"].values())
    report["visual_review_status"] = "reused_exact_cb52_bytes_no_new_review"
    report["decision"] = (
        "pass_cb55_streamed_gold_stress_exact_replay"
        if report["automatic_pass"]
        else "close_cb55_streamed_replay_without_rescue"
    )
    report["claim_ceiling"] = contract["claim_ceiling"]
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = ["evaluate", "load_contract"]
