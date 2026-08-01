"""Hash-bound runner for the fixed selected-LUT path Oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_selected_lut_path_oracle import evaluate_selected_lut_path_oracle


class FiveKSelectedLUTPathOracleRunError(ValueError):
    """Raised when selected-path evidence drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load(root: Path, spec: Mapping[str, Any], key: str) -> dict[str, Any]:
    path = root / str(spec[key])
    hash_key = f"{key}_sha256" if f"{key}_sha256" in spec else "sha256"
    if not path.is_file() or _sha256(path) != str(spec[hash_key]):
        raise FiveKSelectedLUTPathOracleRunError("parent evidence drift")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("router_or_strength_training_allowed") is not False
        or config.get("confirmation_pixels_allowed") is not False
        or config.get("visual_review_allowed") is not False
        or config.get("production_integration_allowed") is not False
    ):
        raise FiveKSelectedLUTPathOracleRunError("Oracle boundary drift")
    decision = _load(root, config["parent_bq8_decision"], "path")
    parent_config = _load(root, config["parent_bq8"], "config")
    report = _load(root, config["parent_bq8"], "report")
    manifest = _load(root, config["parent_manifest"], "path")
    if (
        decision.get("decision") != "close_direct_hard_top1_open_selected_path_oracle"
        or decision.get("selected_path_oracle_allowed") is not True
        or report.get("stable_evidence_id") != config["parent_bq8"]["stable_evidence_id"]
        or report.get("automatic_pass") is not False
        or parent_config.get("model", {}).get("dense_blending_allowed") is not False
        or config["path"].get("selected_case_identity_refit_or_change_allowed") is not False
    ):
        raise FiveKSelectedLUTPathOracleRunError("ineligible parent")
    return {"parent_config": parent_config, "report": report, "manifest": manifest}


def run_selected_lut_path_oracle(
    *, root: Path, config: Mapping[str, Any], config_path: Path,
    output_path: Path, software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    parent_report_path = root / config["parent_bq8"]["report"]
    variants = {}
    for name in config["parent_bq8"].get("variants", validated["parent_config"]["target_variants"]):
        artifact = validated["report"]["coefficient_artifacts"][name]
        artifact_path = parent_report_path.parent / artifact["filename"]
        if not artifact_path.is_file() or _sha256(artifact_path) != artifact["sha256"]:
            raise FiveKSelectedLUTPathOracleRunError("coefficient artifact drift")
        coefficients = np.load(artifact_path, allow_pickle=False)
        rows = load_split_population(
            validated["manifest"], split="development", target_variant=name,
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        variants[name] = evaluate_selected_lut_path_oracle(
            rows=rows, coefficients=coefficients,
            parent_variant=validated["report"]["variants"][name],
            split_spec=validated["parent_config"]["model"],
            path_spec=config["path"], evaluation=config["evaluation"],
            samples_per_image=int(config["decode"]["evaluation_samples_per_image"]),
        )
    stable = {
        "parent_report_sha256": config["parent_bq8"]["report_sha256"],
        "fixed_selected_case_identities": True,
        "confirmation_loaded": False,
        "variants": variants,
        "automatic_pass": all(value["automatic_pass"] for value in variants.values()),
    }
    report = {
        "schema": "neuro_film.u5_r2bq9_fivek_selected_lut_path_oracle.v1",
        "experiment_id": config["experiment_id"], "software_commit": software_commit,
        "config_sha256": _sha256(config_path), **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "router_or_strength_trained": False, "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = ["FiveKSelectedLUTPathOracleRunError", "run_selected_lut_path_oracle", "validate_contract"]
