"""Hash-bound runner for the mature strict-interior LUT case-bank baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_strict_interior_lut_casebank import (
    evaluate_strict_interior_lut_casebank,
)


class FiveKStrictInteriorLUTCasebankRunError(ValueError):
    """Raised when the frozen strict-interior LUT baseline drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _load_hashed(root: Path, spec: Mapping[str, Any], key: str) -> dict[str, Any]:
    path = root / str(spec[key])
    hash_key = f"{key}_sha256" if f"{key}_sha256" in spec else "sha256"
    if not path.is_file() or _sha256(path) != str(spec[hash_key]):
        raise FiveKStrictInteriorLUTCasebankRunError("parent evidence drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FiveKStrictInteriorLUTCasebankRunError("parent must be an object")
    return value


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("source_selector_training_allowed") is not False
        or config.get("confirmation_pixels_allowed") is not False
        or config.get("visual_review_allowed") is not False
        or config.get("production_integration_allowed") is not False
        or config.get("film_or_stock_claim_allowed") is not False
    ):
        raise FiveKStrictInteriorLUTCasebankRunError("baseline boundary drift")
    parent = _load_hashed(root, config["parent_bq5_decision"], "path")
    mature_config = _load_hashed(root, config["mature_baseline"], "config")
    mature_decision = _load_hashed(root, config["mature_baseline"], "decision")
    manifest = _load_hashed(root, config["parent_manifest"], "path")
    operator = mature_config.get("operator", {})
    if (
        parent.get("decision")
        != "close_mature_projection_curve_casebank_on_style_and_safe_dose"
        or mature_decision.get("status")
        != config["mature_baseline"]["required_status"]
        or operator.get("family")
        != "strict_epsilon_interior_headroom_trilinear_residual_lut"
        or operator.get("grid_size") != 4
        or operator.get("boundary_epsilon") != 1.0 / 510.0
        or operator.get("node_coefficient_minimum") != -1.0
        or operator.get("node_coefficient_maximum") != 1.0
        or operator.get("hard_output_clipping_allowed") is not False
        or operator.get("post_operator_gamut_scaling_allowed") is not False
        or manifest.get("split_summary", {}).get("selection_used_target_or_pixels")
        is not False
    ):
        raise FiveKStrictInteriorLUTCasebankRunError("ineligible parent branch")
    return {
        "parent": parent,
        "mature_config": mature_config,
        "mature_decision": mature_decision,
        "manifest": manifest,
    }


def run_strict_interior_lut_casebank(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    variants = {}
    coefficient_artifacts = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for name in config["variants"]:
        rows = load_split_population(
            validated["manifest"],
            split="development",
            target_variant=name,
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        result, coefficients = evaluate_strict_interior_lut_casebank(
            rows=rows,
            operator=validated["mature_config"]["operator"],
            split_spec=config["split"],
            evaluation=config["evaluation"],
            samples_per_image=int(config["decode"]["evaluation_samples_per_image"]),
        )
        artifact_path = output_path.parent / f"{name}_coefficients.npy"
        with artifact_path.open("wb") as handle:
            np.save(handle, np.asarray(coefficients, dtype="<f8"), allow_pickle=False)
        coefficient_artifacts[name] = {
            "filename": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "shape": list(coefficients.shape),
            "array_sha256": result["metrics"]["coefficient_bank_sha256"],
        }
        variants[name] = result
    stable = {
        "mature_config_sha256": config["mature_baseline"]["config_sha256"],
        "parent_manifest_sha256": config["parent_manifest"]["sha256"],
        "confirmation_loaded": False,
        "coefficient_artifacts": coefficient_artifacts,
        "variants": variants,
        "automatic_pass": all(value["automatic_pass"] for value in variants.values()),
    }
    report = {
        "schema": "neuro_film.u5_r2bq6_fivek_strict_interior_lut_casebank.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "source_selector_trained": False,
        "learned_final_rgb_used": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKStrictInteriorLUTCasebankRunError",
    "run_strict_interior_lut_casebank",
    "validate_contract",
]
