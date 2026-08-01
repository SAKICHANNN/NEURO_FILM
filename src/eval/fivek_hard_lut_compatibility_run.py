"""Hash-bound runner for hard source/case LUT compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_hard_lut_compatibility import evaluate_hard_lut_compatibility


class FiveKHardLUTCompatibilityRunError(ValueError):
    """Raised when the frozen hard LUT routing contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_hashed(root: Path, spec: Mapping[str, Any], key: str) -> dict[str, Any]:
    path = root / str(spec[key])
    hash_key = f"{key}_sha256" if f"{key}_sha256" in spec else "sha256"
    if not path.is_file() or _sha256(path) != str(spec[hash_key]):
        raise FiveKHardLUTCompatibilityRunError("parent evidence drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FiveKHardLUTCompatibilityRunError("parent must be an object")
    return value


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    if (
        config.get("schema_version") != 1
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("confirmation_pixels_allowed") is not False
        or config.get("visual_review_allowed_only_after_automatic_pass") is not True
        or config.get("production_integration_allowed") is not False
        or config.get("film_or_stock_claim_allowed") is not False
        or model.get("query_target_used_at_inference") is not False
        or model.get("operator_parameters_used_at_inference") is not False
        or model.get("dense_blending_allowed") is not False
    ):
        raise FiveKHardLUTCompatibilityRunError("hard router boundary drift")
    parent = _load_hashed(root, config["parent_bq7_decision"], "path")
    mature = _load_hashed(root, config["mature_baseline"], "config")
    manifest = _load_hashed(root, config["parent_manifest"], "path")
    operator = mature.get("operator", {})
    if (
        parent.get("decision")
        != "close_dense_lut_prediction_on_style_open_hard_case_routing"
        or parent.get("hard_case_routing_research_allowed") is not True
        or operator.get("grid_size") != 4
        or operator.get("boundary_epsilon") != 1.0 / 510.0
        or operator.get("node_coefficient_minimum") != -1.0
        or operator.get("node_coefficient_maximum") != 1.0
        or manifest.get("split_summary", {}).get("selection_used_target_or_pixels")
        is not False
    ):
        raise FiveKHardLUTCompatibilityRunError("ineligible parent")
    return {"parent": parent, "mature_config": mature, "manifest": manifest}


def run_hard_lut_compatibility(
    *, root: Path, config: Mapping[str, Any], config_path: Path,
    output_path: Path, software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    variants = {}
    artifacts = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for name in config["target_variants"]:
        rows = load_split_population(
            validated["manifest"], split="development", target_variant=name,
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        result, coefficients = evaluate_hard_lut_compatibility(
            rows=rows,
            operator=validated["mature_config"]["operator"],
            descriptor_spec=config["descriptor"],
            model_spec=config["model"],
            selector_spec=config["selector"],
            evaluation=config["evaluation"],
            samples_per_image=int(config["decode"]["evaluation_samples_per_image"]),
        )
        path = output_path.parent / f"{name}_coefficients.npy"
        with path.open("wb") as handle:
            np.save(handle, np.asarray(coefficients, dtype="<f8"), allow_pickle=False)
        artifacts[name] = {"filename": path.name, "sha256": _sha256(path), "shape": list(coefficients.shape)}
        variants[name] = result
    stable = {
        "parent_decision_sha256": config["parent_bq7_decision"]["sha256"],
        "parent_manifest_sha256": config["parent_manifest"]["sha256"],
        "confirmation_loaded": False,
        "coefficient_artifacts": artifacts,
        "variants": variants,
        "automatic_pass": all(value["automatic_pass"] for value in variants.values()),
    }
    report = {
        "schema": "neuro_film.u5_r2bq8_fivek_hard_lut_compatibility.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "dense_blending_used": False,
        "learned_final_rgb_used": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = ["FiveKHardLUTCompatibilityRunError", "run_hard_lut_compatibility", "validate_contract"]
