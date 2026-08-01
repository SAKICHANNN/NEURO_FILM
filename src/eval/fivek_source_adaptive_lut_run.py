"""Hash-bound runner for source-only strict-interior LUT prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_source_adaptive_lut import evaluate_source_adaptive_lut


class FiveKSourceAdaptiveLUTRunError(ValueError):
    """Raised when the frozen source-only LUT protocol drifts."""


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
        raise FiveKSourceAdaptiveLUTRunError("parent evidence drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FiveKSourceAdaptiveLUTRunError("parent must be an object")
    return value


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("confirmation_pixels_allowed") is not False
        or config.get("visual_review_allowed_only_after_automatic_pass") is not True
        or config.get("production_integration_allowed") is not False
        or config.get("film_or_stock_claim_allowed") is not False
    ):
        raise FiveKSourceAdaptiveLUTRunError("predictor boundary drift")
    parent = _load_hashed(root, config["parent_bq6_decision"], "path")
    nearest = _load_hashed(root, config["parent_nearest_decision"], "path")
    mature = _load_hashed(root, config["mature_baseline"], "config")
    manifest = _load_hashed(root, config["parent_manifest"], "path")
    model = config["model"]
    operator = mature.get("operator", {})
    if (
        parent.get("decision")
        != "pass_strict_interior_lut_capacity_open_source_only_routing"
        or parent.get("source_only_routing_research_allowed") is not True
        or nearest.get("selected_family")
        != config["parent_nearest_decision"]["required_selected_family"]
        or operator.get("grid_size") != 4
        or operator.get("boundary_epsilon") != 1.0 / 510.0
        or operator.get("node_coefficient_minimum") != -1.0
        or operator.get("node_coefficient_maximum") != 1.0
        or model.get("lut_basis_rank") != 8
        or model.get("ridge_alphas") != [1.0, 10.0, 100.0, 1000.0]
        or model.get("query_target_used_at_inference") is not False
        or model.get("confirmation_used_for_fit_or_selection") is not False
        or model.get("direct_final_rgb_prediction") is not False
        or manifest.get("split_summary", {}).get("selection_used_target_or_pixels")
        is not False
    ):
        raise FiveKSourceAdaptiveLUTRunError("ineligible parent or model")
    return {
        "parent": parent,
        "nearest": nearest,
        "mature_config": mature,
        "manifest": manifest,
    }


def run_source_adaptive_lut(
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
    for name in config["target_variants"]:
        rows = load_split_population(
            validated["manifest"],
            split="development",
            target_variant=name,
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        result, coefficients = evaluate_source_adaptive_lut(
            rows=rows,
            operator=validated["mature_config"]["operator"],
            descriptor_spec=config["descriptor"],
            model_spec=config["model"],
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
            "array_sha256": result["metrics"].get("coefficient_bank_sha256"),
        }
        variants[name] = result
    stable = {
        "parent_decision_sha256": config["parent_bq6_decision"]["sha256"],
        "mature_config_sha256": config["mature_baseline"]["config_sha256"],
        "parent_manifest_sha256": config["parent_manifest"]["sha256"],
        "confirmation_loaded": False,
        "coefficient_artifacts": coefficient_artifacts,
        "variants": variants,
        "automatic_pass": all(value["automatic_pass"] for value in variants.values()),
    }
    report = {
        "schema": "neuro_film.u5_r2bq7_fivek_source_adaptive_lut.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "learned_final_rgb_used": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKSourceAdaptiveLUTRunError",
    "run_source_adaptive_lut",
    "validate_contract",
]
