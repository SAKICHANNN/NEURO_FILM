"""Hash-bound runner for the fixed strong-style LUT champion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_fixed_strong_lut_champion import evaluate_fixed_strong_lut_champion


class FiveKFixedStrongLUTChampionRunError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load(root: Path, spec: Mapping[str, Any], key: str) -> dict[str, Any]:
    path = root / str(spec[key]); hash_key = f"{key}_sha256" if f"{key}_sha256" in spec else "sha256"
    if not path.is_file() or _sha256(path) != str(spec[hash_key]):
        raise FiveKFixedStrongLUTChampionRunError("parent evidence drift")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("router_training_allowed") is not False
        or config.get("confirmation_pixels_allowed") is not False
        or config.get("visual_review_allowed_only_after_automatic_pass") is not True
        or config.get("production_integration_allowed") is not False
    ):
        raise FiveKFixedStrongLUTChampionRunError("champion boundary drift")
    decision = _load(root, config["parent_bq9_decision"], "path")
    mature = _load(root, config["mature_baseline"], "config")
    manifest = _load(root, config["parent_manifest"], "path")
    if (
        decision.get("decision") != "close_selected_direction_strength_route_open_fixed_strong_medoid"
        or decision.get("fixed_strong_medoid_research_allowed") is not True
        or mature.get("operator", {}).get("grid_size") != 4
        or config["selection"].get("target_or_validation_used_for_selection") is not False
        or config["selection"].get("same_camera_group_rows_excluded_from_candidate_score") is not True
    ):
        raise FiveKFixedStrongLUTChampionRunError("ineligible parent")
    return {"mature_config": mature, "manifest": manifest}


def run_fixed_strong_lut_champion(
    *, root: Path, config: Mapping[str, Any], config_path: Path,
    output_path: Path, software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    variants = {}; artifacts = {}; output_path.parent.mkdir(parents=True, exist_ok=True)
    for name in config["target_variants"]:
        rows = load_split_population(validated["manifest"], split="development", target_variant=name, maximum_side=int(config["decode"]["maximum_side"]))
        result, coefficients = evaluate_fixed_strong_lut_champion(
            rows=rows, operator=validated["mature_config"]["operator"],
            split_spec=config["split"], selection=config["selection"],
            evaluation=config["evaluation"], samples_per_image=int(config["decode"]["evaluation_samples_per_image"]),
        )
        path = output_path.parent / f"{name}_coefficients.npy"
        with path.open("wb") as handle:
            np.save(handle, np.asarray(coefficients, dtype="<f8"), allow_pickle=False)
        artifacts[name] = {"filename": path.name, "sha256": _sha256(path), "shape": list(coefficients.shape)}
        variants[name] = result
    stable = {
        "parent_decision_sha256": config["parent_bq9_decision"]["sha256"],
        "parent_manifest_sha256": config["parent_manifest"]["sha256"],
        "confirmation_loaded": False, "coefficient_artifacts": artifacts,
        "variants": variants, "automatic_pass": all(value["automatic_pass"] for value in variants.values()),
    }
    report = {
        "schema": "neuro_film.u5_r2bq10_fivek_fixed_strong_lut_champion.v1",
        "experiment_id": config["experiment_id"], "software_commit": software_commit,
        "config_sha256": _sha256(config_path), **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "router_trained": False, "claim_ceiling": config["claim_ceiling"],
    }
    output_path.write_bytes(_canonical_bytes(report)); return report


__all__ = ["FiveKFixedStrongLUTChampionRunError", "run_fixed_strong_lut_champion", "validate_contract"]
