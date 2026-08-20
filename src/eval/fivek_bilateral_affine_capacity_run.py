"""Contract validation and deterministic runner for U5.R2BT1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.eval.fivek_bilateral_affine_capacity import evaluate_affine_capacity
from src.eval.fivek_bilateral_gain_capacity_run import _selected_rows, _sha256


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("status") != "contract_frozen_implementation_ready"
        or config.get("confirmation_pixels_allowed")
        or config.get("learned_final_rgb_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise ValueError("BT1 boundary drift")
    parent_spec = config["parents"]["bt0_decision"]
    parent_path = root / parent_spec["path"]
    if _sha256(parent_path) != parent_spec["sha256"]:
        raise ValueError("BT0 parent drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != parent_spec["required_decision"]:
        raise ValueError("BT0 decision drift")
    population = config["population"]
    manifest_path = root / population["manifest"]
    if _sha256(manifest_path) != population["manifest_sha256"]:
        raise ValueError("BT1 manifest drift")
    candidate = config["candidate"]
    if list(candidate["grid_shape"]) != [2, 2, 8] or int(candidate["parameter_count"]) != 384:
        raise ValueError("BT1 representation drift")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_bilateral_affine_capacity(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    manifest = validate_contract(root, config)
    variants = {}
    all_rows = []
    for variant in config["population"]["target_variants"]:
        result = evaluate_affine_capacity(
            _selected_rows(root, manifest, config, str(variant)), config
        )
        variants[str(variant)] = {
            "automatic_pass": result["automatic_pass"],
            "metrics": result["metrics"],
            "gates": result["gates"],
        }
        all_rows.extend(result["rows"])
    population_gates = {
        "target_variant_count": len(variants)
        == int(config["evaluation"]["automatic_gates"]["target_variant_count_exact"]),
        "target_variant_identity": sorted(variants)
        == sorted(map(str, config["population"]["target_variants"])),
    }
    body = {
        "schema": "neuro_film.u5_r2bt1_fivek_bilateral_affine_capacity_formal_report.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "software_commit": software_commit,
        "manifest_sha256": config["population"]["manifest_sha256"],
        "confirmation_rows_loaded": 0,
        "variants": variants,
        "population_gates": population_gates,
        "automatic_pass": all(population_gates.values())
        and all(row["automatic_pass"] for row in variants.values()),
        "rows": sorted(all_rows, key=lambda row: (row["target_variant"], row["pair_id"], row["fold"])),
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_body = {key: value for key, value in body.items() if key != "software_commit"}
    body["stable_evidence_id"] = hashlib.sha256(_canonical(stable_body)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(body))
    return body


__all__ = ["run_bilateral_affine_capacity", "validate_contract"]
