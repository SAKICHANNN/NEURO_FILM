"""Contract validation and deterministic runner for U5.R2BT0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.eval.fivek_bilateral_gain_capacity import evaluate_gain_capacity
from src.eval.fivek_casebank_oracle import load_split_population


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("status") != "contract_frozen_implementation_ready"
        or config.get("confirmation_pixels_allowed")
        or config.get("learned_final_rgb_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise ValueError("BT0 boundary drift")
    population = config["population"]
    manifest_path = root / population["manifest"]
    if _sha256(manifest_path) != population["manifest_sha256"]:
        raise ValueError("BT0 manifest drift")
    candidate = config["candidate"]
    control = config["controls"]["parameter_matched_global_rgb_lut"]
    if (
        list(candidate["grid_shape"]) != [4, 4, 8]
        or int(candidate["parameter_count"]) != 384
        or list(control["grid_shape"]) != [5, 5, 5]
        or int(control["parameter_count"]) != 375
    ):
        raise ValueError("BT0 representation drift")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _selected_rows(
    root: Path, manifest: Mapping[str, Any], config: Mapping[str, Any], variant: str
) -> list[dict[str, Any]]:
    rows = load_split_population(
        manifest,
        split="development",
        target_variant=variant,
        maximum_side=int(config["population"]["maximum_side"]),
    )
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        selected.setdefault(str(row["group"]), row)
    if len(selected) != int(config["population"]["source_count_exact"]):
        raise ValueError("BT0 selected group count drift")
    return [
        {**selected[group], "target_variant": variant} for group in sorted(selected)
    ]


def run_bilateral_gain_capacity(
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
        result = evaluate_gain_capacity(
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
        "schema": "neuro_film.u5_r2bt0_fivek_bilateral_gain_capacity_formal_report.v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "software_commit": software_commit,
        "manifest_sha256": config["population"]["manifest_sha256"],
        "confirmation_rows_loaded": 0,
        "variants": variants,
        "population_gates": population_gates,
        "automatic_pass": all(population_gates.values())
        and all(row["automatic_pass"] for row in variants.values()),
        "rows": sorted(
            all_rows,
            key=lambda row: (row["target_variant"], row["pair_id"], row["fold"]),
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_body = {
        key: value for key, value in body.items() if key != "software_commit"
    }
    body["stable_evidence_id"] = hashlib.sha256(_canonical(stable_body)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical(body))
    return body


__all__ = ["run_bilateral_gain_capacity", "validate_contract"]
