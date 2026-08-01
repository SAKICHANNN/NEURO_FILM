"""Hash-bound runner for source-only FiveK hard retrieval baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_source_hard_retrieval import (
    evaluate_confirmation_family,
    evaluate_development_family,
    prepare_development_evidence,
)


class FiveKSourceHardRetrievalRunError(ValueError):
    """Raised when a parent or source-only selector contract drifts."""


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


def _load_hashed(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    if not path.is_file() or _sha256(path) != str(spec["sha256"]):
        raise FiveKSourceHardRetrievalRunError("parent evidence drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FiveKSourceHardRetrievalRunError("parent must be an object")
    return payload


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("status") != "contract_frozen_implementation_ready"
        or config.get("confirmation_family_selection_allowed") is not False
        or config.get("dense_blending_allowed") is not False
        or config.get("learned_final_rgb_allowed") is not False
        or config.get("production_integration_allowed") is not False
        or config.get("film_or_stock_claim_allowed") is not False
    ):
        raise FiveKSourceHardRetrievalRunError("selector boundary drift")
    oracle = _load_hashed(root, config["parent_oracle"])
    manifest = _load_hashed(root, config["parent_manifest"])
    if (
        oracle.get("automatic_pass") is not True
        or oracle.get("stable_evidence_id")
        != config["parent_oracle"]["stable_evidence_id"]
        or oracle.get("parent_manifest_sha256")
        != config["parent_manifest"]["sha256"]
        or manifest.get("split_summary", {}).get(
            "selection_used_target_or_pixels"
        )
        is not False
    ):
        raise FiveKSourceHardRetrievalRunError("Oracle parent is not eligible")
    families = config.get("descriptor_families", [])
    if families != [
        "global_photometric",
        "spatial_photometric",
        "tone_layout",
    ]:
        raise FiveKSourceHardRetrievalRunError("descriptor inventory drift")
    return {"oracle": oracle, "manifest": manifest}


def run_retrieval(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    oracle = validated["oracle"]
    manifest = validated["manifest"]
    development_results: dict[str, dict[str, Any]] = {
        family: {} for family in config["descriptor_families"]
    }
    loaded_variants: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for variant_name in config["required_pass_variants"]:
        variant = config["target_variants"][variant_name]
        development = load_split_population(
            manifest,
            split="development",
            target_variant=variant["target_variant"],
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        confirmation = load_split_population(
            manifest,
            split="confirmation",
            target_variant=variant["target_variant"],
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        loaded_variants[variant_name] = (development, confirmation)
        prepared = prepare_development_evidence(
            development,
            oracle["variants"][variant_name],
            config["operator"],
            config["development_evaluation"],
        )
        for family in config["descriptor_families"]:
            development_results[family][variant_name] = (
                evaluate_development_family(
                    rows=development,
                    prepared=prepared,
                    family=family,
                    descriptor_spec=config["descriptor"],
                    selector_spec=config["selector"],
                    gates=config["development_gates"],
                )
            )
        del prepared
    eligible = [
        family
        for family in config["descriptor_families"]
        if all(
            development_results[family][variant]["automatic_pass"]
            for variant in config["required_pass_variants"]
        )
    ]
    selected_family = None
    confirmation_results: dict[str, Any] = {}
    if eligible:
        selected_family = max(
            eligible,
            key=lambda family: (
                sum(
                    development_results[family][variant]["metrics"][
                        "mean_improvement_over_global"
                    ]
                    for variant in config["required_pass_variants"]
                ),
                -config["descriptor_families"].index(family),
            ),
        )
        for variant_name in config["required_pass_variants"]:
            development, confirmation = loaded_variants[variant_name]
            confirmation_results[variant_name] = evaluate_confirmation_family(
                development_rows=development,
                confirmation_rows=confirmation,
                oracle_report=oracle["variants"][variant_name],
                family=selected_family,
                descriptor_spec=config["descriptor"],
                selector_spec={
                    **config["selector"],
                    "samples_per_confirmation_image": config[
                        "confirmation_evaluation"
                    ]["samples_per_image"],
                },
                gates=config["confirmation_gates"],
            )
    stable = {
        "parent_oracle_sha256": config["parent_oracle"]["sha256"],
        "development": development_results,
        "eligible_families": eligible,
        "selected_family": selected_family,
        "confirmation": confirmation_results,
        "confirmation_executed": selected_family is not None,
        "automatic_pass": selected_family is not None
        and all(
            confirmation_results[variant]["automatic_pass"]
            for variant in config["required_pass_variants"]
        ),
    }
    report = {
        "schema": "neuro_film.u5_r2bq2a_fivek_source_hard_retrieval.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest(),
        "confirmation_family_selection_used_targets": False,
        "dense_blending_used": False,
        "learned_final_rgb_used": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKSourceHardRetrievalRunError",
    "run_retrieval",
    "validate_contract",
]
