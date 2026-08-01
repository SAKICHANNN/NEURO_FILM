"""Hash-bound runner for the FiveK explicit-operator style ceiling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.fivek_casebank_oracle import load_split_population
from src.eval.fivek_operator_style_ceiling import diagnose_operator_style_ceiling


class FiveKOperatorStyleCeilingRunError(ValueError):
    """Raised when the style-ceiling diagnostic contract drifts."""


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
        raise FiveKOperatorStyleCeilingRunError("parent evidence drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FiveKOperatorStyleCeilingRunError("parent must be an object")
    return value


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema_version") != 1
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("training_or_fitting_allowed") is not False
        or config.get("confirmation_pixels_allowed") is not False
        or config.get("product_integration_allowed") is not False
        or config.get("safe_execution", {}).get("hard_clipping_allowed") is not False
    ):
        raise FiveKOperatorStyleCeilingRunError("diagnostic boundary drift")
    decision = _load_hashed(root, config["parent_bq4_decision"])
    bq4 = _load_hashed(root, config["parent_bq4_report"])
    oracle = _load_hashed(root, config["parent_oracle"])
    manifest = _load_hashed(root, config["parent_manifest"])
    if (
        decision.get("decision")
        != "close_factorized_safe_residual_before_confirmation_on_style_retention"
        or bq4.get("confirmation_executed") is not False
        or bq4.get("stable_evidence_id")
        != config["parent_bq4_report"]["stable_evidence_id"]
        or oracle.get("automatic_pass") is not True
        or oracle.get("stable_evidence_id")
        != config["parent_oracle"]["stable_evidence_id"]
        or manifest.get("split_summary", {}).get(
            "selection_used_target_or_pixels"
        )
        is not False
    ):
        raise FiveKOperatorStyleCeilingRunError("ineligible parent branch")
    return {"decision": decision, "bq4": bq4, "oracle": oracle, "manifest": manifest}


def run_operator_style_ceiling(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    variants = {}
    for name in config["variants"]:
        rows = load_split_population(
            validated["manifest"],
            split="development",
            target_variant=name,
            maximum_side=int(config["decode"]["maximum_side"]),
        )
        variants[name] = diagnose_operator_style_ceiling(
            rows=rows,
            oracle_report=validated["oracle"]["variants"][name],
            split_spec=config["split"],
            safe_spec=config["safe_execution"],
            samples_per_image=int(config["decode"]["samples_per_image"]),
            gates=config["diagnostic_gates"],
        )
    stable = {
        "parent_bq4_report_sha256": config["parent_bq4_report"]["sha256"],
        "parent_oracle_sha256": config["parent_oracle"]["sha256"],
        "confirmation_loaded": False,
        "variants": variants,
        "branch": (
            "operator_capacity_inadequate"
            if any(
                value["branch"] == "operator_capacity_inadequate"
                for value in variants.values()
            )
            else "operator_capacity_adequate_or_selection_limited"
        ),
    }
    report = {
        "schema": "neuro_film.u5_r2bq4d_fivek_operator_style_ceiling.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKOperatorStyleCeilingRunError",
    "run_operator_style_ceiling",
    "validate_contract",
]
