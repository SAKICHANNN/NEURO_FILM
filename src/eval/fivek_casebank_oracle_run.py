"""Hash-bound runner for the frozen FiveK off-diagonal case-bank Oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.eval.fivek_casebank_oracle import (
    evaluate_offdiagonal_oracle,
    load_split_population,
)


class FiveKCasebankOracleRunError(ValueError):
    """Raised when parent evidence or the Oracle protocol drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _load_hashed_json(root: Path, path: str, expected: str) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected:
        raise FiveKCasebankOracleRunError(f"parent evidence drift: {path}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FiveKCasebankOracleRunError("parent evidence must be an object")
    return payload


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKCasebankOracleRunError("Oracle contract is not frozen")
    if (
        config.get("router_training_allowed")
        or config.get("confirmation_target_use_for_selection_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise FiveKCasebankOracleRunError("Oracle boundary drift")
    parent = config["parent_population"]
    manifest = _load_hashed_json(
        root, parent["manifest"], parent["manifest_sha256"]
    )
    report = _load_hashed_json(
        root, parent["report"], parent["report_sha256"]
    )
    summary = manifest.get("split_summary", {})
    if (
        report.get("automatic_pass") is not True
        or summary.get("development_rows") != parent["development_rows"]
        or summary.get("confirmation_rows") != parent["confirmation_rows"]
        or summary.get("selection_used_target_or_pixels") is not False
    ):
        raise FiveKCasebankOracleRunError("parent population is not eligible")
    variants = config["target_variants"]
    if set(variants) != {"aligned_expert", "filtered"}:
        raise FiveKCasebankOracleRunError("target-control inventory drift")
    if config.get("required_pass_variants") != [
        "aligned_expert",
        "filtered",
    ]:
        raise FiveKCasebankOracleRunError("pass-variant policy drift")
    return {"manifest": manifest}


def run_oracle(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    variants = {}
    for name in config["required_pass_variants"]:
        variant = config["target_variants"][name]
        maximum_side = int(config["decode"]["maximum_side"])
        development = load_split_population(
            validated["manifest"],
            split="development",
            target_variant=variant["target_variant"],
            maximum_side=maximum_side,
        )
        confirmation = load_split_population(
            validated["manifest"],
            split="confirmation",
            target_variant=variant["target_variant"],
            maximum_side=maximum_side,
        )
        variants[name] = evaluate_offdiagonal_oracle(
            development,
            confirmation,
            {
                "operator": config["operator"],
                "evaluation": variant["evaluation"],
                "router_training_allowed": False,
                "final_rgb_learning_allowed": False,
            },
        )
        del development, confirmation
    stable = {
        "parent_manifest_sha256": config["parent_population"][
            "manifest_sha256"
        ],
        "variants": variants,
        "automatic_pass": all(
            variants[name]["automatic_pass"]
            for name in config["required_pass_variants"]
        ),
    }
    report = {
        "schema": "neuro_film.u5_r2bq1_casebank_oracle_report.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "router_training_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(report))
    return report


__all__ = [
    "FiveKCasebankOracleRunError",
    "run_oracle",
    "validate_contract",
]
