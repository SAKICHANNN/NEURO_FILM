#!/usr/bin/env python3
"""Evaluate the frozen non-moment CFSM quantile challenger."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cfsm_prior_selection import (  # noqa: E402
    evaluate_split,
    selected_manifest_rows,
)
from src.color_match import canonical_sha256  # noqa: E402
from src.color_match.research import CFSMProjectionPolicy  # noqa: E402
from src.inference import atomic_write_json  # noqa: E402
from src.roll2film.synthetic_recovery import (  # noqa: E402
    generate_operator_manifest,
    manifest_sha256,
)


REPORT_SCHEMA_ID = "neuro-film.cfsm-quantile-report.v1"
_CONFIG_KEYS = {
    "schema_id",
    "experiment_id",
    "parent_config",
    "parent_manifest_sha256",
    "prior_kinds",
    "splits",
    "confirmation_gate",
    "execution",
    "claim_ceiling",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "confirmation"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_contract(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != _CONFIG_KEYS:
        raise ValueError("CFSM quantile config keys mismatch")
    if config["schema_id"] != "neuro-film.cfsm-quantile-experiment.v1":
        raise ValueError("unsupported CFSM quantile schema")
    if config["prior_kinds"] != [
        "fixed-uniform-cube",
        "fixed-uniform-cube-monotone-quantile-v1",
    ]:
        raise ValueError("CFSM quantile controls are not frozen")
    if config["execution"] != {
        "cpu_only": True,
        "external_images_allowed": False,
        "training_allowed": False,
        "parameter_tuning_allowed": False,
        "development_result_cannot_change_confirmation_gate": True,
        "confirmation_operator_and_palette_split_unopened": True,
        "product_integration_allowed": False,
    }:
        raise ValueError("CFSM quantile execution boundary mismatch")
    if (
        config["splits"]["development"]["parent_split"] != "fit"
        or config["splits"]["confirmation"]["parent_split"] != "stress"
    ):
        raise ValueError("CFSM quantile split boundary mismatch")
    parent_path = ROOT / str(config["parent_config"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    manifest = generate_operator_manifest(parent)
    if manifest_sha256(manifest) != config["parent_manifest_sha256"]:
        raise ValueError("CFSM quantile parent manifest hash mismatch")
    return config, parent, manifest


def confirmation_decision(
    config: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    uniform = results["fixed-uniform-cube"]["summary"]
    quantile = results[
        "fixed-uniform-cube-monotone-quantile-v1"
    ]["summary"]
    gate = config["confirmation_gate"]
    median_gain = (
        quantile["median_captured_style_fraction"]
        - uniform["median_captured_style_fraction"]
    )
    worst_loss = (
        uniform["worst_captured_style_fraction"]
        - quantile["worst_captured_style_fraction"]
    )
    checks = {
        "median_gain": median_gain
        >= float(gate["minimum_median_gain_over_uniform"]),
        "improvement_rate": quantile["improvement_rate"]
        >= float(gate["minimum_improvement_rate"]),
        "worst_loss": worst_loss
        <= float(gate["maximum_worst_loss_vs_uniform"]),
        "new_boundary": quantile["maximum_new_boundary_fraction"]
        <= float(gate["maximum_new_boundary_fraction"]),
        "constraints": quantile["constraint_pass_fraction"] == 1.0,
        "identity_fallbacks": quantile["identity_fallback_count"]
        <= int(gate["identity_fallbacks_allowed"]),
    }
    passed = all(checks.values())
    return {
        "status": (
            "quantile-synthetic-confirmation-passed"
            if passed
            else "quantile-route-closed"
        ),
        "checks": checks,
        "median_gain_over_uniform": float(median_gain),
        "worst_loss_vs_uniform": float(worst_loss),
        "real_photo_review_open": passed,
        "product_integration_open": False,
    }


def main() -> int:
    args = _parser().parse_args()
    config, parent, manifest = load_contract(args.config)
    rows = selected_manifest_rows(config, manifest, args.split)
    kinds = tuple(config["prior_kinds"])
    results = evaluate_split(
        config,
        parent,
        rows,
        split=args.split,
        prior_kinds=kinds,
    )
    if args.split == "confirmation":
        decision = confirmation_decision(config, results)
    else:
        decision = {
            "status": "development-diagnostic-only",
            "confirmation_authorized_by_preregistration": True,
            "gate_mutation_allowed": False,
            "product_integration_open": False,
        }
    payload: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_id": "",
        "experiment_id": config["experiment_id"],
        "split": args.split,
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "selection_rows": len(rows),
        "projection_policy": asdict(CFSMProjectionPolicy()),
        "prior_kinds": list(kinds),
        "results": results,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    payload["report_id"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "report_id"}
    )
    report_sha256 = atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "split": args.split,
                "report_id": payload["report_id"],
                "report_sha256": report_sha256,
                "decision": decision,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
