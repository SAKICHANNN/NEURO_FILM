#!/usr/bin/env python3
"""Evaluate source-batch-conditioned Gaussian and quantile CFSM candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cfsm_prior_selection import (  # noqa: E402
    _summarize,
    _working,
)
from src.color_match import canonical_sha256  # noqa: E402
from src.color_match.research import (  # noqa: E402
    CFSMProjectionPolicy,
    fit_cfsm_batch_candidate,
    fit_cfsm_batch_quantile_candidate,
    render_cfsm_candidate,
)
from src.inference import atomic_write_json  # noqa: E402
from src.roll2film.synthetic_recovery import (  # noqa: E402
    generate_operator_manifest,
    manifest_sha256,
    operator_from_manifest,
    palette_cloud,
    palette_pairs,
)


REPORT_SCHEMA_ID = "neuro-film.cfsm-batch-quantile-report.v1"
_CONFIG_KEYS = {
    "schema_id",
    "experiment_id",
    "parent_config",
    "parent_manifest_sha256",
    "confirmation",
    "candidates",
    "confirmation_gate",
    "execution",
    "claim_ceiling",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_contract(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != _CONFIG_KEYS:
        raise ValueError("CFSM batch-quantile config keys mismatch")
    if (
        config["schema_id"]
        != "neuro-film.cfsm-batch-quantile-experiment.v1"
    ):
        raise ValueError("unsupported CFSM batch-quantile schema")
    if config["candidates"] != [
        "uploaded-source-batch-gaussian-v1",
        "uploaded-source-batch-monotone-quantile-v1",
    ]:
        raise ValueError("CFSM batch-quantile candidates are not frozen")
    if config["execution"] != {
        "cpu_only": True,
        "external_images_allowed": False,
        "training_allowed": False,
        "parameter_tuning_allowed": False,
        "confirmation_rows_unopened_before_freeze": True,
        "candidate_may_depend_on_complete_source_batch": True,
        "candidate_must_be_fixed_within_batch": True,
        "product_integration_allowed": False,
    }:
        raise ValueError("CFSM batch-quantile execution boundary mismatch")
    parent = json.loads(
        (ROOT / str(config["parent_config"])).read_text(encoding="utf-8")
    )
    manifest = generate_operator_manifest(parent)
    if manifest_sha256(manifest) != config["parent_manifest_sha256"]:
        raise ValueError("CFSM batch-quantile parent manifest hash mismatch")
    return config, parent, manifest


def selected_rows(
    config: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    specification = config["confirmation"]
    start = int(specification["within_split_start"])
    count = int(specification["operators_per_family"])
    if start < 0 or count < 1:
        raise ValueError("CFSM batch-quantile row window is invalid")
    rows = [
        row
        for row in manifest
        if row["split"] == specification["parent_split"]
        and start
        <= int(row["within_split_index"])
        < start + count
    ]
    families = {str(row["family"]) for row in manifest}
    if len(rows) != count * len(families):
        raise ValueError("CFSM batch-quantile rows are incomplete")
    return rows


def evaluate_confirmation(
    config: dict[str, Any],
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    palette_spec = parent["palettes"]
    pixel_count = int(palette_spec["pixels_per_cloud"])
    if pixel_count != 1024:
        raise ValueError("batch-quantile requires 1024-pixel clouds")
    palette_names = list(palette_spec["stress"])
    if (
        not config["confirmation"]["source_batch_uses_all_stress_palettes"]
        or len(palette_names) < 3
    ):
        raise ValueError("batch-quantile source palette contract mismatch")
    seed = int(parent["seed"])
    clouds = {
        name: palette_cloud(name, pixel_count, seed)
        for name in palette_names
    }
    source_batch = tuple(
        _working(clouds[name], f"batch-{name}") for name in palette_names
    )
    records: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in config["candidates"]
    }
    policy = CFSMProjectionPolicy()
    for row in rows:
        operator = operator_from_manifest(row)
        pairs = palette_pairs(
            palette_names,
            operator_index=int(row["family_index"]),
            observations=int(
                config["confirmation"]["observations_per_operator"]
            ),
        )
        for observation_index, (query_name, reference_name) in enumerate(
            pairs
        ):
            query = clouds[query_name]
            reference = operator.apply(clouds[reference_name])
            target = operator.apply(query)
            source_working = _working(
                query,
                f"{operator.operator_id}-{observation_index}-source",
            )
            reference_working = _working(
                reference,
                f"{operator.operator_id}-{observation_index}-reference",
            )
            candidates = {
                "uploaded-source-batch-gaussian-v1": (
                    fit_cfsm_batch_candidate(
                        reference_working,
                        source_batch,
                        policy=policy,
                    )
                ),
                "uploaded-source-batch-monotone-quantile-v1": (
                    fit_cfsm_batch_quantile_candidate(
                        reference_working,
                        source_batch,
                        policy=policy,
                    )
                ),
            }
            identity_rmse = float(
                np.sqrt(np.mean((query - target) ** 2))
            )
            for candidate_name, candidate in candidates.items():
                rendered = render_cfsm_candidate(
                    candidate,
                    source_working,
                ).pixels.astype(np.float64).reshape(-1, 3)
                candidate_rmse = float(
                    np.sqrt(np.mean((rendered - target) ** 2))
                )
                source_boundary = np.any(
                    (query <= 1.0 / 65535.0)
                    | (query >= 1.0 - 1.0 / 65535.0),
                    axis=-1,
                )
                candidate_boundary = np.any(
                    (rendered <= 1.0 / 65535.0)
                    | (rendered >= 1.0 - 1.0 / 65535.0),
                    axis=-1,
                )
                records[candidate_name].append(
                    {
                        "operator_id": operator.operator_id,
                        "family": operator.family,
                        "query_palette": query_name,
                        "reference_palette": reference_name,
                        "candidate_id": candidate.candidate_id,
                        "projected_strength": (
                            candidate.diagnostics.projected_strength
                        ),
                        "source_batch_count": len(source_batch),
                        "identity_rgb_rmse": identity_rmse,
                        "candidate_rgb_rmse": candidate_rmse,
                        "captured_style_fraction": (
                            (identity_rmse - candidate_rmse)
                            / max(identity_rmse, 1e-12)
                        ),
                        "new_boundary_fraction": float(
                            np.mean(
                                candidate_boundary & ~source_boundary,
                                dtype=np.float64,
                            )
                        ),
                        "constraint_passed": (
                            candidate.diagnostics.constraint_report.passes
                        ),
                        "identity_fallback": (
                            candidate.diagnostics.used_identity_fallback
                        ),
                    }
                )
    return {
        name: {"summary": _summarize(rows_), "records": rows_}
        for name, rows_ in records.items()
    }


def confirmation_decision(
    config: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    gaussian = results[
        "uploaded-source-batch-gaussian-v1"
    ]["summary"]
    quantile = results[
        "uploaded-source-batch-monotone-quantile-v1"
    ]["summary"]
    gate = config["confirmation_gate"]
    median_gain = (
        quantile["median_captured_style_fraction"]
        - gaussian["median_captured_style_fraction"]
    )
    worst_loss = (
        gaussian["worst_captured_style_fraction"]
        - quantile["worst_captured_style_fraction"]
    )
    checks = {
        "median_gain": median_gain
        >= float(gate["minimum_median_gain_over_gaussian_batch"]),
        "improvement_rate": quantile["improvement_rate"]
        >= float(gate["minimum_improvement_rate"]),
        "worst_loss": worst_loss
        <= float(gate["maximum_worst_loss_vs_gaussian_batch"]),
        "new_boundary": quantile["maximum_new_boundary_fraction"]
        <= float(gate["maximum_new_boundary_fraction"]),
        "constraints": quantile["constraint_pass_fraction"] == 1.0,
        "identity_fallbacks": quantile["identity_fallback_count"]
        <= int(gate["identity_fallbacks_allowed"]),
    }
    passed = all(checks.values())
    return {
        "status": (
            "batch-quantile-synthetic-confirmation-passed"
            if passed
            else "batch-quantile-route-closed"
        ),
        "checks": checks,
        "median_gain_over_gaussian_batch": float(median_gain),
        "worst_loss_vs_gaussian_batch": float(worst_loss),
        "real_photo_review_open": passed,
        "reference_only_recipe_claimed": False,
        "product_integration_open": False,
    }


def main() -> int:
    args = _parser().parse_args()
    config, parent, manifest = load_contract(args.config)
    rows = selected_rows(config, manifest)
    results = evaluate_confirmation(config, parent, rows)
    decision = confirmation_decision(config, results)
    payload: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_id": "",
        "experiment_id": config["experiment_id"],
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "selection_rows": len(rows),
        "projection_policy": asdict(CFSMProjectionPolicy()),
        "candidates": list(config["candidates"]),
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
