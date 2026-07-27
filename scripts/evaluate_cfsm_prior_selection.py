#!/usr/bin/env python3
"""Select analytic CFSM priors without tuning on the real 30-pair matrix."""

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

from src.color_match import canonical_sha256  # noqa: E402
from src.color_match.research import (  # noqa: E402
    CFSMProjectionPolicy,
    EmpiricalNeutralPrior,
    fit_cfsm_analytic_candidate,
    fit_cfsm_candidate,
    fit_cfsm_empirical_candidate,
    render_cfsm_candidate,
)
from src.inference import atomic_write_json  # noqa: E402
from src.preprocess import SourceProfile, WorkingImage  # noqa: E402
from src.roll2film.synthetic_recovery import (  # noqa: E402
    generate_operator_manifest,
    manifest_sha256,
    operator_from_manifest,
    palette_cloud,
    palette_pairs,
)


REPORT_SCHEMA_ID = "neuro-film.cfsm-prior-selection-report.v1"
_CONFIG_KEYS = {
    "schema_id",
    "experiment_id",
    "parent_config",
    "parent_manifest_sha256",
    "prior_kinds",
    "splits",
    "selection",
    "execution",
    "claim_ceiling",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "confirmation"),
        required=True,
    )
    parser.add_argument(
        "--selected-prior",
        help="Required only for confirmation; validation winner to compare.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != _CONFIG_KEYS:
        raise ValueError("CFSM prior-selection config keys mismatch")
    if config["schema_id"] != "neuro-film.cfsm-prior-selection.v1":
        raise ValueError("unsupported CFSM prior-selection schema")
    parent_path = ROOT / str(config["parent_config"])
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    manifest = generate_operator_manifest(parent)
    if manifest_sha256(manifest) != config["parent_manifest_sha256"]:
        raise ValueError("CFSM prior-selection parent manifest hash mismatch")
    kinds = tuple(config["prior_kinds"])
    if (
        not kinds
        or kinds[0] != "fixed-uniform-cube"
        or len(set(kinds)) != len(kinds)
    ):
        raise ValueError("CFSM prior kinds must begin with a unique uniform control")
    return config, parent, manifest


def selected_manifest_rows(
    config: dict[str, Any],
    manifest: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    if split not in config["splits"]:
        raise ValueError(f"unsupported selection split: {split}")
    specification = config["splits"][split]
    parent_split = str(specification["parent_split"])
    limit = int(specification["operators_per_family"])
    if limit < 1:
        raise ValueError("operators_per_family must be positive")
    rows = [
        row
        for row in manifest
        if row["split"] == parent_split
        and int(row["within_split_index"]) < limit
    ]
    families = tuple(
        sorted({str(row["family"]) for row in manifest})
    )
    if len(rows) != limit * len(families):
        raise ValueError("selection rows do not cover every operator family")
    return rows


def _working(pixels: np.ndarray, name: str) -> WorkingImage:
    values = np.asarray(pixels, dtype=np.float32).reshape(32, 32, 3)
    return WorkingImage(
        pixels=values,
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile(
            "cfsm_prior_selection_v1",
            "generated numeric cube benchmark",
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path(f"synthetic/{name}.exr"),
    )


def _candidate(
    prior_kind: str,
    reference: WorkingImage,
    policy: CFSMProjectionPolicy,
    empirical_prior: EmpiricalNeutralPrior | None = None,
):
    if prior_kind == "fixed-uniform-cube":
        return fit_cfsm_candidate(reference, policy=policy)
    if prior_kind == "empirical-neutral-photo-v1":
        if empirical_prior is None:
            raise ValueError(
                "empirical prior kind requires a validated artifact"
            )
        return fit_cfsm_empirical_candidate(
            reference,
            prior_mean=empirical_prior.mean,
            prior_covariance=empirical_prior.covariance,
            prior_id=empirical_prior.prior_id,
            source_image_count=empirical_prior.source_image_count,
            source_pixel_count=empirical_prior.source_pixel_count,
            policy=policy,
        )
    return fit_cfsm_analytic_candidate(
        reference,
        prior_kind=prior_kind,
        policy=policy,
    )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty prior result")
    captured = np.asarray(
        [row["captured_style_fraction"] for row in records],
        dtype=np.float64,
    )
    return {
        "observations": len(records),
        "improved_count": int(np.count_nonzero(captured > 0.0)),
        "improvement_rate": float(np.mean(captured > 0.0)),
        "median_captured_style_fraction": float(np.median(captured)),
        "worst_captured_style_fraction": float(np.min(captured)),
        "median_candidate_rgb_rmse": float(
            np.median([row["candidate_rgb_rmse"] for row in records])
        ),
        "maximum_new_boundary_fraction": float(
            np.max([row["new_boundary_fraction"] for row in records])
        ),
        "constraint_pass_fraction": float(
            np.mean([row["constraint_passed"] for row in records])
        ),
        "identity_fallback_count": int(
            np.count_nonzero([row["identity_fallback"] for row in records])
        ),
    }


def evaluate_split(
    config: dict[str, Any],
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    split: str,
    prior_kinds: tuple[str, ...],
    empirical_prior: EmpiricalNeutralPrior | None = None,
) -> dict[str, Any]:
    palette_spec = parent["palettes"]
    pixel_count = int(palette_spec["pixels_per_cloud"])
    if pixel_count != 1024:
        raise ValueError("CFSM selection requires 1024-pixel palette clouds")
    parent_split = str(config["splits"][split]["parent_split"])
    palette_names = (
        list(palette_spec["development"])
        if parent_split in {"fit", "validation"}
        else list(palette_spec[parent_split])
    )
    seed = int(parent["seed"])
    clouds = {
        name: palette_cloud(name, pixel_count, seed)
        for name in palette_names
    }
    policy = CFSMProjectionPolicy()
    records_by_prior: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in prior_kinds
    }
    for row in rows:
        operator = operator_from_manifest(row)
        pairs = palette_pairs(
            palette_names,
            operator_index=int(row["family_index"]),
            observations=2,
        )
        for observation_index, (query_name, reference_name) in enumerate(pairs):
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
            identity_rmse = float(
                np.sqrt(np.mean((query - target) ** 2))
            )
            for prior_kind in prior_kinds:
                candidate = _candidate(
                    prior_kind,
                    reference_working,
                    policy,
                    empirical_prior,
                )
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
                records_by_prior[prior_kind].append(
                    {
                        "operator_id": operator.operator_id,
                        "family": operator.family,
                        "query_palette": query_name,
                        "reference_palette": reference_name,
                        "candidate_id": candidate.candidate_id,
                        "projected_strength": (
                            candidate.diagnostics.projected_strength
                        ),
                        "identity_rgb_rmse": identity_rmse,
                        "candidate_rgb_rmse": candidate_rmse,
                        "captured_style_fraction": (
                            0.0
                            if identity_rmse <= 1e-15
                            else 1.0 - candidate_rmse / identity_rmse
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
        kind: {
            "summary": _summarize(records),
            "records": records,
        }
        for kind, records in records_by_prior.items()
    }


def validation_decision(
    config: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    uniform = results["fixed-uniform-cube"]["summary"]
    ranking = sorted(
        results,
        key=lambda kind: (
            results[kind]["summary"]["median_captured_style_fraction"],
            results[kind]["summary"]["improvement_rate"],
            results[kind]["summary"]["worst_captured_style_fraction"],
        ),
        reverse=True,
    )
    winner = ranking[0]
    winner_summary = results[winner]["summary"]
    gain = (
        winner_summary["median_captured_style_fraction"]
        - uniform["median_captured_style_fraction"]
    )
    selection = config["selection"]
    structural_pass = (
        winner_summary["constraint_pass_fraction"] == 1.0
        and winner_summary["identity_fallback_count"]
        <= int(selection["identity_fallbacks_allowed"])
    )
    passed = (
        winner != "fixed-uniform-cube"
        and gain
        >= float(
            selection["minimum_validation_median_gain_over_uniform"]
        )
        and structural_pass
    )
    return {
        "status": (
            "analytic-prior-selected"
            if passed
            else "no-analytic-prior-selected"
        ),
        "winner": winner,
        "ranking": ranking,
        "median_gain_over_uniform": float(gain),
        "structural_pass": structural_pass,
        "confirmation_open": passed,
    }


def confirmation_decision(
    config: dict[str, Any],
    results: dict[str, Any],
    selected_prior: str,
) -> dict[str, Any]:
    uniform = results["fixed-uniform-cube"]["summary"]
    selected = results[selected_prior]["summary"]
    median_gain = (
        selected["median_captured_style_fraction"]
        - uniform["median_captured_style_fraction"]
    )
    worst_loss = (
        uniform["worst_captured_style_fraction"]
        - selected["worst_captured_style_fraction"]
    )
    selection = config["selection"]
    passed = (
        median_gain
        >= float(
            selection["minimum_confirmation_median_gain_over_uniform"]
        )
        and worst_loss
        <= float(selection["maximum_confirmation_worst_loss_vs_uniform"])
        and selected["constraint_pass_fraction"] == 1.0
        and selected["identity_fallback_count"]
        <= int(selection["identity_fallbacks_allowed"])
    )
    return {
        "status": (
            "synthetic-confirmation-passed"
            if passed
            else "synthetic-confirmation-failed"
        ),
        "selected_prior": selected_prior,
        "median_gain_over_uniform": float(median_gain),
        "worst_loss_vs_uniform": float(worst_loss),
        "real_matrix_confirmation_open": passed,
    }


def main() -> int:
    args = _parser().parse_args()
    config, parent, manifest = load_contract(args.config)
    rows = selected_manifest_rows(config, manifest, args.split)
    if args.split == "confirmation":
        if (
            not args.selected_prior
            or args.selected_prior == "fixed-uniform-cube"
            or args.selected_prior not in config["prior_kinds"]
        ):
            raise ValueError(
                "confirmation requires one frozen analytic selected prior"
            )
        prior_kinds = ("fixed-uniform-cube", args.selected_prior)
    else:
        if args.selected_prior is not None:
            raise ValueError(
                "selected-prior is valid only for confirmation"
            )
        prior_kinds = tuple(config["prior_kinds"])
    results = evaluate_split(
        config,
        parent,
        rows,
        split=args.split,
        prior_kinds=prior_kinds,
    )
    if args.split == "validation":
        decision = validation_decision(config, results)
    elif args.split == "confirmation":
        decision = confirmation_decision(
            config,
            results,
            str(args.selected_prior),
        )
    else:
        decision = {
            "status": "development-only",
            "confirmation_open": False,
        }
    payload: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_id": "",
        "experiment_id": config["experiment_id"],
        "split": args.split,
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "selection_rows": len(rows),
        "projection_policy": asdict(CFSMProjectionPolicy()),
        "prior_kinds": list(prior_kinds),
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
