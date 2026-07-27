#!/usr/bin/env python3
"""Evaluate the isolated CFSM-v0 challenger on the frozen promotion matrix."""

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

from src.color_match import (  # noqa: E402
    ContextInvarianceBatchMetrics,
    PhotographicSafetyBatchMetrics,
    adjudicate_promotion,
    aggregate_known_operator_samples,
    canonical_sha256,
    evaluate_context_invariance_outputs,
    evaluate_known_operator_batch,
    evaluate_photographic_probe,
    make_context_invariance_probes,
    make_photographic_probe,
)
from src.color_match.research import (  # noqa: E402
    CFSM_ALGORITHM_ID,
    CFSM_BATCH_ALGORITHM_ID,
    CFSMProjectionPolicy,
    fit_cfsm_batch_candidate,
    fit_cfsm_candidate,
    render_cfsm_batch,
    render_cfsm_candidate,
)
from src.inference import atomic_write_json, sha256_file  # noqa: E402
from src.preprocess import load_working_image  # noqa: E402


REPORT_SCHEMA_ID = "neuro-film.cfsm-promotion-report.v0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen reference/source/target matrix through the "
            "isolated fixed-LUT CFSM-v0 challenger."
        )
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        required=True,
        help="Existing promotion report that freezes paths and hashes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-neutral-axis-error",
        type=float,
        default=0.04,
        help=(
            "Research ablation only: maximum linear-RGB neutral-axis error "
            "allowed by the safe projection."
        ),
    )
    parser.add_argument(
        "--prior-mode",
        choices=("fixed-cube", "source-batch"),
        default="fixed-cube",
    )
    return parser


def _batch_photographic(samples: tuple[Any, ...]) -> PhotographicSafetyBatchMetrics:
    failed = sum(not sample.passed for sample in samples)
    return PhotographicSafetyBatchMetrics(
        samples=samples,
        passed_recipe_count=len(samples) - failed,
        failed_recipe_count=failed,
        maximum_neutral_chroma_p95=max(
            sample.neutral_chroma_p95 for sample in samples
        ),
        maximum_tone_reversal_fraction=max(
            sample.tone_reversal_fraction for sample in samples
        ),
        largest_tone_reversal_delta_l=min(
            sample.tone_largest_reversal_delta_l for sample in samples
        ),
        maximum_tone_plateau_fraction=max(
            sample.tone_plateau_fraction for sample in samples
        ),
        maximum_new_boundary_fraction=max(
            sample.new_boundary_fraction for sample in samples
        ),
        maximum_semantic_hue_rotation_p95_degrees=max(
            sample.semantic_hue_rotation_p95_degrees for sample in samples
        ),
    )


def _batch_context(samples: tuple[Any, ...]) -> ContextInvarianceBatchMetrics:
    failed = sum(not sample.passed for sample in samples)
    return ContextInvarianceBatchMetrics(
        samples=samples,
        passed_recipe_count=len(samples) - failed,
        failed_recipe_count=failed,
        maximum_delta_e76_median=max(
            sample.delta_e76_median for sample in samples
        ),
        maximum_delta_e76_p95=max(
            sample.delta_e76_p95 for sample in samples
        ),
        maximum_delta_e76=max(
            sample.delta_e76_maximum for sample in samples
        ),
    )


def _paths(payload: dict[str, Any], key: str) -> dict[str, Path]:
    values = payload.get(key)
    if not isinstance(values, dict):
        raise ValueError(f"baseline report {key} is invalid")
    return {str(sample_id): Path(str(path)) for sample_id, path in values.items()}


def _verify_frozen_files(
    paths: dict[str, Path],
    hashes: dict[str, str],
    label: str,
) -> None:
    if set(paths) != set(hashes):
        raise ValueError(f"{label} path/hash IDs differ")
    for sample_id, path in paths.items():
        if not path.is_file() or sha256_file(path) != hashes[sample_id]:
            raise ValueError(
                f"{label} file does not match frozen hash: {sample_id}"
            )


def main() -> int:
    args = _parser().parse_args()
    baseline_bytes = args.baseline_report.read_bytes()
    baseline = json.loads(baseline_bytes)
    sample_ids = tuple(str(value) for value in baseline["sample_ids"])
    if len(sample_ids) < 2 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("baseline sample IDs are invalid")
    reference_paths = _paths(baseline, "reference_paths")
    source_paths = _paths(baseline, "source_paths")
    target_paths = _paths(baseline, "target_paths")
    reference_hashes = dict(baseline["reference_file_sha256"])
    source_hashes = dict(baseline["source_file_sha256"])
    target_hashes = dict(baseline["target_file_sha256"])
    _verify_frozen_files(reference_paths, reference_hashes, "reference")
    _verify_frozen_files(source_paths, source_hashes, "source")
    _verify_frozen_files(target_paths, target_hashes, "target")

    policy = CFSMProjectionPolicy(
        maximum_neutral_axis_error=args.maximum_neutral_axis_error,
    )
    candidates = {}
    for sample_id in sample_ids:
        reference = load_working_image(reference_paths[sample_id])
        if args.prior_mode == "fixed-cube":
            candidate = fit_cfsm_candidate(reference, policy=policy)
        else:
            candidate = fit_cfsm_batch_candidate(
                reference,
                (
                    load_working_image(source_paths[source_id])
                    for source_id in sample_ids
                    if source_id != sample_id
                ),
                policy=policy,
            )
        candidates[sample_id] = candidate
    rows = []
    for reference_id in sample_ids:
        candidate = candidates[reference_id]
        for source_id in sample_ids:
            if source_id == reference_id:
                continue
            source = load_working_image(source_paths[source_id])
            target = load_working_image(target_paths[source_id])
            rendered = render_cfsm_candidate(candidate, source)
            row = evaluate_known_operator_batch(
                [source],
                [target],
                [rendered],
                sample_ids=[f"{reference_id}->{source_id}"],
            ).samples[0]
            rows.append(row)
    known_operator = aggregate_known_operator_samples(rows)

    photographic_probe = make_photographic_probe()
    photographic_samples = tuple(
        evaluate_photographic_probe(
            photographic_probe,
            render_cfsm_candidate(candidate, photographic_probe),
        )
        for candidate in candidates.values()
    )
    photographic = _batch_photographic(photographic_samples)

    first_probe, second_probe, shared_region = make_context_invariance_probes()
    context_samples = []
    for candidate in candidates.values():
        first_output, second_output = render_cfsm_batch(
            candidate,
            (first_probe, second_probe),
        )
        context_samples.append(
            evaluate_context_invariance_outputs(
                first_output,
                second_output,
                shared_region=shared_region,
            )
        )
    context = _batch_context(tuple(context_samples))
    decision = adjudicate_promotion(
        known_operator,
        photographic,
        context,
    )
    payload: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_id": "",
        "algorithm_id": (
            CFSM_ALGORITHM_ID
            if args.prior_mode == "fixed-cube"
            else CFSM_BATCH_ALGORITHM_ID
        ),
        "projection_policy": asdict(policy),
        "baseline_report_sha256": sha256_file(args.baseline_report),
        "sample_ids": list(sample_ids),
        "reference_paths": {
            key: str(value.resolve()) for key, value in reference_paths.items()
        },
        "source_paths": {
            key: str(value.resolve()) for key, value in source_paths.items()
        },
        "target_paths": {
            key: str(value.resolve()) for key, value in target_paths.items()
        },
        "reference_file_sha256": reference_hashes,
        "source_file_sha256": source_hashes,
        "target_file_sha256": target_hashes,
        "candidate_ids": {
            key: candidate.candidate_id
            for key, candidate in candidates.items()
        },
        "candidate_diagnostics": {
            key: {
                "projected_strength": (
                    candidate.diagnostics.projected_strength
                ),
                "used_identity_fallback": (
                    candidate.diagnostics.used_identity_fallback
                ),
                "constraint_report": (
                    candidate.diagnostics.constraint_report.to_dict()
                ),
            }
            for key, candidate in candidates.items()
        },
        "known_operator": asdict(known_operator),
        "photographic_safety": asdict(photographic),
        "context_invariance": asdict(context),
        "visual_review": None,
        "promotion_decision": asdict(decision),
    }
    payload["report_id"] = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "report_id",
                "reference_paths",
                "source_paths",
                "target_paths",
            }
        }
    )
    report_sha256 = atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "report_id": payload["report_id"],
                "report_sha256": report_sha256,
                "sample_count": len(known_operator.samples),
                "promotion_status": decision.status,
                "reasons": list(decision.reasons),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
