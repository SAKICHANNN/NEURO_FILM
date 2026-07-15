"""Run the one-shot frozen BlueNeg two-roll matched-control evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.baselines import fit_lab_mean_std_operator  # noqa: E402
from src.roll2film.blueneg import load_blueneg_transformations  # noqa: E402
from src.roll2film.blueneg_eval import (  # noqa: E402
    BlueNegRollSamples,
    load_confirmatory_roll_samples,
    load_development_roll_samples,
    shuffled_target_support_groups,
)
from src.roll2film.ct5_evaluation import (  # noqa: E402
    cluster_bootstrap_improvement,
    evaluate_sampled_candidate,
    summarize_operator_output,
)
from src.roll2film.operators import AffineColorOperator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_evaluator.json",
    )
    parser.add_argument(
        "--development-decision",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_development_decision.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "confirmatory_report.json",
    )
    return parser.parse_args()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fit(samples: BlueNegRollSamples) -> Any:
    return fit_lab_mean_std_operator(samples.support_source, samples.support_target)


def _metadata_by_filename(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["filename"]): row for row in rows}


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_config.read_text(encoding="utf-8"))
    if decision["evaluator_policy_sha256"] != _sha(args.policy):
        raise ValueError("BlueNeg evaluator policy hash mismatch")
    development_report = (
        ROOT / "outputs" / "roll2film" / "blueneg_v1" / "development_report.json"
    )
    if _sha(development_report) != decision["development_report_sha256"]:
        raise ValueError("BlueNeg development report hash mismatch")
    if decision["selected_family"] != "lab_mean_std":
        raise ValueError("confirmatory runner implements only the frozen Lab family")
    root = (ROOT / pilot["root"]).resolve()
    evidence = (ROOT / pilot["evidence_dir"]).resolve()
    source = policy["source_contract"]
    checks = {
        evidence / "report.json": source["metadata_report_sha256"],
        evidence / "frames.jsonl": source["frames_manifest_sha256"],
        evidence / "acquisition.json": source["acquisition_manifest_sha256"],
        ROOT / "outputs" / "roll2film" / "blueneg_v1" / "download_report.json": source[
            "download_report_sha256"
        ],
        root / "transformations.pkl": source["transformations_sha256"],
    }
    for path, expected in checks.items():
        if _sha(path) != expected:
            raise ValueError(f"BlueNeg frozen hash mismatch: {path.name}")
    rows = _jsonl(evidence / "frames.jsonl")
    metadata = _metadata_by_filename(rows)
    transformations = load_blueneg_transformations(root / "transformations.pkl")
    budget = policy["fixed_budget"]
    alignment = policy["decode_and_alignment"]
    common = dict(
        root=root,
        rows=rows,
        transformations=transformations,
        support_frames=budget["support_frames_per_roll"],
        support_pixels=budget["support_pixels_per_frame_per_domain"],
        query_pixels=budget["query_pixels_per_frame"],
        support_seed=budget["support_target_permutation_seed"],
        query_seed=budget["query_sampling_seed"],
        minimum_side=alignment["minimum_aligned_side"],
    )
    development = {
        roll_id: load_development_roll_samples(
            **common,
            roll_id=roll_id,
            query_frames=budget["development_query_frames_per_roll"],
        )
        for roll_id in policy["matched_core"]["development_rolls"]
    }
    confirmatory = {
        roll_id: load_confirmatory_roll_samples(
            **common,
            roll_id=roll_id,
            query_frames=budget["confirmatory_query_frames_per_roll"],
        )
        for roll_id in policy["matched_core"]["confirmatory_rolls"]
    }
    dev_operators = {roll_id: _fit(samples) for roll_id, samples in development.items()}
    pooled_development = fit_lab_mean_std_operator(
        np.concatenate([samples.support_source for samples in development.values()]),
        np.concatenate([samples.support_target for samples in development.values()]),
    )
    confirm_operators = {roll_id: _fit(samples) for roll_id, samples in confirmatory.items()}
    confirm_rolls = tuple(confirmatory)
    shuffled = shuffled_target_support_groups(
        tuple(confirmatory[roll_id] for roll_id in confirm_rolls),  # type: ignore[arg-type]
        frames_per_roll=budget["support_frames_per_roll"],
        pixels_per_frame=budget["support_pixels_per_frame_per_domain"],
        seed=policy["statistics"]["bootstrap_seed"],
    )
    shuffled_operators = {
        roll_id: fit_lab_mean_std_operator(
            confirmatory[roll_id].support_source,
            shuffled[roll_id],
        )
        for roll_id in confirm_rolls
    }
    results: dict[str, Any] = {}
    operator_bundles: dict[str, Any] = {}
    for roll_id, samples in confirmatory.items():
        other_roll = next(value for value in confirm_rolls if value != roll_id)
        controls = {
            "correct_roll": confirm_operators[roll_id],
            "identity": AffineColorOperator.identity(),
            "pooled_development_rolls": pooled_development,
            **{
                f"development_wrong_roll_{dev_roll}": operator
                for dev_roll, operator in dev_operators.items()
            },
            "other_confirmatory_roll": confirm_operators[other_roll],
            "deterministic_shuffled_support_groups": shuffled_operators[roll_id],
        }
        operator_bundles[roll_id] = {
            name: operator.to_dict() for name, operator in controls.items()
        }
        candidate_results = {}
        for name, operator in controls.items():
            rendered = summarize_operator_output(operator, samples.query_source)
            summary, per_query = evaluate_sampled_candidate(
                samples.query_source,
                samples.query_target,
                rendered,
            )
            candidate_results[name] = {"summary": summary, "per_query": per_query}
        control_names = [name for name in controls if name != "correct_roll"]
        best_control = min(
            control_names,
            key=lambda name: candidate_results[name]["summary"][
                "mean_delta_e00_to_target"
            ],
        )
        support_locations = sorted(
            {str(metadata[name]["location"]) for name in samples.support_content_ids}
        )
        query_metadata = []
        for index, filename in enumerate(samples.query_content_ids):
            row = metadata[filename]
            query_metadata.append(
                {
                    "filename": filename,
                    "location": row["location"],
                    "scene_property": row["scene_property"],
                    "location_seen_in_support": str(row["location"]) in support_locations,
                    "correct_delta_e00": candidate_results["correct_roll"]["per_query"][index][
                        "mean_delta_e00_to_target"
                    ],
                    "best_control_delta_e00": candidate_results[best_control]["per_query"][index][
                        "mean_delta_e00_to_target"
                    ],
                }
            )
        results[roll_id] = {
            "support_content_ids": samples.support_content_ids,
            "support_locations": support_locations,
            "query_content_ids": samples.query_content_ids,
            "best_control": best_control,
            "candidates": candidate_results,
            "query_metadata": query_metadata,
        }
    correct_values = []
    baseline_values = []
    cluster_ids = []
    per_roll_gain = {}
    location_gains: dict[str, list[float]] = {"seen": [], "unseen": []}
    for roll_id, result in results.items():
        best = result["best_control"]
        correct = result["candidates"]["correct_roll"]["per_query"]
        baseline = result["candidates"][best]["per_query"]
        deltas = []
        for query, correct_row, baseline_row in zip(
            result["query_metadata"], correct, baseline
        ):
            correct_value = correct_row["mean_delta_e00_to_target"]
            baseline_value = baseline_row["mean_delta_e00_to_target"]
            correct_values.append(correct_value)
            baseline_values.append(baseline_value)
            cluster_ids.append(roll_id)
            gain = baseline_value - correct_value
            deltas.append(gain)
            location_gains["seen" if query["location_seen_in_support"] else "unseen"].append(
                gain
            )
        per_roll_gain[roll_id] = float(np.mean(deltas))
    bootstrap = cluster_bootstrap_improvement(
        baseline_values,
        correct_values,
        cluster_ids,
        seed=policy["statistics"]["bootstrap_seed"],
        resamples=policy["statistics"]["bootstrap_resamples"],
    )
    aggregate_control_means = {}
    control_names = [
        name for name in next(iter(results.values()))["candidates"] if name != "correct_roll"
    ]
    for name in control_names:
        aggregate_control_means[name] = float(
            np.mean(
                [
                    query["mean_delta_e00_to_target"]
                    for result in results.values()
                    for query in result["candidates"][name]["per_query"]
                ]
            )
        )
    correct_mean = float(np.mean(correct_values))
    style_mean = float(
        np.mean(
            [
                result["candidates"]["correct_roll"]["summary"][
                    "median_delta_e00_from_input"
                ]
                for result in results.values()
            ]
        )
    )
    both_positive = all(value > 0.0 for value in per_roll_gain.values())
    beats_every_control = all(correct_mean < value for value in aggregate_control_means.values())
    location_summary = {
        key: {
            "queries": len(values),
            "mean_gain": float(np.mean(values)) if values else None,
        }
        for key, values in location_gains.items()
    }
    nuisance_dependent = any(
        value["queries"] and value["mean_gain"] <= 0.0
        for value in location_summary.values()
    )
    statistical_pass = (
        both_positive
        and bootstrap["ci95_low"] > 0.0
        and beats_every_control
        and style_mean
        >= policy["development_selection"][
            "style_floor_median_delta_e00_from_query_input"
        ]
        and not nuisance_dependent
    )
    if statistical_pass:
        state = "awaiting_original_resolution_severe_adjudication"
    elif all(value <= 0.0 for value in per_roll_gain.values()):
        state = "fail_roll_information"
    else:
        state = "ambiguous_roll_information"
    report = {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "policy_sha256": _sha(args.policy),
        "development_decision_sha256": _sha(args.development_decision),
        "software_commit": _commit(),
        "selected_family": decision["selected_family"],
        "results": results,
        "operator_bundles": operator_bundles,
        "aggregate": {
            "correct_roll_mean_delta_e00": correct_mean,
            "correct_roll_mean_style_delta_e00": style_mean,
            "control_mean_delta_e00": aggregate_control_means,
            "per_roll_gain_vs_own_best_control": per_roll_gain,
            "bootstrap_vs_own_best_control": bootstrap,
            "location_strata_gain": location_summary,
            "both_rolls_positive": both_positive,
            "beats_every_control_aggregate": beats_every_control,
            "nuisance_dependent": nuisance_dependent,
        },
        "decision_state": state,
        "original_resolution_severe_adjudication": None,
        "confirmatory_roll_pixels_decoded": True,
        "filmset_final_628_parsed_or_decoded": False,
        "claim_ceiling": policy["claim_ceiling"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"output": str(args.output), "decision_state": state}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
