"""Select one BlueNeg operator family using development rolls only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.baselines import (  # noqa: E402
    fit_basic_adjustment_family,
    fit_lab_mean_std_operator,
    fit_per_channel_quantile_operator,
)
from src.roll2film.blueneg import load_blueneg_transformations  # noqa: E402
from src.roll2film.blueneg_eval import load_development_roll_samples  # noqa: E402
from src.roll2film.ct5_evaluation import (  # noqa: E402
    evaluate_sampled_candidate,
    summarize_operator_output,
)
from src.roll2film.identification import (  # noqa: E402
    estimate_affine_spline_transport_operator,
    estimate_gaussian_transport_operator,
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
        "--pilot-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "development_report.json",
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fit_candidates(source: Any, target: Any) -> dict[str, Any]:
    candidates: dict[str, Any] = {"identity": AffineColorOperator.identity()}
    candidates["wb_contrast_saturation"] = fit_basic_adjustment_family(source, target)[
        "wb_contrast_saturation"
    ]
    candidates["lab_mean_std"] = fit_lab_mean_std_operator(source, target)
    candidates["per_channel_quantile"] = fit_per_channel_quantile_operator(source, target)
    candidates["gaussian_bures"] = estimate_gaussian_transport_operator(source, [target])
    candidates["pooled_l2"] = estimate_affine_spline_transport_operator(source, [target])
    return candidates


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_config.read_text(encoding="utf-8"))
    source = policy["source_contract"]
    evidence = (ROOT / pilot["evidence_dir"]).resolve()
    root = (ROOT / pilot["root"]).resolve()
    hash_paths = {
        "metadata_report_sha256": evidence / "report.json",
        "frames_manifest_sha256": evidence / "frames.jsonl",
        "acquisition_manifest_sha256": evidence / "acquisition.json",
        "download_report_sha256": ROOT
        / "outputs"
        / "roll2film"
        / "blueneg_v1"
        / "download_report.json",
        "transformations_sha256": root / "transformations.pkl",
    }
    for key, path in hash_paths.items():
        if _sha(path) != source[key]:
            raise ValueError(f"BlueNeg frozen hash mismatch: {key}")
    download_report = json.loads(hash_paths["download_report_sha256"].read_text())
    if download_report["all_sizes_and_lfs_sha256_verified"] is not True:
        raise ValueError("BlueNeg payload verification did not pass")
    rows = _read_jsonl(hash_paths["frames_manifest_sha256"])
    transformations = load_blueneg_transformations(root / "transformations.pkl")
    budget = policy["fixed_budget"]
    alignment = policy["decode_and_alignment"]
    roll_samples = {}
    for roll_id in policy["matched_core"]["development_rolls"]:
        roll_samples[roll_id] = load_development_roll_samples(
            root=root,
            rows=rows,
            transformations=transformations,
            roll_id=roll_id,
            support_frames=budget["support_frames_per_roll"],
            support_pixels=budget["support_pixels_per_frame_per_domain"],
            query_frames=budget["development_query_frames_per_roll"],
            query_pixels=budget["query_pixels_per_frame"],
            support_seed=budget["support_target_permutation_seed"],
            query_seed=budget["query_sampling_seed"],
            minimum_side=alignment["minimum_aligned_side"],
        )
    results: dict[str, Any] = {}
    bundles: dict[str, Any] = {}
    order = policy["development_candidate_order"]
    for roll_id, samples in roll_samples.items():
        candidates = _fit_candidates(samples.support_source, samples.support_target)
        bundles[roll_id] = {name: candidates[name].to_dict() for name in order}
        roll_results = {}
        for name in order:
            output = summarize_operator_output(candidates[name], samples.query_source)
            summary, per_image = evaluate_sampled_candidate(
                samples.query_source,
                samples.query_target,
                output,
            )
            roll_results[name] = {"summary": summary, "per_query": per_image}
        results[roll_id] = {
            "support_content_ids": samples.support_content_ids,
            "query_content_ids": samples.query_content_ids,
            "candidates": roll_results,
        }
    selection = policy["development_selection"]
    aggregate = {}
    for name in order:
        summaries = [results[roll]["candidates"][name]["summary"] for roll in results]
        aggregate[name] = {
            "mean_delta_e00_to_hidden_query_target": sum(
                value["mean_delta_e00_to_target"] for value in summaries
            )
            / len(summaries),
            "median_style_delta_e00_from_query_input": sum(
                value["median_delta_e00_from_input"] for value in summaries
            )
            / len(summaries),
            "beats_identity_on_both_rolls": all(
                results[roll]["candidates"][name]["summary"]["mean_delta_e00_to_target"]
                < results[roll]["candidates"]["identity"]["summary"][
                    "mean_delta_e00_to_target"
                ]
                for roll in results
            ),
        }
    eligible = [
        name
        for name in order
        if name != "identity"
        and aggregate[name]["beats_identity_on_both_rolls"]
        and aggregate[name]["median_style_delta_e00_from_query_input"]
        >= selection["style_floor_median_delta_e00_from_query_input"]
    ]
    selected = None
    if eligible:
        best = min(aggregate[name]["mean_delta_e00_to_hidden_query_target"] for name in eligible)
        selected = next(
            name
            for name in order
            if name in eligible
            and aggregate[name]["mean_delta_e00_to_hidden_query_target"]
            <= best + selection["simplest_within_delta_e00_of_best"]
        )
    report = {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "policy_sha256": _sha(args.policy),
        "software_commit": _commit(),
        "development_rolls": list(results),
        "results": results,
        "aggregate": aggregate,
        "operator_bundles": bundles,
        "selected_family": selected,
        "decision_state": "family_selected" if selected else "stop_no_eligible_family",
        "confirmatory_roll_pixels_decoded": False,
        "filmset_final_628_parsed_or_decoded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_family": selected,
                "confirmatory_roll_pixels_decoded": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
