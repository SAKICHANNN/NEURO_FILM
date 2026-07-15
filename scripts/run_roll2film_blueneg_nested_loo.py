"""Run the frozen RF1.2 BlueNeg nested leave-one-frame-out diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roll2film.baselines import fit_lab_mean_std_operator  # noqa: E402
from src.roll2film.blueneg import load_blueneg_transformations  # noqa: E402
from src.roll2film.blueneg_eval import load_aligned_blueneg_pair  # noqa: E402
from src.roll2film.blueneg_nested_loo import (  # noqa: E402
    count_preserving_label_permutation,
    nearest_other_rolls,
    robust_standardize,
    source_descriptor,
    stable_select,
    style_match_output,
)
from src.roll2film.ct5_evaluation import (  # noqa: E402
    cluster_bootstrap_improvement,
    evaluate_sampled_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_nested_loo.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_blueneg_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "blueneg_v1" / "nested_loo_report.json",
    )
    return parser.parse_args()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(base: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{base}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


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


def _sample(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    pixels = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if len(pixels) < count:
        raise ValueError("frame cannot satisfy frozen pixel budget")
    indices = np.random.default_rng(seed).choice(len(pixels), count, replace=False)
    return pixels[indices]


def _sample_aligned(
    source: np.ndarray, target: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    source_pixels = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target_pixels = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source_pixels.shape != target_pixels.shape or len(source_pixels) < count:
        raise ValueError("aligned query cannot satisfy frozen pixel budget")
    indices = np.random.default_rng(seed).choice(len(source_pixels), count, replace=False)
    return source_pixels[indices], target_pixels[indices]


def _verify_payloads(root: Path, inventory: list[dict[str, Any]], paths: set[str]) -> None:
    by_path = {str(row["path"]): row for row in inventory}
    for relative in sorted(paths):
        if relative not in by_path:
            raise ValueError(f"used payload absent from frozen inventory: {relative}")
        path = root / Path(*relative.split("/"))
        row = by_path[relative]
        if path.stat().st_size != int(row["size"]) or _sha(path) != str(row["sha256"]):
            raise ValueError(f"BlueNeg payload integrity failure: {relative}")


def _fit(
    names: tuple[str, ...],
    source_support: dict[str, np.ndarray],
    target_support: dict[str, np.ndarray],
) -> Any:
    return fit_lab_mean_std_operator(
        np.concatenate([source_support[name] for name in names]),
        np.concatenate([target_support[name] for name in names]),
    )


def _evaluate(
    source: np.ndarray,
    target: np.ndarray,
    correct_output: np.ndarray,
    output: np.ndarray,
    *,
    alpha_max: float,
) -> dict[str, Any]:
    raw_summary, _ = evaluate_sampled_candidate(
        source[None, ...], target[None, ...], output[None, ...]
    )
    matched, alpha, achieved = style_match_output(
        source, output, correct_output, alpha_max=alpha_max
    )
    matched_summary, _ = evaluate_sampled_candidate(
        source[None, ...], target[None, ...], matched[None, ...]
    )
    return {
        "raw": raw_summary,
        "style_matched": matched_summary,
        "style_match_alpha": alpha,
        "style_match_achieved_mean_delta_e00": achieved,
    }


def _mean(rows: list[float]) -> float:
    return float(np.mean(np.asarray(rows, dtype=np.float64)))


def _categorical_means(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        category = row[key]
        if isinstance(category, dict):
            category = json.dumps(category, sort_keys=True)
        grouped[str(category)].append(float(row[value]))
    return {
        category: {"queries": len(values), "mean": _mean(values)}
        for category, values in sorted(grouped.items())
    }


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_config.read_text(encoding="utf-8"))
    source_contract = policy["source_contract"]
    evidence = (ROOT / pilot["evidence_dir"]).resolve()
    root = (ROOT / pilot["root"]).resolve()
    bound_paths = {
        evidence / "report.json": source_contract["metadata_report_sha256"],
        evidence / "frames.jsonl": source_contract["frames_manifest_sha256"],
        evidence / "acquisition.json": source_contract["acquisition_manifest_sha256"],
        ROOT / "outputs" / "roll2film" / "blueneg_v1" / "download_report.json": source_contract["download_report_sha256"],
        root / "transformations.pkl": source_contract["transformations_sha256"],
        ROOT / "outputs" / "roll2film" / "blueneg_v1" / "development_report.json": source_contract["development_report_sha256"],
        ROOT / "outputs" / "roll2film" / "blueneg_v1" / "confirmatory_report.json": source_contract["confirmatory_report_sha256"],
        ROOT / "configs" / "roll2film_blueneg_confirmatory_decision.json": source_contract["confirmatory_decision_sha256"],
    }
    for path, expected in bound_paths.items():
        if _sha(path) != expected:
            raise ValueError(f"frozen upstream hash mismatch: {path}")

    all_rows = _jsonl(evidence / "frames.jsonl")
    rolls = tuple(policy["cohort"]["rolls"])
    rows = [
        row
        for row in all_rows
        if (
            str(row["roll_id"]) in rolls
            and row.get("pseudogt_path")
            and row.get("frame_role") != "support_source_only"
        )
    ]
    expected_counts = policy["cohort"]["complete_aligned_frames_per_roll"]
    actual_counts = {
        roll: sum(str(row["roll_id"]) == roll for row in rows) for roll in rolls
    }
    if actual_counts != expected_counts or len(rows) != 43:
        raise ValueError(f"unexpected RF1.2 cohort: {actual_counts}")
    download_report = json.loads(
        (ROOT / "outputs" / "roll2film" / "blueneg_v1" / "download_report.json").read_text(encoding="utf-8")
    )
    used_paths = {
        str(row[key])
        for row in rows
        for key in ("preview_path", "pseudogt_path")
    }
    _verify_payloads(root, download_report["file_inventory"], used_paths)
    transformations = load_blueneg_transformations(root / "transformations.pkl")
    operator_policy = policy["operator"]
    seed = int(policy["selection"]["seed"])

    source_support: dict[str, np.ndarray] = {}
    target_support: dict[str, np.ndarray] = {}
    query_source: dict[str, np.ndarray] = {}
    query_target: dict[str, np.ndarray] = {}
    descriptors: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: str(item["filename"])):
        name = str(row["filename"])
        source, target = load_aligned_blueneg_pair(
            root,
            row,
            transformations[name],
            minimum_side=int(operator_policy["minimum_aligned_side"]),
        )
        descriptors[name] = source_descriptor(source)
        source_support[name] = _sample(
            source,
            int(operator_policy["support_pixels_per_frame_per_domain"]),
            _seed(seed, f"support-source:{name}"),
        )
        target_support[name] = _sample(
            target,
            int(operator_policy["support_pixels_per_frame_per_domain"]),
            _seed(seed, f"support-target:{name}"),
        )
        query_source[name], query_target[name] = _sample_aligned(
            source,
            target,
            int(operator_policy["query_pixels_per_frame"]),
            _seed(seed, f"query:{name}"),
        )
        metadata[name] = row
    standardized = robust_standardize(descriptors)
    frame_to_roll = {name: str(row["roll_id"]) for name, row in metadata.items()}
    by_roll = {
        roll: tuple(sorted(name for name, value in frame_to_roll.items() if value == roll))
        for roll in rolls
    }
    shuffled = count_preserving_label_permutation(frame_to_roll, seed=seed)
    support_count = int(operator_policy["support_frames"])
    alpha_max = float(policy["strength_control"]["alpha_range"][1])
    query_results: list[dict[str, Any]] = []

    for query in sorted(metadata):
        query_roll = frame_to_roll[query]
        correct_names = stable_select(
            by_roll[query_roll],
            count=support_count,
            seed=seed,
            namespace=f"correct:{query}",
            exclude=query,
        )
        wrong_roll_names = {
            roll: stable_select(
                by_roll[roll],
                count=support_count,
                seed=seed,
                namespace=f"wrong:{query}:{roll}",
            )
            for roll in rolls
            if roll != query_roll
        }
        other_frames = tuple(
            name for roll, names in by_roll.items() if roll != query_roll for name in names
        )
        pooled_names = stable_select(
            other_frames,
            count=support_count,
            seed=seed,
            namespace=f"pooled:{query}",
        )
        retrieval_names = nearest_other_rolls(
            query, standardized, frame_to_roll, count=support_count
        )
        pseudo_label = shuffled[query]
        shuffled_names = stable_select(
            [name for name, label in shuffled.items() if label == pseudo_label],
            count=support_count,
            seed=seed,
            namespace=f"shuffled:{query}",
            exclude=query,
        )
        support_sets = {
            "correct_roll": correct_names,
            "pooled_wrong": pooled_names,
            "content_retrieval_wrong": retrieval_names,
            "shuffled_roll": shuffled_names,
            **{
                f"wrong_roll_{roll}": names for roll, names in wrong_roll_names.items()
            },
        }
        operators = {
            arm: _fit(names, source_support, target_support)
            for arm, names in support_sets.items()
        }
        source = query_source[query]
        target = query_target[query]
        correct_output = operators["correct_roll"].apply(source)
        arm_results = {
            arm: _evaluate(
                source,
                target,
                correct_output,
                operator.apply(source),
                alpha_max=alpha_max,
            )
            for arm, operator in operators.items()
        }
        arm_results["identity"] = _evaluate(
            source, target, correct_output, source, alpha_max=alpha_max
        )
        correct_center = np.mean(
            [standardized[name] for name in correct_names], axis=0
        )
        retrieval_center = np.mean(
            [standardized[name] for name in retrieval_names], axis=0
        )
        query_results.append(
            {
                "query": query,
                "roll_id": query_roll,
                "location": metadata[query]["location"],
                "scene_property": metadata[query]["scene_property"],
                "supports": {arm: list(names) for arm, names in support_sets.items()},
                "descriptor_distance_to_correct_support": float(
                    np.linalg.norm(standardized[query] - correct_center)
                ),
                "descriptor_distance_to_retrieval_support": float(
                    np.linalg.norm(standardized[query] - retrieval_center)
                ),
                "correct_support_same_location_fraction": float(
                    np.mean(
                        [metadata[name]["location"] == metadata[query]["location"] for name in correct_names]
                    )
                ),
                "retrieval_support_same_location_fraction": float(
                    np.mean(
                        [metadata[name]["location"] == metadata[query]["location"] for name in retrieval_names]
                    )
                ),
                "correct_support_same_scene_property_fraction": float(
                    np.mean(
                        [metadata[name]["scene_property"] == metadata[query]["scene_property"] for name in correct_names]
                    )
                ),
                "retrieval_support_same_scene_property_fraction": float(
                    np.mean(
                        [metadata[name]["scene_property"] == metadata[query]["scene_property"] for name in retrieval_names]
                    )
                ),
                "arms": arm_results,
            }
        )

    common_arms = (
        "identity",
        "pooled_wrong",
        "content_retrieval_wrong",
        "shuffled_roll",
    )
    summaries: dict[str, Any] = {}
    for mode in ("raw", "style_matched"):
        correct_values = np.asarray(
            [row["arms"]["correct_roll"][mode]["mean_delta_e00_to_target"] for row in query_results]
        )
        arm_means = {
            arm: _mean(
                [row["arms"][arm][mode]["mean_delta_e00_to_target"] for row in query_results]
            )
            for arm in ("correct_roll", *common_arms)
        }
        best_control = min(common_arms, key=lambda arm: arm_means[arm])
        best_values = np.asarray(
            [row["arms"][best_control][mode]["mean_delta_e00_to_target"] for row in query_results]
        )
        wrong_mean_values = np.asarray(
            [
                np.mean(
                    [
                        arm[mode]["mean_delta_e00_to_target"]
                        for name, arm in row["arms"].items()
                        if name.startswith("wrong_roll_")
                    ]
                )
                for row in query_results
            ]
        )
        wrong_oracle_values = np.asarray(
            [
                min(
                    arm[mode]["mean_delta_e00_to_target"]
                    for name, arm in row["arms"].items()
                    if name.startswith("wrong_roll_")
                )
                for row in query_results
            ]
        )
        clusters = [row["roll_id"] for row in query_results]
        versus = {}
        for arm in common_arms:
            values = np.asarray(
                [row["arms"][arm][mode]["mean_delta_e00_to_target"] for row in query_results]
            )
            versus[arm] = {
                "mean_gain": float(np.mean(values - correct_values)),
                "correct_win_rate": float(np.mean(correct_values < values)),
                "bootstrap": cluster_bootstrap_improvement(
                    values,
                    correct_values,
                    clusters,
                    seed=int(policy["statistics"]["bootstrap_seed"]),
                    resamples=int(policy["statistics"]["bootstrap_resamples"]),
                ),
            }
        per_roll = {}
        for roll in rolls:
            indices = np.asarray([row["roll_id"] == roll for row in query_results])
            per_roll[roll] = {
                "queries": int(np.sum(indices)),
                "correct_delta_e00": float(np.mean(correct_values[indices])),
                "best_control_delta_e00": float(np.mean(best_values[indices])),
                "gain": float(np.mean(best_values[indices] - correct_values[indices])),
            }
        summaries[mode] = {
            "arm_means": arm_means,
            "best_common_control": best_control,
            "versus_controls": versus,
            "individual_wrong_roll_diagnostic": {
                "mean_of_wrong_rolls_delta_e00": float(np.mean(wrong_mean_values)),
                "per_query_oracle_wrong_roll_delta_e00": float(np.mean(wrong_oracle_values)),
                "gain_vs_mean_wrong_roll": float(np.mean(wrong_mean_values - correct_values)),
                "gain_vs_per_query_oracle_wrong_roll": float(np.mean(wrong_oracle_values - correct_values)),
                "correct_win_rate_vs_mean_wrong_roll": float(np.mean(correct_values < wrong_mean_values)),
                "correct_win_rate_vs_per_query_oracle_wrong_roll": float(np.mean(correct_values < wrong_oracle_values)),
                "oracle_is_diagnostic_not_a_fixed_deployable_control": True,
            },
            "per_roll_vs_best_common_control": per_roll,
        }

    best_matched = summaries["style_matched"]["best_common_control"]
    for row in query_results:
        row["style_matched_gain_vs_best_common_control"] = float(
            row["arms"][best_matched]["style_matched"]["mean_delta_e00_to_target"]
            - row["arms"]["correct_roll"]["style_matched"]["mean_delta_e00_to_target"]
        )
    gains = np.asarray(
        [row["style_matched_gain_vs_best_common_control"] for row in query_results]
    )
    distances = np.asarray(
        [row["descriptor_distance_to_correct_support"] for row in query_results]
    )
    correlation = float(np.corrcoef(gains, distances)[0, 1])
    matched_roll_gains = summaries["style_matched"]["per_roll_vs_best_common_control"]
    all_rolls_positive = all(float(row["gain"]) > 0.0 for row in matched_roll_gains.values())
    all_controls_positive = all(
        float(row["mean_gain"]) > 0.0
        for row in summaries["style_matched"]["versus_controls"].values()
    )
    lower_bounds_positive = all(
        float(row["bootstrap"]["ci95_low"]) > 0.0
        for row in summaries["style_matched"]["versus_controls"].values()
    )
    if all_rolls_positive and all_controls_positive and lower_bounds_positive:
        decision = "diagnostic_support"
    elif any(
        summaries["style_matched"]["arm_means"][arm]
        < summaries["style_matched"]["arm_means"]["correct_roll"]
        for arm in ("pooled_wrong", "content_retrieval_wrong", "shuffled_roll")
    ):
        decision = "nuisance_explanation"
    else:
        decision = "heterogeneous_or_unidentified"
    report = {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "software_commit": _commit(),
        "policy_sha256": _sha(args.policy),
        "upstream_hashes_verified_before_decode": True,
        "used_payloads_verified_before_decode": len(used_paths),
        "cohort": {
            "queries": len(query_results),
            "roll_counts": actual_counts,
            "query_excluded_from_support": True,
        },
        "summaries": summaries,
        "associations": {
            "pearson_gain_vs_correct_support_descriptor_distance": correlation,
            "mean_correct_support_descriptor_distance": _mean(
                [row["descriptor_distance_to_correct_support"] for row in query_results]
            ),
            "mean_retrieval_support_descriptor_distance": _mean(
                [row["descriptor_distance_to_retrieval_support"] for row in query_results]
            ),
            "mean_correct_support_same_location_fraction": _mean(
                [row["correct_support_same_location_fraction"] for row in query_results]
            ),
            "mean_retrieval_support_same_location_fraction": _mean(
                [row["retrieval_support_same_location_fraction"] for row in query_results]
            ),
            "mean_correct_support_same_scene_property_fraction": _mean(
                [row["correct_support_same_scene_property_fraction"] for row in query_results]
            ),
            "mean_retrieval_support_same_scene_property_fraction": _mean(
                [row["retrieval_support_same_scene_property_fraction"] for row in query_results]
            ),
            "by_location": _categorical_means(
                query_results, "location", "style_matched_gain_vs_best_common_control"
            ),
            "by_scene_property": _categorical_means(
                query_results, "scene_property", "style_matched_gain_vs_best_common_control"
            ),
        },
        "decision_checks": {
            "all_four_roll_gains_positive": all_rolls_positive,
            "mean_gain_positive_against_all_common_controls": all_controls_positive,
            "cluster_ci_lower_positive_against_all_common_controls": lower_bounds_positive,
        },
        "decision": decision,
        "claim_ceiling": policy["claim_ceiling"],
        "queries": query_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(encoded)
    print(json.dumps({
        "report": str(args.output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "decision": decision,
        "style_matched": summaries["style_matched"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
