"""Run the untouched CT5 confirmatory fold under the frozen pilot decision."""

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

from src.roll2film.baselines import (  # noqa: E402
    LAB_STATS_SCHEMA,
    SLICED_SCHEMA,
    LabMeanStdOperator,
    SlicedTransportOperator,
)
from src.roll2film.ct5_evaluation import (  # noqa: E402
    cluster_bootstrap_improvement,
    evaluate_sampled_candidate,
    paired_per_image_affine_oracle,
    summarize_operator_output,
)
from src.roll2film.operators import OPERATOR_SCHEMA, AffineColorOperator  # noqa: E402
from src.roll2film.splines import (  # noqa: E402
    L2_OPERATOR_SCHEMA,
    AffineMonotoneSplineOperator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct5_baselines.json",
    )
    parser.add_argument(
        "--implementation-config",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct5_implementation_v1.json",
    )
    parser.add_argument(
        "--pilot-decision",
        type=Path,
        default=ROOT / "configs" / "roll2film_ct5_pilot_decision.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "data_cache",
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "pilot_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "confirmatory_report.json",
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


def _load_cache(cache_dir: Path, name: str, expected: str) -> np.ndarray:
    path = cache_dir / name
    observed = _sha(path)
    if observed != expected:
        raise ValueError(f"cache hash mismatch for {name}: {observed}")
    return np.load(path, allow_pickle=False)


def _operator(payload: dict[str, Any]) -> Any:
    schema = payload.get("schema")
    if schema == OPERATOR_SCHEMA:
        return AffineColorOperator.from_dict(payload)
    if schema == LAB_STATS_SCHEMA:
        return LabMeanStdOperator.from_dict(payload)
    if schema == SLICED_SCHEMA:
        return SlicedTransportOperator.from_dict(payload)
    if schema == L2_OPERATOR_SCHEMA:
        return AffineMonotoneSplineOperator.from_dict(payload)
    raise ValueError(f"unsupported frozen CT5 operator schema: {schema!r}")


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy_config.read_text(encoding="utf-8"))
    implementation = json.loads(args.implementation_config.read_text(encoding="utf-8"))
    decision = json.loads(args.pilot_decision.read_text(encoding="utf-8"))
    if len({policy["experiment_id"], implementation["experiment_id"], decision["experiment_id"]}) != 1:
        raise ValueError("CT5 experiment IDs differ")
    if _sha(args.pilot_report) != decision["pilot_report_sha256"]:
        raise ValueError("frozen pilot report hash mismatch")
    pilot = json.loads(args.pilot_report.read_text(encoding="utf-8"))
    if pilot["final_628_parsed_or_decoded"] is not False:
        raise ValueError("pilot did not preserve final-628 seal")
    cache_dir = args.cache_dir.resolve()
    cache_report_path = cache_dir / "report.json"
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if cache_report["final_628_parsed_or_decoded"] is not False:
        raise ValueError("cache did not preserve final-628 seal")
    hashes = cache_report["arrays_sha256"]
    confirm_input = _load_cache(
        cache_dir, "dev_confirmatory_input.npy", hashes["dev_confirmatory_input.npy"]
    )
    membership_path = cache_dir / "membership.json"
    if _sha(membership_path) != hashes["membership.json"]:
        raise ValueError("CT5 membership hash mismatch")
    membership = json.loads(membership_path.read_text(encoding="utf-8"))["internal_dev"]
    confirm_members = [row for row in membership if row["fold"] == "confirmatory"]
    cluster_ids = tuple(str(row["cluster_id"]) for row in confirm_members)
    if len(cluster_ids) != len(confirm_input):
        raise ValueError("confirmatory cache/membership length mismatch")

    bootstrap = decision["bootstrap"]
    report_domains: dict[str, Any] = {}
    overall_candidates: dict[str, list[bool]] = {
        name: [] for name in decision["confirmatory_candidates"]
    }
    for domain_index, domain in enumerate(policy["dataset"]["domains"]):
        target = _load_cache(
            cache_dir,
            f"dev_confirmatory_target_{domain}.npy",
            hashes[f"dev_confirmatory_target_{domain}.npy"],
        )
        bundles = pilot["operator_bundles"][domain]
        oracle = paired_per_image_affine_oracle(confirm_input, target)
        oracle_summary, oracle_per_image = evaluate_sampled_candidate(
            confirm_input, target, oracle
        )
        domain_candidates: dict[str, Any] = {}
        for candidate_index, name in enumerate(decision["confirmatory_candidates"]):
            stratum = decision["candidate_stratum_by_domain"][domain][name]
            basic_name = decision["best_basic_by_domain_and_stratum"][domain][stratum]
            candidate_output = summarize_operator_output(_operator(bundles[name]), confirm_input)
            basic_output = summarize_operator_output(_operator(bundles[basic_name]), confirm_input)
            candidate_summary, candidate_per_image = evaluate_sampled_candidate(
                confirm_input, target, candidate_output
            )
            basic_summary, basic_per_image = evaluate_sampled_candidate(
                confirm_input, target, basic_output
            )
            comparison = cluster_bootstrap_improvement(
                [row["mean_delta_e00_to_target"] for row in basic_per_image],
                [row["mean_delta_e00_to_target"] for row in candidate_per_image],
                cluster_ids,
                seed=int(bootstrap["seed"]) + domain_index * 100 + candidate_index,
                resamples=int(bootstrap["resamples"]),
            )
            style_pass = (
                float(candidate_summary["median_delta_e00_from_input"])
                >= float(basic_summary["median_delta_e00_from_input"])
                * float(policy["promotion"]["style_floor_relative_to_best_basic"])
            )
            fidelity_pass = float(comparison["ci95_low"]) > 0.0
            eligible = style_pass and fidelity_pass
            basic_gap = float(basic_summary["mean_delta_e00_to_target"]) - float(
                oracle_summary["mean_delta_e00_to_target"]
            )
            closure = (
                float(basic_summary["mean_delta_e00_to_target"])
                - float(candidate_summary["mean_delta_e00_to_target"])
            ) / max(basic_gap, 1e-12)
            overall_candidates[name].append(eligible)
            domain_candidates[name] = {
                "frozen_stratum": stratum,
                "frozen_best_basic": basic_name,
                "candidate_summary": candidate_summary,
                "best_basic_summary": basic_summary,
                "cluster_bootstrap_improvement": comparison,
                "style_floor_pass": style_pass,
                "fidelity_ci_pass": fidelity_pass,
                "eligible_for_full_resolution_adjudication": eligible,
                "paired_affine_gap_closure": closure,
            }
        report_domains[domain] = {
            "paired_affine_oracle_summary": oracle_summary,
            "candidates": domain_candidates,
        }
    report = {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "fold": "confirmatory",
        "policy_config_sha256": _sha(args.policy_config),
        "implementation_config_sha256": _sha(args.implementation_config),
        "pilot_decision_sha256": _sha(args.pilot_decision),
        "pilot_report_sha256": _sha(args.pilot_report),
        "cache_report_sha256": _sha(cache_report_path),
        "software_commit": _commit(),
        "domains": report_domains,
        "candidate_all_domain_statistical_pass": {
            name: all(values) for name, values in overall_candidates.items()
        },
        "final_628_parsed_or_decoded": False,
        "decision_state": "statistical_confirmatory_only_full_resolution_severe_not_cleared",
        "claim_boundary": policy["claim_boundary"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(
        json.dumps(
            {
                "report": str(output),
                "all_domain_statistical_pass": report["candidate_all_domain_statistical_pass"],
                "final_628_parsed_or_decoded": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
