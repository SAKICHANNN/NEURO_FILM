"""Fit frozen CT5 CPU baselines and evaluate the internal pilot fold only."""

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
    fit_basic_adjustment_family,
    fit_lab_mean_std_operator,
    fit_per_channel_quantile_operator,
    fit_sliced_transport_operator,
)
from src.roll2film.ct5_evaluation import (  # noqa: E402
    evaluate_sampled_candidate,
    paired_per_image_affine_oracle,
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
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "data_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "roll2film" / "ct5_v1" / "pilot_report.json",
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


def _operator_dict(operator: Any) -> dict[str, Any]:
    payload = operator.to_dict()
    payload["fitted_from"] = "content-disjoint unpaired FilmSet training manifests"
    return payload


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy_config.read_text(encoding="utf-8"))
    implementation = json.loads(args.implementation_config.read_text(encoding="utf-8"))
    if policy["experiment_id"] != implementation["experiment_id"]:
        raise ValueError("CT5 policy and implementation experiment IDs differ")
    cache_dir = args.cache_dir.resolve()
    cache_report = json.loads((cache_dir / "report.json").read_text(encoding="utf-8"))
    if cache_report["config_sha256"] != _sha(args.policy_config):
        raise ValueError("CT5 cache was built under a different policy config")
    if cache_report["final_628_parsed_or_decoded"] is not False:
        raise ValueError("CT5 cache does not preserve the final-628 seal")
    hashes = cache_report["arrays_sha256"]
    source_images = _load_cache(cache_dir, "training_source.npy", hashes["training_source.npy"])
    pilot_input = _load_cache(cache_dir, "dev_pilot_input.npy", hashes["dev_pilot_input.npy"])
    source = source_images.reshape(-1, 3)
    fit_config = implementation["training_fit"]
    quantiles = tuple(float(value) for value in fit_config["quantiles"])
    report_domains: dict[str, Any] = {}
    operator_bundles: dict[str, Any] = {}
    for domain_index, domain in enumerate(policy["dataset"]["domains"]):
        print(f"CT5 pilot fitting {domain}", flush=True)
        target_images = _load_cache(
            cache_dir,
            f"training_target_{domain}.npy",
            hashes[f"training_target_{domain}.npy"],
        )
        pilot_target = _load_cache(
            cache_dir,
            f"dev_pilot_target_{domain}.npy",
            hashes[f"dev_pilot_target_{domain}.npy"],
        )
        target = target_images.reshape(-1, 3)
        operators: dict[str, Any] = {"identity": AffineColorOperator.identity()}
        operators.update(fit_basic_adjustment_family(source, target))
        operators["lab_mean_std"] = fit_lab_mean_std_operator(source, target)
        operators["per_channel_quantile"] = fit_per_channel_quantile_operator(
            source, target, quantiles=quantiles
        )
        operators["gaussian_bures"] = estimate_gaussian_transport_operator(source, [target])
        operators["sliced_ot"] = fit_sliced_transport_operator(
            source,
            target,
            iterations=int(fit_config["sliced_iterations"]),
            seed=int(policy["seed"]) + int(fit_config["sliced_seed_offset"]) + domain_index,
            quantiles=quantiles,
        )
        operators["pooled_l2"] = estimate_affine_spline_transport_operator(
            source,
            [target],
            knot_quantiles=quantiles,
            iterations=int(fit_config["pooled_l2_iterations"]),
        )
        domain_results: dict[str, Any] = {}
        operator_bundles[domain] = {}
        for name in policy["baseline_order"]:
            if name == "paired_per_image_oracle":
                rendered = paired_per_image_affine_oracle(pilot_input, pilot_target)
                operator_bundles[domain][name] = {
                    "schema": "evaluator_only.per_image_affine_oracle.v1",
                    "deployable": False,
                }
            else:
                operator = operators[name]
                rendered = summarize_operator_output(operator, pilot_input)
                operator_bundles[domain][name] = _operator_dict(operator)
            summary, per_image = evaluate_sampled_candidate(
                pilot_input,
                pilot_target,
                rendered,
            )
            domain_results[name] = {"summary": summary, "per_image": per_image}
        basic_names = [
            name
            for name, role in policy["baseline_roles"].items()
            if role == "best_basic_candidate"
        ]
        best_basic = min(
            basic_names,
            key=lambda name: domain_results[name]["summary"]["mean_delta_e00_to_target"],
        )
        report_domains[domain] = {
            "best_basic_pilot": best_basic,
            "candidates": domain_results,
        }
    report = {
        "schema_version": 1,
        "experiment_id": policy["experiment_id"],
        "fold": "pilot",
        "policy_config_sha256": _sha(args.policy_config),
        "implementation_config_sha256": _sha(args.implementation_config),
        "cache_report_sha256": _sha(cache_dir / "report.json"),
        "software_commit": _commit(),
        "domains": report_domains,
        "operator_bundles": operator_bundles,
        "final_628_parsed_or_decoded": False,
        "sampled_metric_limits": implementation["sampled_metric_limit"],
        "decision_state": "pilot_only_strength_strata_and_best_basic_not_yet_frozen",
        "claim_boundary": policy["claim_boundary"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(
        json.dumps(
            {
                "report": str(output),
                "best_basic": {
                    domain: result["best_basic_pilot"] for domain, result in report_domains.items()
                },
                "final_628_parsed_or_decoded": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
