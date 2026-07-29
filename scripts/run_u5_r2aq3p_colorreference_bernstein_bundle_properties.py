#!/usr/bin/env python
"""Compile and audit the held-validated bounded recorder proxy bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from scripts.run_u5_r2aq2_colorreference_velvia100f_held_group_baselines import (  # noqa: E402
    load_pair_table,
)
from src.roll2film.colorreference_bounded_bernstein import (  # noqa: E402
    BoundedBernsteinModel,
    apply_bounded_bernstein,
    bernstein_tensor_basis,
    fit_bounded_bernstein,
)
from src.roll2film.colorreference_proxy_baselines import (  # noqa: E402
    xyz_to_lab_d50,
)


CONFIG_SHA256 = "2c29507a4cb0a5e0ce53efdea1e565c23f6399122553b8e3472522aac9ea481b"
EXPERIMENT_ID = "u5.r2aq3p-colorreference-bernstein-bundle-properties-v1"
REPORT_SCHEMA = "neuro-film.u5.r2aq3p.bernstein-bundle-properties.v1"
BUNDLE_SCHEMA = "neuro-film.colorreference-recorder-proxy-bundle.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AQ3P requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ3P config hash mismatch")
    config = json.loads(raw)
    bundle = config["bundle_contract"]
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or bundle["model_id"] != "bernstein_d3"
        or bundle["degree"] != 3
        or bundle["post_fit_clipping"]
        or not config["fit_allowed"]
        or config["training_allowed"]
        or config["render_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AQ3P frozen contract mismatch")
    return config


def regular_grid(size: int) -> np.ndarray:
    if size < 2:
        raise ValueError("grid size must be at least two")
    levels = np.linspace(0.0, 1.0, size, dtype=np.float64)
    red, green, blue = np.meshgrid(
        levels, levels, levels, indexing="ij"
    )
    return np.column_stack(
        (red.reshape(-1), green.reshape(-1), blue.reshape(-1))
    )


def local_axis_distances(values: np.ndarray, *, size: int) -> np.ndarray:
    grid = np.asarray(values, dtype=np.float64)
    if grid.shape != (size**3, 3):
        raise ValueError("grid output shape mismatch")
    volume = grid.reshape(size, size, size, 3)
    distances = [
        np.linalg.norm(np.diff(volume, axis=axis), axis=-1).reshape(-1)
        for axis in range(3)
    ]
    return np.concatenate(distances)


def _load_parent(config: dict[str, Any]) -> dict[str, Any]:
    payload = (ROOT / config["parent"]["report"]).read_bytes()
    if _sha256(payload) != config["parent"]["report_sha256"]:
        raise ValueError("AQ3P parent report hash mismatch")
    report = json.loads(payload)
    if (
        not report["automatic_pass"]
        or report["champion"] != "bernstein_d3"
        or report["decision"]
        != "retain_simplest_eligible_bounded_bernstein_proxy_and_open_global_bundle_property_audit"
    ):
        raise ValueError("AQ3P parent result mismatch")
    return report


def _model_from_dict(payload: dict[str, Any]) -> BoundedBernsteinModel:
    return BoundedBernsteinModel(
        degree=int(payload["degree"]),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        lower_bound=float(payload["lower_bound"]),
        upper_bound=float(payload["upper_bound"]),
    )


def run_experiment(
    config: dict[str, Any],
) -> tuple[dict[str, Any], bytes, bytes]:
    parent = _load_parent(config)
    table = load_pair_table(
        ROOT / config["parent"]["pair_table"],
        expected_sha256=config["parent"]["pair_table_sha256"],
    )
    numerics = config["fit_numerics"]
    bundle_contract = config["bundle_contract"]
    model = fit_bounded_bernstein(
        table["source"],
        table["xyz"],
        degree=bundle_contract["degree"],
        lower_bound=numerics["coefficient_lower_bound"],
        upper_bound=numerics["coefficient_upper_bound"],
        tolerance=numerics["lsq_linear_tolerance"],
        maximum_iterations=numerics[
            "lsq_linear_maximum_iterations"
        ],
    )
    commit = _git_commit()
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "model_id": model.model_id,
        "degree": model.degree,
        "input_semantics": bundle_contract["input_semantics"],
        "output_semantics": bundle_contract["output_semantics"],
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "fit_config_sha256": CONFIG_SHA256,
        "software_commit": commit,
        "lower_bound": model.lower_bound,
        "upper_bound": model.upper_bound,
        "coefficients": model.coefficients.tolist(),
        "post_fit_clipping": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    bundle_bytes = _canonical_json(bundle)
    audit = config["property_audit"]
    dense_rgb = regular_grid(audit["dense_grid_size"])
    dense_basis = bernstein_tensor_basis(
        dense_rgb, degree=model.degree
    )
    partition_error = float(
        np.max(np.abs(dense_basis.sum(axis=1) - 1.0))
    )
    dense_xyz = dense_basis @ model.coefficients
    dense_f32 = np.ascontiguousarray(dense_xyz.astype("<f4")).tobytes()
    local = local_axis_distances(
        dense_xyz, size=audit["dense_grid_size"]
    )
    spans = np.ptp(dense_xyz, axis=0)
    cross_rgb = regular_grid(audit["cross_fold_grid_size"])
    global_cross_xyz = apply_bounded_bernstein(model, cross_rgb)
    global_cross_lab = xyz_to_lab_d50(global_cross_xyz)
    cross_fold = []
    for row in parent["joint_folds"]:
        fold_model = _model_from_dict(
            row["models"]["bernstein_d3"]["model"]
        )
        fold_lab = xyz_to_lab_d50(
            apply_bounded_bernstein(fold_model, cross_rgb)
        )
        delta = np.linalg.norm(fold_lab - global_cross_lab, axis=1)
        cross_fold.append(
            {
                "held_target_set": row["held_target_set"],
                "held_source_colour_fold": row[
                    "held_source_colour_fold"
                ],
                "median_deltae76": float(np.median(delta)),
                "p95_deltae76": float(np.percentile(delta, 95.0)),
                "maximum_deltae76": float(np.max(delta)),
            }
        )
    train_xyz = apply_bounded_bernstein(model, table["source"])
    train_delta = np.linalg.norm(
        xyz_to_lab_d50(train_xyz) - table["lab"], axis=1
    )
    checks = [
        {
            "name": "bundle_shape_and_coefficient_bounds",
            "passed": model.coefficients.shape == (64, 3)
            and bool(
                np.all(
                    model.coefficients
                    >= bundle_contract["coefficient_lower_bound"]
                )
            )
            and bool(
                np.all(
                    model.coefficients
                    <= bundle_contract["coefficient_upper_bound"]
                )
            ),
        },
        {
            "name": "dense_grid_shape",
            "passed": dense_xyz.shape == (audit["dense_grid_rows"], 3),
        },
        {
            "name": "dense_grid_finite_and_bounded",
            "passed": bool(np.all(np.isfinite(dense_xyz)))
            and float(np.min(dense_xyz)) >= audit["output_lower_bound"]
            and float(np.max(dense_xyz)) <= audit["output_upper_bound"],
        },
        {
            "name": "dense_grid_noncollapsed_channel_spans",
            "passed": bool(
                np.all(
                    spans
                    >= audit["minimum_output_span_each_xyz_channel"]
                )
            ),
        },
        {
            "name": "partition_of_unity",
            "passed": partition_error
            <= audit["analytic_partition_of_unity_tolerance"],
        },
        {
            "name": "local_axis_neighbor_p99",
            "passed": float(np.percentile(local, 99.0))
            <= audit["local_axis_neighbor_xyz_distance_p99_max"],
        },
        {
            "name": "local_axis_neighbor_maximum",
            "passed": float(np.max(local))
            <= audit[
                "local_axis_neighbor_xyz_distance_maximum_max"
            ],
        },
        {
            "name": "cross_fold_median_stability",
            "passed": all(
                row["median_deltae76"]
                <= audit[
                    "cross_fold_to_global_deltae76_median_each_fold_max"
                ]
                for row in cross_fold
            ),
        },
        {
            "name": "cross_fold_p95_stability",
            "passed": all(
                row["p95_deltae76"]
                <= audit[
                    "cross_fold_to_global_deltae76_p95_each_fold_max"
                ]
                for row in cross_fold
            ),
        },
        {
            "name": "cross_fold_maximum_stability",
            "passed": all(
                row["maximum_deltae76"]
                <= audit[
                    "cross_fold_to_global_deltae76_maximum_each_fold_max"
                ]
                for row in cross_fold
            ),
        },
    ]
    passed = all(check["passed"] for check in checks)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": CONFIG_SHA256,
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "parent_report_sha256": config["parent"]["report_sha256"],
        "bundle_sha256": _sha256(bundle_bytes),
        "dense_grid_xyz_f32_sha256": _sha256(dense_f32),
        "coefficient_minimum": float(np.min(model.coefficients)),
        "coefficient_maximum": float(np.max(model.coefficients)),
        "coefficient_lower_bound_count": int(
            np.sum(model.coefficients == model.lower_bound)
        ),
        "coefficient_upper_bound_count": int(
            np.sum(model.coefficients == model.upper_bound)
        ),
        "dense_grid": {
            "size": audit["dense_grid_size"],
            "rows": int(dense_xyz.shape[0]),
            "minimum_xyz": np.min(dense_xyz, axis=0).tolist(),
            "maximum_xyz": np.max(dense_xyz, axis=0).tolist(),
            "span_xyz": spans.tolist(),
            "partition_of_unity_maximum_error": partition_error,
            "local_axis_neighbor_xyz_distance_p99": float(
                np.percentile(local, 99.0)
            ),
            "local_axis_neighbor_xyz_distance_maximum": float(
                np.max(local)
            ),
        },
        "cross_fold_stability": cross_fold,
        "cross_fold_worst_median_deltae76": max(
            row["median_deltae76"] for row in cross_fold
        ),
        "cross_fold_worst_p95_deltae76": max(
            row["p95_deltae76"] for row in cross_fold
        ),
        "cross_fold_worst_maximum_deltae76": max(
            row["maximum_deltae76"] for row in cross_fold
        ),
        "all_row_fit_diagnostic": {
            "mean_deltae76": float(np.mean(train_delta)),
            "median_deltae76": float(np.median(train_delta)),
            "p95_deltae76": float(np.percentile(train_delta, 95.0)),
            "maximum_deltae76": float(np.max(train_delta)),
        },
        "automatic_checks": checks,
        "automatic_pass": passed,
        "decision": config["decision_branches"][
            "pass" if passed else "fail"
        ],
        "post_fit_clipping": False,
        "render_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, bundle_bytes, dense_f32


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, CONFIG_SHA256)
    report, bundle_bytes, dense_f32 = run_experiment(config)
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "bundle.json", bundle_bytes)
    _atomic_write(output_root / "dense_grid_xyz_f32.bin", dense_f32)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "bundle_sha256": report["bundle_sha256"],
                "decision": report["decision"],
                "dense_grid": report["dense_grid"],
                "cross_fold_worst_median_deltae76": report[
                    "cross_fold_worst_median_deltae76"
                ],
                "cross_fold_worst_p95_deltae76": report[
                    "cross_fold_worst_p95_deltae76"
                ],
                "cross_fold_worst_maximum_deltae76": report[
                    "cross_fold_worst_maximum_deltae76"
                ],
                "report_sha256": _sha256(report_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aq3p_colorreference_bernstein_bundle_properties_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=CONFIG_SHA256,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq3p_colorreference_bernstein_bundle_properties_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
