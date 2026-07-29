#!/usr/bin/env python
"""Validate ColorReference suite topology and freeze in-domain colour folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from scripts.run_u5_r2aq2d_colorreference_source_support_diagnostic import (  # noqa: E402
    load_source_grids,
)


CONFIG_SHA256 = "b6518b88ba89496512a0f6b0eeef59f92d5b65bdf2ccbf2110a475b985b75df2"
REPORT_SCHEMA = "neuro-film.u5.r2aq3t.calibration-suite-topology.v1"


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
            raise RuntimeError("AQ3T requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ3T config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2aq3t-colorreference-calibration-suite-topology-v1"
        or config["fit_allowed"]
        or config["target_fields_allowed"]
        or config["aq2_reopened"]
        or config["operator_promotion_allowed"]
    ):
        raise ValueError("AQ3T frozen contract mismatch")
    return config


def colour_fold(rgb_uint8: np.ndarray, *, salt: str) -> np.ndarray:
    values = np.asarray(rgb_uint8)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
        or np.any(values > 255)
    ):
        raise ValueError("colour folds require uint8-compatible Nx3")
    prefix = salt.encode("utf-8")
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(prefix + bytes(row.tolist())).digest()[:8],
                "big",
            )
            % 5
            for row in values.astype(np.uint8)
        ],
        dtype=np.int64,
    )


def _support_metrics(
    held: np.ndarray,
    development: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, float | int]:
    held_float = held.astype(np.float64) / 255.0
    development_float = development.astype(np.float64) / 255.0
    tree = cKDTree(development_float)
    held_distance = tree.query(held_float, k=1, workers=1)[0]
    development_distance = tree.query(
        development_float, k=2, workers=1
    )[0][:, 1]
    held_p95 = float(np.percentile(held_distance, 95.0))
    development_p95 = float(
        np.percentile(development_distance, 95.0)
    )
    hull = ConvexHull(development_float)
    signed = (
        held_float @ hull.equations[:, :-1].T
        + hull.equations[:, -1]
    )
    return {
        "held_rows": int(held.shape[0]),
        "development_rows": int(development.shape[0]),
        "held_nearest_distance_median": float(
            np.median(held_distance)
        ),
        "held_nearest_distance_p95": held_p95,
        "development_nearest_distance_p95": development_p95,
        "p95_nearest_distance_ratio": held_p95 / development_p95,
        "outside_development_convex_hull_fraction": float(
            np.mean(np.any(signed > tolerance, axis=1))
        ),
    }


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    table = ROOT / config["parent"]["pair_table"]
    grids = load_source_grids(
        table,
        expected_sha256=config["parent"]["pair_table_sha256"],
    )
    topology = config["topology_contract"]
    suite = np.concatenate(
        [grids[slide] for slide in (1, 2, 3, 4)], axis=0
    )
    codes = np.rint(suite * 255.0).astype(np.uint8)
    prefix = codes[: topology["ordered_grid_prefix_rows"]]
    hard = codes[topology["ordered_grid_prefix_rows"] :]
    red_levels = set(topology["nominal_grid_levels_r_uint8"])
    gb_levels = set(topology["nominal_grid_levels_gb_uint8"])
    on_nominal = np.asarray(
        [
            int(red) in red_levels
            and int(green) in gb_levels
            and int(blue) in gb_levels
            for red, green, blue in prefix
        ]
    )
    off_indices = np.flatnonzero(~on_nominal).tolist()
    off_rows = prefix[~on_nominal]
    all_source = np.concatenate(
        [grids[slide] for slide in (1, 2, 3, 4, 5)], axis=0
    )
    all_codes = np.rint(all_source * 255.0).astype(np.uint8)
    fold_contract = config["source_colour_fold_contract"]
    folds = colour_fold(all_codes, salt=fold_contract["salt"])
    counts = np.bincount(folds, minlength=5).tolist()
    fold_metrics = []
    for fold in range(5):
        fold_metrics.append(
            {
                "fold": fold,
                **_support_metrics(
                    all_codes[folds == fold],
                    all_codes[folds != fold],
                    tolerance=1e-10,
                ),
            }
        )
    duplicate_consistent = True
    seen: dict[tuple[int, int, int], int] = {}
    for row, fold in zip(all_codes, folds):
        key = tuple(int(value) for value in row)
        if key in seen and seen[key] != int(fold):
            duplicate_consistent = False
        seen[key] = int(fold)
    checks = [
        {
            "name": "calibration_suite_shape",
            "passed": suite.shape
            == (topology["calibration_suite_rows"], 3),
        },
        {
            "name": "nominal_grid_prefix_count",
            "passed": int(on_nominal.sum())
            == topology["grid_prefix_rows_on_nominal_levels"],
        },
        {
            "name": "nominal_grid_unique_count",
            "passed": int(np.unique(prefix[on_nominal], axis=0).shape[0])
            == topology["grid_prefix_unique_nominal_rows"],
        },
        {
            "name": "off_nominal_rows",
            "passed": off_indices
            == topology["grid_prefix_off_nominal_indices"]
            and bool(
                np.all(
                    off_rows
                    == np.asarray(
                        topology["grid_prefix_off_nominal_rgb_uint8"]
                    )
                )
            ),
        },
        {
            "name": "hard_tail_unique_count",
            "passed": int(np.unique(hard, axis=0).shape[0])
            == topology["hard_tail_unique_rows"],
        },
        {
            "name": "fold_counts",
            "passed": counts == fold_contract["expected_fold_row_counts"],
        },
        {
            "name": "duplicate_grouping",
            "passed": duplicate_consistent,
        },
        {
            "name": "fold_local_support",
            "passed": all(
                row["p95_nearest_distance_ratio"]
                <= fold_contract["maximum_p95_nearest_distance_ratio"]
                for row in fold_metrics
            ),
        },
        {
            "name": "fold_convex_hull_support",
            "passed": all(
                row["outside_development_convex_hull_fraction"]
                <= fold_contract[
                    "maximum_outside_development_convex_hull_fraction"
                ]
                for row in fold_metrics
            ),
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "topology": {
            "suite_rows": int(suite.shape[0]),
            "grid_prefix_on_nominal_rows": int(on_nominal.sum()),
            "grid_prefix_unique_nominal_rows": int(
                np.unique(prefix[on_nominal], axis=0).shape[0]
            ),
            "off_nominal_indices": off_indices,
            "off_nominal_rows": off_rows.tolist(),
            "hard_tail_unique_rows": int(
                np.unique(hard, axis=0).shape[0]
            ),
        },
        "fold_counts": counts,
        "fold_metrics": fold_metrics,
        "automatic_checks": checks,
        "automatic_pass": passed,
        "decision": config["decision_branches"][
            "pass" if passed else "fail"
        ],
        "aq2_reopened": False,
        "fit_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, CONFIG_SHA256)
    report = run_audit(config)
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "fold_counts": report["fold_counts"],
                "fold_metrics": report["fold_metrics"],
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
        / "configs/u5_r2aq3t_colorreference_calibration_suite_topology_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256", default=CONFIG_SHA256
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq3t_colorreference_calibration_suite_topology_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
