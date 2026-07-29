#!/usr/bin/env python
"""Attribute rejected AQ2 fits using source-only support geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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


CONFIG_SHA256 = "5818cf339be8c50dd209c15ee552839812b796c5fe1b3ed7f3eb8538a9b61114"
REPORT_SCHEMA = "neuro-film.u5.r2aq2d.source-support-diagnostic.v1"


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
            raise RuntimeError("AQ2D requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ2D config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2aq2d-colorreference-source-support-diagnostic-v1"
        or config["fit_allowed"]
        or config["target_fields_allowed"]
        or config["operator_promotion_allowed"]
        or config["capacity_rescue_allowed"]
        or config["render_allowed"]
    ):
        raise ValueError("AQ2D frozen contract mismatch")
    return config


def load_source_grids(
    path: Path, *, expected_sha256: str
) -> dict[int, np.ndarray]:
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("AQ2D pair-table hash mismatch")
    reader = csv.DictReader(
        io.StringIO(payload.decode("utf-8")), delimiter="\t"
    )
    by_key: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for row in reader:
        key = (int(row["test_set"]), int(row["slide_index"]))
        by_key.setdefault(key, []).append(
            (
                float(row["source_r"]),
                float(row["source_g"]),
                float(row["source_b"]),
            )
        )
    expected_keys = {
        (test_set, slide)
        for test_set in (1, 2, 3, 4, 5, 9)
        for slide in (1, 2, 3, 4, 5)
    }
    if set(by_key) != expected_keys:
        raise ValueError("AQ2D source key coverage mismatch")
    grids = {}
    for slide in (1, 2, 3, 4, 5):
        reference = np.asarray(by_key[(1, slide)], dtype=np.float64)
        if reference.shape != (288, 3):
            raise ValueError("AQ2D source grid shape mismatch")
        for test_set in (2, 3, 4, 5, 9):
            other = np.asarray(
                by_key[(test_set, slide)], dtype=np.float64
            )
            if not np.array_equal(reference, other):
                raise ValueError("AQ2D source grid drifts across sets")
        grids[slide] = reference
    return grids


def source_support_metrics(
    held: np.ndarray,
    development: np.ndarray,
    *,
    hull_tolerance: float,
) -> dict[str, float | int]:
    """Compute fixed nearest-neighbour and convex-hull support metrics."""

    query = np.asarray(held, dtype=np.float64)
    support = np.asarray(development, dtype=np.float64)
    if (
        query.shape != (288, 3)
        or support.shape != (1152, 3)
        or not np.all(np.isfinite(query))
        or not np.all(np.isfinite(support))
    ):
        raise ValueError("AQ2D support arrays have the wrong shape")
    tree = cKDTree(support)
    held_distance = tree.query(query, k=1, workers=1)[0]
    development_distance = tree.query(
        support, k=2, workers=1
    )[0][:, 1]
    hull = ConvexHull(support)
    signed = (
        query @ hull.equations[:, :-1].T
        + hull.equations[:, -1][None, :]
    )
    outside = np.any(signed > hull_tolerance, axis=1)
    support_tuples = {
        tuple(row.tolist()) for row in support
    }
    overlap = np.asarray(
        [tuple(row.tolist()) in support_tuples for row in query]
    )
    development_p95 = float(
        np.percentile(development_distance, 95.0)
    )
    held_p95 = float(np.percentile(held_distance, 95.0))
    return {
        "held_row_count": int(query.shape[0]),
        "development_row_count": int(support.shape[0]),
        "held_nearest_distance_median": float(
            np.median(held_distance)
        ),
        "held_nearest_distance_p95": held_p95,
        "held_nearest_distance_maximum": float(
            np.max(held_distance)
        ),
        "development_leave_one_nearest_distance_median": float(
            np.median(development_distance)
        ),
        "development_leave_one_nearest_distance_p95": development_p95,
        "held_to_development_p95_nearest_distance_ratio": (
            held_p95 / development_p95
            if development_p95 > 0.0
            else float("inf")
        ),
        "held_outside_development_convex_hull_fraction": float(
            np.mean(outside)
        ),
        "held_exact_rgb_overlap_fraction": float(np.mean(overlap)),
        "held_maximum_hull_equation_excess": float(
            np.max(np.maximum(signed, 0.0))
        ),
    }


def run_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    aq2_payload = (
        ROOT / config["parent"]["aq2_report"]
    ).read_bytes()
    if _sha256(aq2_payload) != config["parent"]["aq2_report_sha256"]:
        raise ValueError("AQ2D parent report hash mismatch")
    aq2 = json.loads(aq2_payload)
    if (
        aq2["automatic_pass"]
        or aq2["decision"]
        != "close_recorder_proxy_operator_identifiability"
    ):
        raise ValueError("AQ2D requires the exact rejected AQ2 parent")
    grids = load_source_grids(
        ROOT / config["parent"]["pair_table"],
        expected_sha256=config["parent"]["pair_table_sha256"],
    )
    tolerance = config["source_contract"]["convex_hull_tolerance"]
    per_slide = []
    for slide in config["source_contract"]["slide_indices"]:
        development = np.concatenate(
            [grid for key, grid in grids.items() if key != slide],
            axis=0,
        )
        metrics = source_support_metrics(
            grids[slide],
            development,
            hull_tolerance=tolerance,
        )
        per_slide.append({"held_slide_index": slide, **metrics})
    thresholds = config["diagnostic_thresholds"]
    gap_slides = [
        row["held_slide_index"]
        for row in per_slide
        if (
            row["held_to_development_p95_nearest_distance_ratio"]
            > thresholds[
                "held_to_development_p95_nearest_distance_ratio"
            ]
            or row[
                "held_outside_development_convex_hull_fraction"
            ]
            > thresholds[
                "held_outside_development_convex_hull_fraction"
            ]
        )
    ]
    decision = (
        config["decision_branches"]["source_support_gap"]
        if gap_slides
        else config["decision_branches"][
            "in_support_representation_failure"
        ]
    )
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "aq2_report_sha256": config["parent"]["aq2_report_sha256"],
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "per_slide": per_slide,
        "source_support_gap_slides": gap_slides,
        "decision": decision,
        "fit_allowed": False,
        "operator_promotion_allowed": False,
        "capacity_rescue_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, CONFIG_SHA256)
    report = run_diagnostic(config)
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "gap_slides": report["source_support_gap_slides"],
                "per_slide": report["per_slide"],
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
        / "configs/u5_r2aq2d_colorreference_source_support_diagnostic_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256", default=CONFIG_SHA256
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq2d_colorreference_source_support_diagnostic_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
