#!/usr/bin/env python
"""Run the frozen U5.R2X0 analytic palette-transfer audit."""

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

from src.roll2film.palette_transfer_affine import (  # noqa: E402
    apply_palette_transfer_direct,
    collapse_palette_transfer_affine,
    generalized_affine_barycentric_weights,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _software_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _standard_tetrahedron() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _cube_corners() -> np.ndarray:
    return np.stack(
        np.meshgrid(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)


def run_audit(
    config: dict[str, Any], *, config_sha256: str, software_commit: str
) -> dict[str, Any]:
    if config.get("status") != "implemented_frozen_pending_formal_run":
        raise ValueError("unexpected U5.R2X0 contract status")
    equivalence = config["equivalence"]
    rng = np.random.default_rng(int(equivalence["seed"]))
    count = int(equivalence["source_palette_count"])
    if count != int(equivalence["target_palette_count"]):
        raise ValueError("frozen equivalence witness requires equal palette sizes")
    source = rng.uniform(0.05, 0.95, size=(count, 3))
    target = rng.uniform(0.05, 0.95, size=(count, 3))
    queries = rng.uniform(
        0.0, 1.0, size=(int(equivalence["query_count"]), 3)
    )
    alpha = float(equivalence["transport_identity_weight"])
    identity = np.eye(count, dtype=np.float64)
    cyclic = np.roll(identity, 1, axis=1)
    transport = alpha * identity + (1.0 - alpha) * cyclic
    operator = collapse_palette_transfer_affine(source, target, transport)
    direct = apply_palette_transfer_direct(source, target, transport, queries)
    collapsed = operator.apply(queries)
    weights = generalized_affine_barycentric_weights(source, queries)
    source_weights = generalized_affine_barycentric_weights(source, source)
    source_reconstruction = source_weights @ source

    tetrahedron = _standard_tetrahedron()
    red_blue_permutation = np.eye(4, dtype=np.float64)[[0, 3, 2, 1]]
    orientation_operator = collapse_palette_transfer_affine(
        tetrahedron, tetrahedron, red_blue_permutation
    )
    orientation_determinant = float(
        np.linalg.det(orientation_operator.matrix)
    )

    compact_source = 0.4 + 0.2 * tetrahedron
    range_operator = collapse_palette_transfer_affine(
        compact_source, tetrahedron, np.eye(4, dtype=np.float64)
    )
    cube_output = range_operator.apply(_cube_corners())
    gates = config["gates"]
    metrics = {
        "direct_vs_affine_maximum_absolute_error": float(
            np.max(np.abs(direct - collapsed))
        ),
        "weight_sum_maximum_absolute_error": float(
            np.max(np.abs(np.sum(weights, axis=1) - 1.0))
        ),
        "source_colour_reconstruction_maximum_absolute_error": float(
            np.max(np.abs(source_reconstruction - source))
        ),
        "affine_matrix_determinant": float(np.linalg.det(operator.matrix)),
    }
    gate_results = {
        "direct_vs_affine_equivalence": metrics[
            "direct_vs_affine_maximum_absolute_error"
        ]
        <= float(gates["maximum_direct_vs_affine_error"]),
        "weight_partition": metrics["weight_sum_maximum_absolute_error"]
        <= float(gates["maximum_weight_sum_error"]),
        "source_colour_reconstruction": metrics[
            "source_colour_reconstruction_maximum_absolute_error"
        ]
        <= float(gates["maximum_source_colour_reconstruction_error"]),
        "orientation_witness_fails_positive_orientation": (
            orientation_determinant <= 0.0
        ),
        "range_witness_exits_cube": (
            float(np.min(cube_output))
            < float(gates["cube_safe_requires_output_minimum_at_least"])
            and float(np.max(cube_output))
            > float(gates["cube_safe_requires_output_maximum_at_most"])
        ),
    }
    expected = config["expected_witnesses"]
    witness_matches = {
        "orientation_determinant": bool(
            np.isclose(
                orientation_determinant,
                float(expected["orientation"]["determinant"]),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "range_matrix": bool(
            np.allclose(
                range_operator.matrix,
                5.0 * np.eye(3),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "range_bias": bool(
            np.allclose(
                range_operator.bias,
                np.full(3, -2.0),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "range_extrema": bool(
            np.isclose(np.min(cube_output), -2.0, atol=1e-12, rtol=0.0)
            and np.isclose(
                np.max(cube_output), 3.0, atol=1e-12, rtol=0.0
            )
        ),
    }
    all_valid = all(gate_results.values()) and all(witness_matches.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "status": "formal_run_pending_exact_repeat",
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "source": config["source"],
        "scope": config["scope"],
        "equivalence": {
            "source_palette_count": count,
            "target_palette_count": count,
            "query_count": len(queries),
            "transport_row_sum_error": float(
                np.max(np.abs(np.sum(transport, axis=1) - 1.0))
            ),
            "transport_column_sum_error": float(
                np.max(np.abs(np.sum(transport, axis=0) - 1.0))
            ),
            "metrics": metrics,
        },
        "orientation_witness": {
            "transport_row_sums": np.sum(
                red_blue_permutation, axis=1
            ).tolist(),
            "transport_column_sums": np.sum(
                red_blue_permutation, axis=0
            ).tolist(),
            "matrix": orientation_operator.matrix.tolist(),
            "bias": orientation_operator.bias.tolist(),
            "determinant": orientation_determinant,
        },
        "range_witness": {
            "matrix": range_operator.matrix.tolist(),
            "bias": range_operator.bias.tolist(),
            "determinant": float(np.linalg.det(range_operator.matrix)),
            "cube_output_minimum": float(np.min(cube_output)),
            "cube_output_maximum": float(np.max(cube_output)),
        },
        "gate_results": gate_results,
        "expected_witnesses_match": witness_matches,
        "decision_branch_before_repeat": (
            "analytic_affine_only_close"
            if all_valid
            else "published_equation_not_reproduced"
        ),
        "images_accessed": 0,
        "external_code_or_cplex_executed": False,
        "automatic_visual_shortlist_generated": False,
        "current_stock_pixels_accessed": False,
        "current_stock_training_or_operator_fitting_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2x0_palette_transfer_affine_collapse_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "eval"
        / "u5_r2x0_palette_transfer_affine_audit_report.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = run_audit(
        config,
        config_sha256=_sha256_file(config_path),
        software_commit=_software_commit(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(_sha256_file(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
