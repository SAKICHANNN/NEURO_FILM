#!/usr/bin/env python
"""Run the frozen U5.R2J0 positive-film-response representation audit."""

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

from src.roll2film.positive_film import (  # noqa: E402
    PositiveFilmResponseOperator,
    finite_difference_jacobians,
    positive_film_operator_from_config,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _best_affine_residual(source: np.ndarray, target: np.ndarray) -> float:
    design = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(np.sqrt(np.mean((design @ coefficients - target) ** 2)))


def run_audit(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    grid_size = int(config["audit_grid_size"])
    axis = np.linspace(0.0, 1.0, grid_size)
    flat = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    interior = np.stack(
        np.meshgrid(axis[1:-1], axis[1:-1], axis[1:-1], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    gates = config["gates"]
    witnesses: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, payload in config["witnesses"].items():
        operator = positive_film_operator_from_config(
            payload,
            exposure_floor=float(config["exposure_floor"]),
            matrix_minimum_determinant=float(
                config["parameter_bounds"]["matrix_minimum_determinant"]
            ),
            minimum_endpoint_span=float(
                config["parameter_bounds"]["minimum_endpoint_span"]
            ),
        )
        output = operator.apply(flat)
        outputs[name] = output
        endpoints = operator.apply(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )
        jacobians = finite_difference_jacobians(
            operator, interior, step=float(config["finite_difference_step"])
        )
        replay = PositiveFilmResponseOperator.from_dict(
            json.loads(json.dumps(operator.to_dict(), sort_keys=True))
        )
        split = len(flat) // 3
        partitioned = np.concatenate(
            (
                operator.apply(flat[:split]),
                operator.apply(flat[split : 2 * split]),
                operator.apply(flat[2 * split :]),
            )
        )
        witnesses[name] = {
            "endpoint_maximum_absolute_error": float(
                np.max(
                    np.abs(
                        endpoints
                        - np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
                    )
                )
            ),
            "output_minimum": float(np.min(output)),
            "output_maximum": float(np.max(output)),
            "minimum_finite_difference_jacobian_determinant": float(
                np.min(np.linalg.det(jacobians))
            ),
            "minimum_directional_derivative": float(np.min(jacobians)),
            "serialization_replay_maximum_absolute_error": float(
                np.max(np.abs(replay.apply(flat) - output))
            ),
            "partition_maximum_absolute_error": float(
                np.max(np.abs(partitioned - output))
            ),
            "strength_zero_maximum_absolute_error": float(
                np.max(np.abs(operator.apply(flat, strength=0.0) - flat))
            ),
            "strength_one_maximum_absolute_error": float(
                np.max(np.abs(operator.apply(flat, strength=1.0) - output))
            ),
            "identity_rgb_rmse": float(np.sqrt(np.mean((output - flat) ** 2))),
            "best_affine_rgb_residual_rmse": _best_affine_residual(flat, output),
            "endpoint_span": operator.endpoint_span.tolist(),
        }

    names = list(outputs)
    pairwise = {
        f"{left}__{right}": float(
            np.sqrt(np.mean((outputs[left] - outputs[right]) ** 2))
        )
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
    }
    non_neutral = [
        values["identity_rgb_rmse"]
        for name, values in witnesses.items()
        if name != "neutral_positive_reference"
    ]
    checks = {
        "endpoints": all(
            values["endpoint_maximum_absolute_error"]
            <= gates["endpoint_maximum_absolute_error"]
            for values in witnesses.values()
        ),
        "bounded": all(
            values["output_minimum"] >= gates["output_minimum"]
            and values["output_maximum"] <= gates["output_maximum"]
            for values in witnesses.values()
        ),
        "positive_jacobian": all(
            values["minimum_finite_difference_jacobian_determinant"]
            > gates["minimum_finite_difference_jacobian_determinant"]
            for values in witnesses.values()
        ),
        "non_negative_directional_derivatives": all(
            values["minimum_directional_derivative"]
            >= gates["minimum_directional_derivative"]
            for values in witnesses.values()
        ),
        "serialization_replay": all(
            values["serialization_replay_maximum_absolute_error"]
            <= gates["serialization_replay_maximum_absolute_error"]
            for values in witnesses.values()
        ),
        "partition_parity": all(
            values["partition_maximum_absolute_error"] == 0.0
            for values in witnesses.values()
        ),
        "strength_endpoints": all(
            values["strength_zero_maximum_absolute_error"] == 0.0
            and values["strength_one_maximum_absolute_error"] == 0.0
            for values in witnesses.values()
        ),
        "non_neutral_identity_distance": (
            min(non_neutral) >= gates["minimum_non_neutral_identity_rgb_rmse"]
        ),
        "pairwise_diversity": (
            min(pairwise.values()) >= gates["minimum_pairwise_witness_rgb_rmse"]
        ),
        "non_affine_residual": all(
            values["best_affine_rgb_residual_rmse"]
            >= gates["minimum_residual_after_best_affine_rgb_fit"]
            for values in witnesses.values()
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": config_sha256,
        "grid_size": grid_size,
        "grid_points": len(flat),
        "witnesses": witnesses,
        "pairwise_uniform_rgb_rmse": pairwise,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2j0_positive_film_response_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2j0_positive_film_response_v1/audit.json",
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    report = run_audit(json.loads(config_bytes), _sha256(config_bytes))
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(encoded),
                "all_checks_passed": report["all_checks_passed"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
