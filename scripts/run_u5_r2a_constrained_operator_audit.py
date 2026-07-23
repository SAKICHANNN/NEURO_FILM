"""Run the frozen U5.R2A data-independent operator audit."""

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

from src.roll2film.constrained import (  # noqa: E402
    ConstrainedGlobalColorOperator,
    LUTConstraintSpec,
    audit_lut_constraints,
    identity_lut,
)
from src.roll2film.lut import DenseLUT3D, bake_dense_lut  # noqa: E402
from src.roll2film.operators import AffineColorOperator  # noqa: E402
from src.roll2film.splines import (  # noqa: E402
    AffineMonotoneSplineOperator,
    RationalQuadraticSpline,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _software_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _matrix_lut(matrix: np.ndarray, size: int) -> DenseLUT3D:
    return bake_dense_lut(
        AffineColorOperator(matrix, np.zeros(3)),
        size,
        interpolation="tetrahedral",
    )


def _scalar_tetrahedral(lut: DenseLUT3D, rgb: np.ndarray) -> np.ndarray:
    result = []
    for point in np.asarray(rgb, dtype=np.float64):
        coordinate = (point - lut.domain_min) / (lut.domain_max - lut.domain_min)
        coordinate *= lut.size - 1
        lower = np.minimum(np.floor(coordinate).astype(int), lut.size - 2)
        red, green, blue = coordinate - lower
        r, g, b = lower
        c000 = lut.values[r, g, b]
        c100 = lut.values[r + 1, g, b]
        c010 = lut.values[r, g + 1, b]
        c001 = lut.values[r, g, b + 1]
        c110 = lut.values[r + 1, g + 1, b]
        c101 = lut.values[r + 1, g, b + 1]
        c011 = lut.values[r, g + 1, b + 1]
        c111 = lut.values[r + 1, g + 1, b + 1]
        if red >= green >= blue:
            value = c000 + red * (c100 - c000) + green * (c110 - c100) + blue * (c111 - c110)
        elif red >= blue > green:
            value = c000 + red * (c100 - c000) + blue * (c101 - c100) + green * (c111 - c101)
        elif blue > red >= green:
            value = c000 + blue * (c001 - c000) + red * (c101 - c001) + green * (c111 - c101)
        elif green > red >= blue:
            value = c000 + green * (c010 - c000) + red * (c110 - c010) + blue * (c111 - c110)
        elif green >= blue > red:
            value = c000 + green * (c010 - c000) + blue * (c011 - c010) + red * (c111 - c011)
        else:
            value = c000 + blue * (c001 - c000) + green * (c011 - c001) + red * (c111 - c011)
        result.append(value)
    return np.asarray(result)


def run_audit(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lut_config = config["lut"]
    gates = config["property_gates"]
    spec = LUTConstraintSpec(
        output_minimum=lut_config["output_minimum"],
        output_maximum=lut_config["output_maximum"],
        maximum_residual_amplitude=lut_config["maximum_residual_amplitude"],
        maximum_first_axis_step=lut_config["maximum_first_axis_step"],
        maximum_second_axis_difference=lut_config["maximum_second_axis_difference"],
        maximum_neutral_axis_error=lut_config["maximum_neutral_axis_error"],
        minimum_tetrahedron_jacobian_determinant=lut_config[
            "minimum_tetrahedron_jacobian_determinant"
        ],
    )
    identity = ConstrainedGlobalColorOperator.identity(
        lut_config["audit_size"],
        spec,
        config["working_space"],
    )
    golden = np.asarray(config["golden_vectors"], dtype=np.float64)
    golden_output = identity.apply(golden)

    curve_x = np.asarray([-0.25, 0.0, 0.2, 0.5, 0.8, 1.0, 1.25])
    curve_y = np.asarray([-0.2, 0.0, 0.16, 0.53, 0.83, 1.0, 1.2])
    curve = RationalQuadraticSpline.from_knots(curve_x, curve_y)
    curve_probes = np.linspace(-0.5, 1.5, 4097)
    curve_roundtrip = float(
        np.max(np.abs(curve.inverse(curve.apply(curve_probes)) - curve_probes))
    )
    curve_minimum_derivative = float(np.min(curve.derivative(curve_probes)))

    rng = np.random.default_rng(config["seed"])
    nonlinear_values = identity_lut(7).values.copy()
    nonlinear_values[..., 0] += 0.01 * nonlinear_values[..., 1] * (
        1.0 - nonlinear_values[..., 2]
    )
    nonlinear = DenseLUT3D(
        nonlinear_values,
        np.zeros(3),
        np.ones(3),
        "tetrahedral",
    )
    probes = rng.uniform(0.0, 1.0, size=(2048, 3))
    tetrahedral_reference_error = float(
        np.max(np.abs(nonlinear.apply(probes) - _scalar_tetrahedral(nonlinear, probes)))
    )

    replay = ConstrainedGlobalColorOperator.from_dict(
        json.loads(json.dumps(identity.to_dict(), sort_keys=True))
    )
    replay_error = float(np.max(np.abs(replay.apply(golden) - golden_output)))

    counter = config["non_safety_counterexample"]
    counter_lut = _matrix_lut(np.asarray(counter["matrix"], dtype=np.float64), lut_config["audit_size"])
    counter_report = audit_lut_constraints(counter_lut, spec)
    counter_operator = ConstrainedGlobalColorOperator(
        AffineMonotoneSplineOperator.identity(config["working_space"]),
        counter_lut,
        spec,
    )
    blue_probe = np.asarray([counter["blue_probe"]], dtype=np.float64)
    blue_output = counter_operator.apply(blue_probe)
    red_increase = float(blue_output[0, 0] - blue_probe[0, 0])

    source = probes.copy()
    source_before = source.copy()
    nonlinear.apply(source)
    source_nonmutation = bool(np.array_equal(source, source_before))
    invalid_input_fails_closed = False
    try:
        identity.apply(np.asarray([[np.nan, 0.5, 0.5]]))
    except ValueError:
        invalid_input_fails_closed = True

    identity_error = float(np.max(np.abs(golden_output - golden)))
    report = {
        "schema_version": 1,
        "node": "U5.R2A",
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "software_commit": _software_commit(),
        "seed": config["seed"],
        "identity": {
            "maximum_absolute_error": identity_error,
            "golden_inputs": golden.tolist(),
            "golden_outputs": golden_output.tolist(),
            "constraint_report": audit_lut_constraints(identity.lut, spec).to_dict(),
        },
        "curve": {
            "roundtrip_maximum_absolute_error": curve_roundtrip,
            "minimum_derivative": curve_minimum_derivative,
        },
        "tetrahedral": {
            "scalar_reference_maximum_absolute_error": tetrahedral_reference_error,
        },
        "serialization": {
            "replay_maximum_absolute_error": replay_error,
        },
        "safety_boundaries": {
            "source_nonmutation": source_nonmutation,
            "invalid_input_fails_closed": invalid_input_fails_closed,
        },
        "non_safety_counterexample": {
            "input": blue_probe[0].tolist(),
            "output": blue_output[0].tolist(),
            "red_increase": red_increase,
            "numerical_contract": counter_report.to_dict(),
            "semantic_safety_claim": False,
        },
    }
    report["automatic_gate"] = {
        "identity_pass": identity_error <= gates["identity_maximum_absolute_error"],
        "curve_roundtrip_pass": curve_roundtrip <= gates["curve_roundtrip_maximum_absolute_error"],
        "curve_derivative_pass": curve_minimum_derivative > gates["curve_minimum_derivative"],
        "tetrahedral_reference_pass": (
            tetrahedral_reference_error
            <= gates["tetrahedral_scalar_reference_maximum_absolute_error"]
        ),
        "serialization_pass": (
            replay_error <= gates["serialization_replay_maximum_absolute_error"]
        ),
        "source_nonmutation_pass": source_nonmutation is gates["source_nonmutation"],
        "invalid_input_pass": invalid_input_fails_closed is gates["invalid_input_fails_closed"],
        "counterexample_numerical_pass": counter_report.passes,
        "counterexample_semantic_shift_pass": (
            red_increase >= counter["minimum_red_increase"]
        ),
    }
    report["decision"] = (
        "pass_numerical_representation_not_semantic_safety"
        if all(report["automatic_gate"].values())
        else "fail"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "u5_r2a_constrained_global_operator_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "u5_r2a_constrained_global_operator_v1" / "report.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    report = run_audit(config_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": report["decision"], "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
