from __future__ import annotations

import numpy as np

from src.eval.constrained_explicit_distillation import (
    BoundedCurveMatrixOperator,
    synthetic_grid,
)
from src.eval.quantization_headroom import (
    QuantizedHeadroomOperator,
    operator_diagnostics,
)


def _config() -> dict:
    return {
        "structure_gates": {
            "minimum_raw_output": 1.0 / 255.0,
            "maximum_raw_output": 254.0 / 255.0,
            "minimum_corresponding_channel_grid_step": 1e-7,
            "minimum_tetrahedron_jacobian_determinant": 1e-8,
            "maximum_replay_absolute_error": 1e-12,
        }
    }


def test_headroom_maps_identity_to_interior_codes() -> None:
    curves = np.tile(np.linspace(0.0, 1.0, 9), (3, 1))
    base = BoundedCurveMatrixOperator(curves, np.eye(3))
    operator = QuantizedHeadroomOperator(base, 1.0 / 255.0, 254.0 / 255.0)
    output = operator.apply(synthetic_grid(17))
    codes = np.rint(output * 255.0).astype(np.uint8)
    assert int(np.min(codes)) == 1
    assert int(np.max(codes)) == 254
    assert np.min(output) == 1.0 / 255.0
    assert np.max(output) == 254.0 / 255.0


def test_headroom_serialization_and_structure_are_exact() -> None:
    curves = np.tile(np.linspace(0.0, 1.0, 9), (3, 1))
    base = BoundedCurveMatrixOperator(curves, np.eye(3))
    operator = QuantizedHeadroomOperator(base, 1.0 / 255.0, 254.0 / 255.0)
    replay = QuantizedHeadroomOperator.from_dict(operator.to_dict())
    grid = synthetic_grid(17)
    assert np.array_equal(operator.apply(grid), replay.apply(grid))
    assert operator_diagnostics(operator, _config())["structure_safe"]
