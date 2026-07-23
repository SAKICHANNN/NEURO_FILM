from __future__ import annotations

import numpy as np

from src.eval.projected_lut_frontier import (
    convert_public_lut,
    identity_lut,
    lut_diagnostics,
    project_lut,
    structure_safe,
)


PROJECTION = {
    "node_minimum": 0.0,
    "node_maximum": 1.0,
    "minimum_corresponding_channel_grid_step": 1e-7,
    "minimum_tetrahedron_jacobian_determinant": 1e-8,
    "alpha_prefix_grid_samples": 33,
    "alpha_bisection_iterations": 24,
}


def _public_identity(size: int = 5) -> np.ndarray:
    return np.transpose(identity_lut(size), (3, 2, 1, 0))


def test_public_layout_roundtrip_identity() -> None:
    public = _public_identity()
    np.testing.assert_allclose(convert_public_lut(public), identity_lut(5))


def test_identity_is_structure_safe() -> None:
    diagnostics = lut_diagnostics(identity_lut(5))
    assert structure_safe(diagnostics, PROJECTION)
    assert diagnostics["minimum_tetrahedron_jacobian_determinant"] > 0.0


def test_clip_control_stays_bounded_but_can_be_structurally_unsafe() -> None:
    public = _public_identity()
    public[0, :, :, 2:] = -1.0
    values, audit = project_lut(
        public,
        {
            "identity_contraction_cap": 1.0,
            "enforce_structure_prefix": False,
        },
        PROJECTION,
    )
    assert values.min() >= 0.0 and values.max() <= 1.0
    assert not audit["structure_safe"]


def test_structure_prefix_contracts_unsafe_lut() -> None:
    public = _public_identity()
    public[0, :, :, 2:] = -1.0
    values, audit = project_lut(
        public,
        {
            "identity_contraction_cap": 1.0,
            "enforce_structure_prefix": True,
        },
        PROJECTION,
    )
    assert 0.0 <= audit["applied_alpha"] < 1.0
    assert audit["structure_safe"]
    assert structure_safe(lut_diagnostics(values), PROJECTION)


def test_safe_cap_is_respected() -> None:
    _, audit = project_lut(
        _public_identity(),
        {
            "identity_contraction_cap": 0.75,
            "enforce_structure_prefix": True,
        },
        PROJECTION,
    )
    assert audit["applied_alpha"] == 0.75
