from __future__ import annotations

import numpy as np
import pytest

from src.preprocess.dng_forward_matrix import (
    PCS_XYZ,
    DngForwardMatrixError,
    build_dual_illuminant_camera_to_pcs,
    normalize_forward_matrix,
)


def _kwargs() -> dict[str, object]:
    return {
        "color_matrix1": np.eye(3),
        "color_matrix2": np.eye(3),
        "forward_matrix1": np.eye(3),
        "forward_matrix2": np.eye(3),
        "calibration_illuminant1": 17,
        "calibration_illuminant2": 21,
        "as_shot_neutral": [1.0, 1.0, 1.0],
    }


def test_forward_normalization_maps_camera_one_to_pcs() -> None:
    matrix = normalize_forward_matrix(
        [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.3, 0.7]]
    )
    np.testing.assert_allclose(matrix @ np.ones(3), PCS_XYZ, atol=1e-15, rtol=0)


def test_identity_profile_is_finite_and_deterministic() -> None:
    first = build_dual_illuminant_camera_to_pcs(**_kwargs())
    second = build_dual_illuminant_camera_to_pcs(**_kwargs())
    np.testing.assert_array_equal(first.camera_to_pcs, second.camera_to_pcs)
    np.testing.assert_allclose(
        first.camera_to_pcs @ first.camera_white,
        PCS_XYZ,
        atol=1e-15,
        rtol=0,
    )


@pytest.mark.parametrize("illuminants", [(21, 21), (17, 23), (1, 21)])
def test_unsupported_illuminants_fail_closed(illuminants: tuple[int, int]) -> None:
    kwargs = _kwargs()
    kwargs["calibration_illuminant1"], kwargs["calibration_illuminant2"] = illuminants
    with pytest.raises(DngForwardMatrixError, match="one A and one D65"):
        build_dual_illuminant_camera_to_pcs(**kwargs)


def test_nonpositive_neutral_fails_closed() -> None:
    kwargs = _kwargs()
    kwargs["as_shot_neutral"] = [1.0, 0.0, 1.0]
    with pytest.raises(DngForwardMatrixError, match="must be positive"):
        build_dual_illuminant_camera_to_pcs(**kwargs)
