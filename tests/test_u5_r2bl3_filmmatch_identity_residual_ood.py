import numpy as np

from src.eval.filmmatch_identity_residual_ood import (
    _apply_bl1,
    _encode_scene_linear,
)
from src.roll2film.identity_residual_sigmoid import IdentityResidualSigmoidOperator


def test_bl3_row_execution_is_partition_exact() -> None:
    rng = np.random.default_rng(31)
    scene = rng.uniform(0.0, 1.0, size=(17, 11, 3)).astype(np.float32)
    encoded = _encode_scene_linear(scene, rows=4)
    whole = _encode_scene_linear(scene, rows=17)
    assert np.array_equal(encoded, whole)

    operator = IdentityResidualSigmoidOperator(
        capture_matrix=np.eye(3),
        response_midpoints=np.full(3, -3.0),
        response_slopes=np.ones(3),
        scan_matrix=np.eye(3),
    )
    assert np.array_equal(
        _apply_bl1(operator, encoded, rows=4),
        _apply_bl1(operator, encoded, rows=17),
    )
