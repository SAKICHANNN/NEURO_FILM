import numpy as np

from src.eval.filmmatch_strict_interior_ood import _apply_rows, _operator_from_report


def test_bl6_operator_parser_and_partition_exactness() -> None:
    report = {
        "final_fit": {
            "operator": {
                "capture_matrix": np.eye(3).tolist(),
                "response_midpoints": [-3.0, -3.0, -3.0],
                "response_slopes": [1.0, 1.0, 1.0],
                "scan_matrix": np.eye(3).tolist(),
                "nonlinear_strength": 0.75,
                "exposure_floor": 2.0**-16,
                "output_epsilon": 0.5 / 255.0,
            }
        }
    }
    operator = _operator_from_report(report)
    image = np.random.default_rng(61).uniform(size=(17, 11, 3)).astype(np.float32)
    np.testing.assert_array_equal(
        _apply_rows(operator, image, rows=4),
        _apply_rows(operator, image, rows=17),
    )
    code = np.rint(_apply_rows(operator, image) * 65535.0).astype(np.uint16)
    assert np.all(code > 0)
    assert np.all(code < 65535)
