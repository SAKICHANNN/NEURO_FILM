import numpy as np

from src.eval.filmmatch_identity_residual_validation import _operator_from_report
from src.roll2film.identity_residual_sigmoid import IdentityResidualSigmoidOperator


def test_operator_round_trips_from_bl1_report_payload() -> None:
    payload = {
        "stable_evidence_id": "0" * 64,
        "final_fit": {
            "operator": {
                "capture_matrix": np.eye(3).tolist(),
                "response_midpoints": [-3.0, -3.0, -3.0],
                "response_slopes": [1.0, 1.0, 1.0],
                "scan_matrix": np.eye(3).tolist(),
                "nonlinear_strength": 0.75,
                "exposure_floor": 2.0**-16,
            }
        },
    }
    operator = _operator_from_report(payload)
    rgb = np.asarray([[0.0, 0.25, 1.0]], dtype=np.float64)
    expected = IdentityResidualSigmoidOperator(
        capture_matrix=np.eye(3),
        response_midpoints=np.full(3, -3.0),
        response_slopes=np.ones(3),
        scan_matrix=np.eye(3),
    ).apply(rgb)
    assert np.array_equal(operator.apply(rgb), expected)


def test_operator_parser_rejects_unbounded_matrix() -> None:
    payload = {
        "final_fit": {
            "operator": {
                "capture_matrix": [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "response_midpoints": [-3.0, -3.0, -3.0],
                "response_slopes": [1.0, 1.0, 1.0],
                "scan_matrix": np.eye(3).tolist(),
                "nonlinear_strength": 0.75,
                "exposure_floor": 2.0**-16,
            }
        }
    }
    try:
        _operator_from_report(payload)
    except ValueError as error:
        assert "row-stochastic" in str(error)
    else:
        raise AssertionError("unbounded matrix was accepted")
