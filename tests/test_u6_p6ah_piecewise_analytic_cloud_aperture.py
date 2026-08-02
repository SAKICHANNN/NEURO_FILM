from pathlib import Path

import numpy as np
import pytest

from src.eval.analytic_cloud_aperture import (
    evaluate_analytic_cloud_aperture,
    load_contract,
)
from src.film_physics.analytic_cloud_aperture import (
    AnalyticCloudApertureError,
    integrate_marked_cloud_aperture_analytic,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6ah_piecewise_analytic_cloud_aperture_v1.json"
PARENT = ROOT / "configs/u6_p6ag_continuous_cloud_aperture_integration_decision_v1.json"


def _integrate(centers, radii, axis="x", mark=0.2):
    return integrate_marked_cloud_aperture_analytic(
        np.asarray(centers, dtype=np.float64).reshape(-1, 2),
        radii,
        mark_optical_density=mark,
        aperture_center_yx=(0.0, 0.0),
        aperture_radius=1.0,
        integration_axis=axis,
    )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_analytic_circle_controls(axis):
    transmission = 10.0**-0.2
    assert _integrate([], np.empty(0), axis) == pytest.approx(1.0, abs=1e-12)
    assert _integrate([[4.0, 4.0]], 0.5, axis) == pytest.approx(1.0, abs=1e-12)
    assert _integrate([[0.0, 0.0]], 2.0, axis) == pytest.approx(transmission, abs=1e-12)
    assert _integrate([[0.0, 0.0]], 0.5, axis) == pytest.approx(
        0.25 * transmission + 0.75, abs=1e-12
    )
    nested = (
        0.35**2 * transmission**2 + (0.75**2 - 0.35**2) * transmission + (1.0 - 0.75**2)
    )
    assert _integrate([[0.0, 0.0], [0.0, 0.0]], [0.35, 0.75], axis) == pytest.approx(
        nested, abs=1e-12
    )


def test_invalid_request_is_rejected():
    with pytest.raises(AnalyticCloudApertureError):
        _integrate([[0.0, 0.0]], -1.0)


def test_contract_loads():
    assert load_contract(CONTRACT)["integration"][
        "independent_confirmation"
    ].startswith("swap physical")


@pytest.mark.skipif(not PARENT.is_file(), reason="P6AG decision unavailable")
def test_frozen_evaluator_executes_with_internal_replay():
    report = evaluate_analytic_cloud_aperture(load_contract(CONTRACT), ROOT)
    assert report["metrics"]["maximum_repeat_error"] == 0
    assert report["metrics"]["maximum_partition_error"] == 0
