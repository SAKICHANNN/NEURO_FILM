from __future__ import annotations

import numpy as np

from src.eval.flickr_weak_pair_operator_development import _fit_basic, _registered_samples


def _config() -> dict:
    return {
        "paired_samples": {
            "mask_erosion_pixels": 2,
            "encoded_boundary_code_minimum": 2,
            "encoded_boundary_code_maximum": 253,
            "maximum_combined_encoded_luma_gradient_quantile": 0.75,
            "maximum_samples_per_scene": 512,
        },
        "operators": {
            "family_basic_logit": {
                "log_scale_bounds": [-0.7, 0.7],
                "shift_bounds": [-1.0, 1.0],
                "identity_shrinkage": 0.05,
                "maximum_fit_evaluations": 100,
            }
        },
    }


def test_registered_samples_are_deterministic_and_bounded() -> None:
    y, x = np.mgrid[0:128, 0:160]
    digital = np.stack(((x * 3 + y) % 240 + 8, (x + y * 2) % 240 + 8, (x * 2 + y * 3) % 240 + 8), axis=-1).astype(np.uint8)
    film = digital.copy()
    first = _registered_samples(digital, film, np.eye(3), _config()["paired_samples"])
    second = _registered_samples(digital, film, np.eye(3), _config()["paired_samples"])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[0].shape == (512, 3)
    assert np.all((first[0] > 0) & (first[0] < 1))


def test_basic_logit_recovers_independent_transport() -> None:
    rng = np.random.default_rng(8)
    source = rng.uniform(0.02, 0.98, size=(4000, 3))
    parameters = np.zeros(14)
    parameters[[0, 1, 2, 4, 6, 10]] = [0.1, -0.2, -0.15, 0.1, 0.05, 0.25]
    from src.roll2film.triangular_logit_transport import TriangularLogitTransport

    target = TriangularLogitTransport(parameters).apply(source)
    fitted, converged = _fit_basic(source, target, _config())
    assert converged
    np.testing.assert_allclose(fitted.parameters, parameters, atol=2e-4)
    assert np.sqrt(np.mean(np.square(fitted.apply(source) - target))) < 1e-5
