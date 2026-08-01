from __future__ import annotations

import numpy as np

from src.roll2film.source_logit_canonicalizer import estimate_source_logit_shift, logit_shift


def test_logit_shift_roundtrip_and_boundaries() -> None:
    rng = np.random.default_rng(13)
    values = rng.random((1000, 3))
    values = np.concatenate((values, np.asarray([[0.0, 0.5, 1.0]])), axis=0)
    shift = np.asarray([0.7, -0.4, 0.2])
    transformed = logit_shift(values, shift)
    recovered = logit_shift(transformed, -shift)
    np.testing.assert_allclose(recovered, values, atol=2e-15)
    assert transformed[-1, 0] == 0.0
    assert transformed[-1, 2] == 1.0


def test_source_estimator_uses_only_given_samples_and_is_bounded() -> None:
    samples = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.4, 0.8], [0.3, 0.6, 0.9]])
    shift = estimate_source_logit_shift(samples, maximum_absolute_shift=1.0)
    assert shift.shape == (3,)
    assert np.max(np.abs(shift)) <= 1.0
    canonical = logit_shift(samples, shift)
    assert np.all(canonical > 0.0)
    assert np.all(canonical < 1.0)
