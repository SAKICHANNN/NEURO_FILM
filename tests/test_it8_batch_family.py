from __future__ import annotations

import numpy as np

from src.real_film.it8_batch_family import (
    CappedVariancePCA,
    exact_stratified_fixed_score_p,
    residual_features,
)
from src.real_film.it8_batch_family import ChargeMeasurement


def _charge(name: str, family: str, offset: float) -> ChargeMeasurement:
    return ChargeMeasurement(
        name=name,
        family=family,
        year=2000 + int(name[1:3]),
        sample_ids=("A1", "A2"),
        lab=np.asarray([[50.0 + offset, 1.0, 2.0], [60.0, 3.0, 4.0]]),
        spectra_pct=np.asarray([[1.0 + offset, 2.0], [3.0, 4.0]]),
        mean_de_median=0.5,
        mean_de_p90=0.7,
    )


def test_residual_features_fit_pooled_aim_on_development_only() -> None:
    charges = [
        _charge("E040227.zip", "E", 0.0),
        _charge("V040102.zip", "V", 2.0),
        _charge("E080220.zip", "E", 100.0),
    ]
    features = residual_features(
        charges, np.asarray([0, 1]), view="lab"
    )
    assert features.shape == (3, 6)
    assert features[0, 0] == -1.0
    assert features[1, 0] == 1.0
    assert features[2, 0] == 99.0


def test_exact_fixed_score_p_enumerates_stratified_assignments() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    truth = np.asarray([0, 0, 1, 1])
    p_value, assignments = exact_stratified_fixed_score_p(scores, truth)
    assert assignments == 6
    assert p_value == 1 / 6


def test_capped_variance_pca_never_exceeds_frozen_rank() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(40, 80))
    fitted = CappedVariancePCA(maximum_components=7).fit(values)
    assert fitted.n_components_ == 7
    assert fitted.transform(values).shape == (40, 7)
