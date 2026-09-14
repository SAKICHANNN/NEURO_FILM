import numpy as np
import pytest

from src.data.fable_cdfe_preprocessing import donor_training_examples, fitting_target_normalizer
from src.eval.fable_canonical_prior import canonical_measure
from src.preprocess.fable_canonical_raw import linear16_to_q8


def test_native_target_stays_q8_and_shared_noise_is_repeatable():
    linear = np.arange(18 * 24 * 3, dtype=np.uint16).reshape(18, 24, 3) * 41
    bounds = dict(epsilon=1 / 510, scale_floor=.001, tensor_limit=8)
    treatments = [{'id': 'a', 'u': [.4, -.6, .7, -.8]}, {'id': 'repeat', 'u': [.4, -.6, .7, -.8]}]
    rows = list(donor_training_examples(linear, treatments, seed=13, measurement=bounds,
                                      input_size=8, slope_limit=1.25, offset_limit=.35))
    expected = canonical_measure(linear16_to_q8(linear).astype(float) / 255, **bounds)['target']
    np.testing.assert_array_equal(rows[0]['target'], expected)
    np.testing.assert_array_equal(rows[0]['input'], rows[1]['input'])
    assert rows[0]['measurement_shape'] == linear.shape
    assert rows[0]['input'].shape == (3, 8, 8)


def test_normalizer_rejects_duplicates_holdouts_and_degeneracy():
    values = np.array([[1, 2, 3, 4], [3, 6, 9, 12]], dtype=float)
    result = fitting_target_normalizer(values, ['a', 'b'], expected_identities=['a', 'b'], minimum_scale=1e-12)
    np.testing.assert_array_equal(result['scale'], [1, 2, 3, 4])
    for ids in [['a', 'a'], ['a', 'held']]:
        with pytest.raises(ValueError):
            fitting_target_normalizer(values, ids, expected_identities=['a', 'b'], minimum_scale=1e-12)
    with pytest.raises(ValueError, match='degenerate'):
        fitting_target_normalizer(np.ones((2, 4)), ['a', 'b'], expected_identities=['a', 'b'], minimum_scale=1e-12)
