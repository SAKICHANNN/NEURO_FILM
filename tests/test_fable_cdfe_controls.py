import numpy as np
import pytest

from src.eval.fable_cdfe_controls import decode_assessment_controls
from src.eval.fable_canonical_prior import decode_canonical_prior


def test_shuffle_retains_actual_after_measurement_and_matching_treatment():
    rng = np.random.default_rng(3)
    prediction = rng.uniform(-.04, .04, (32, 32, 4))
    mu = rng.uniform(-.04, .04, (32, 32, 3))
    scale = rng.uniform(.97, 1.03, (32, 32))
    ids = [str(j) for j in range(32)]
    mapping = {str(j): str((j+1)%32) for j in range(32)}
    kwargs = dict(fitting_mean=np.zeros(4), donor_ids=ids, cameras=['camera']*32,
                  shuffle_mapping=mapping, slope_limit=1.25, offset_limit=.35)
    result = decode_assessment_controls(prediction, mu, scale, **kwargs)
    expected = decode_canonical_prior({'mu': mu[0, 7], 'scale': scale[0, 7]}, prediction[1, 7],
                                     slope_limit=1.25, offset_limit=.35)['u']
    np.testing.assert_array_equal(result['shuffled'][0, 7], expected)
    assert not np.array_equal(result['shuffled'][0, 7], result['learned'][1, 7])
    with pytest.raises(ValueError, match='within-camera'):
        decode_assessment_controls(prediction, mu, scale, **(kwargs | {'cameras': ['other']+['camera']*31}))
    with pytest.raises(ValueError, match='within-camera'):
        decode_assessment_controls(prediction, mu, scale, **(kwargs | {'shuffle_mapping': dict(zip(ids, ids))}))
