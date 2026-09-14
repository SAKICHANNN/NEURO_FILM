import numpy as np
import pytest

from src.eval.fable_canonical_prior import canonical_measure, decode_canonical_prior
from src.eval.fable_reference_photometry import transform, quantize8


def test_continuous_equivariance_and_recovery():
    image = np.random.default_rng(13).uniform(.02, .98, (19, 17, 3))
    u = np.array([.7, -.8, .4, -.3])
    after = transform(image, u, slope_limit=1.25, offset_limit=.35)
    before_m = canonical_measure(image, epsilon=0, scale_floor=1e-3, tensor_limit=8)
    after_m = canonical_measure(after, epsilon=0, scale_floor=1e-3, tensor_limit=8)
    np.testing.assert_allclose(before_m['tensor'], after_m['tensor'], atol=2e-14)
    result = decode_canonical_prior(after_m, before_m['target'], slope_limit=1.25, offset_limit=.35)
    np.testing.assert_allclose(result['raw'], u, atol=2e-14)


def test_quantized_equivariance_is_not_assumed():
    x = quantize8(np.random.default_rng(24).uniform(0, 1, (16, 21, 3)))
    a = quantize8(transform(x, np.array([1., 1., -1., .5]), slope_limit=1.25, offset_limit=.35))
    before = canonical_measure(x, epsilon=1/510, scale_floor=1e-3, tensor_limit=8)
    after = canonical_measure(a, epsilon=1/510, scale_floor=1e-3, tensor_limit=8)
    assert not np.allclose(before['tensor'], after['tensor'], atol=1e-6)
    assert np.isfinite(after['tensor']).all()


def test_degenerate_scale_and_endpoint_reporting():
    x = np.ones((7, 5, 3)) * np.array([0., .5, 1.])
    m = canonical_measure(x, epsilon=1/510, scale_floor=1e-3, tensor_limit=8)
    assert m['scale_floored'] and m['scale'] == 1e-3
    assert m['clamp_fraction'] == 2/3
    np.testing.assert_allclose(m['tensor'], 0, atol=1e-10)
    with pytest.raises(ValueError, match='interior'):
        canonical_measure(x, epsilon=0, scale_floor=1e-3, tensor_limit=8)


def test_tensor_limit_and_decoding_bounds():
    x = np.full((100, 100, 3), .5)
    x[0, 0] = .99
    m = canonical_measure(x, epsilon=1/510, scale_floor=1e-3, tensor_limit=8)
    assert m['tensor_clipping_fraction'] > 0
    assert np.max(np.abs(m['tensor'])) == 8
    p = decode_canonical_prior(m, np.array([4., -4., 0., -6.]), slope_limit=1.25, offset_limit=.35)
    assert p['clipped'].any() and np.max(np.abs(p['u'])) <= 1


def test_half_code_boundaries_and_endpoints():
    x = np.broadcast_to(np.array([0., .5/255, 1.]), (2, 2, 3))
    q = quantize8(x)
    np.testing.assert_array_equal(q[0, 0], [0., 1/255, 1.])
    m = canonical_measure(q, epsilon=1/510, scale_floor=1e-3, tensor_limit=8)
    assert m['clamp_fraction'] == 2/3
