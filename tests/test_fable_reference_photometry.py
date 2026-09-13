import numpy as np
import pytest

from src.eval.fable_reference_photometry import antithetic_draws, canonical_donor, dequantize8, quantize8, transform


def test_transform_identity_endpoints_and_quantized_detail_loss():
    rgb = np.repeat((np.arange(256)/255)[:, None], 3, axis=1)
    kwargs = dict(slope_limit=1.25, offset_limit=.35)
    assert np.array_equal(transform(rgb, np.zeros(4), **kwargs), rgb)
    changed = transform(rgb, np.array([1, -1, -1, -1]), **kwargs)
    assert np.array_equal(changed[[0, -1]], rgb[[0, -1]])
    assert np.all(np.diff(changed, axis=0) > 0)
    assert np.array_equal(quantize8(changed)[0], quantize8(changed)[1])


def test_quantizer_half_rounds_up_and_dequantizer_recovers_codes():
    assert quantize8(np.array([.5/255]))[0] == 1/255
    codes = np.arange(256)/255
    a = dequantize8(codes, seed=91)
    assert np.array_equal(quantize8(a), codes)
    assert np.array_equal(a, dequantize8(codes, seed=91))
    assert not np.array_equal(a, dequantize8(codes, seed=92))
    with pytest.raises(ValueError):
        dequantize8(np.array([.12345]), seed=1)


def test_draws_pair_and_stream_isolation():
    a = antithetic_draws(pairs=16, seed=20260913)
    b = antithetic_draws(pairs=16, seed=20260914)
    assert np.array_equal(a[::2], -a[1::2])
    assert np.array_equal(a, antithetic_draws(pairs=16, seed=20260913))
    assert not np.array_equal(a, b)
    assert np.max(abs(a)) < 1


def test_canonicalization_is_idempotent_on_canonical_128_codes():
    rgb = np.random.default_rng(8).integers(0, 256, (128, 128, 3))/255
    assert np.array_equal(canonical_donor(rgb, side=128), rgb)
