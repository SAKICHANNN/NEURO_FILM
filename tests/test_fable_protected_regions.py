import numpy as np
import pytest

from src.eval.fable_protected_regions import protected_region_counts, protected_region_accuracy


def identity():
    return np.repeat((np.arange(256) / 255)[:, None], 3, axis=1)


def test_small_region_failure_cannot_be_diluted_by_background_or_other_region():
    source = np.full((20, 20, 3), 100, dtype=np.uint8)
    source[0, 0] = 50
    mask = np.zeros((20, 20), dtype=bool)
    mask[0, 0] = True
    oracle, prediction = identity(), identity()
    oracle[[50, 100]] += 10 / 255
    prediction[100] += 10 / 255
    counts = protected_region_counts(source, {'text': mask, 'background': ~mask})
    result = protected_region_accuracy(counts, prediction[None], oracle[None])
    assert result['regions']['background']['passed']
    assert not result['regions']['text']['passed'] and not result['passed']
    whole = protected_region_counts(source, {'all': np.ones_like(mask)})
    assert protected_region_accuracy(whole, prediction[None], oracle[None])['passed']
    crop = protected_region_counts(source[:1, :1], {'text': np.ones((1, 1), bool)})
    assert protected_region_accuracy(crop, prediction[None], oracle[None])['regions']['text'] == result['regions']['text']


def test_zero_energy_and_equal_case_pooling():
    counts = protected_region_counts(np.full((1, 1, 3), 100, np.uint8), {'face': np.ones((1, 1), bool)})
    base = identity()
    assert protected_region_accuracy(counts, base[None], base[None])['passed']
    changed = base.copy()
    changed[100] += 1 / 255
    assert not protected_region_accuracy(counts, changed[None], base[None])['passed']
    stronger = base.copy()
    stronger[100] += 4 / 255
    result = protected_region_accuracy(counts, np.stack([changed, stronger]), np.stack([base, stronger]))
    assert result['regions']['face']['D'] == 8
    assert result['regions']['face']['E'] == .5
    assert result['passed']


def test_masked_histograms_equal_native_rounding_and_endpoints():
    rng = np.random.default_rng(174)
    codes = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
    codes[0, 0] = [0, 255, 128]
    mask = rng.random((9, 11)) > .5
    mask[0, 0] = True
    p = np.clip(identity() + .5 / 255, 0, 1)
    o = np.clip(identity() - 3 / 255, 0, 1)
    result = protected_region_accuracy(protected_region_counts(codes, {'detail': mask}), p[None], o[None])['regions']['detail']
    pp = np.stack([np.floor(255 * p[codes[..., c], c] + .5) for c in range(3)], -1)
    oo = np.stack([np.floor(255 * o[codes[..., c], c] + .5) for c in range(3)], -1)
    assert result['E'] == pytest.approx(np.mean((pp[mask] - oo[mask]) ** 2))
    assert result['D'] == pytest.approx(np.mean((oo[mask] - codes[mask].astype(float)) ** 2))


@pytest.mark.parametrize('mask', [np.zeros((2, 2), bool), np.ones((1, 2), bool), np.ones((2, 2), np.uint8)])
def test_invalid_regions_are_rejected(mask):
    with pytest.raises(ValueError):
        protected_region_counts(np.ones((2, 2, 3), np.uint8), {'region': mask})
