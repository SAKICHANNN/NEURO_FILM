import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from skimage.color import deltaE_ciede2000, rgb2lab

from src.eval import tst_operator_visibility as core


def grid():
    axis = np.linspace(0, 1, 7)
    identity = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1)
    result = identity.copy()
    result[..., 0] = .08+1.05*identity[..., 0]**.85
    result[..., 1] = -.04+.91*identity[..., 1]+.03*identity[..., 2]
    result[..., 2] = .13+.9*identity[..., 2]**1.1
    return result.reshape(343, 3)


@pytest.mark.parametrize('shape', [(1, 1, 3), (7, 11, 3), (3, 10923, 3), (257, 257, 3)])
def test_dense_independent_native_reference_and_chunk_boundary(shape):
    source = np.random.default_rng(48).integers(0, 65536, shape, dtype=np.uint16)
    values = grid()
    actual = core.mean_native_delta_e(source, values)
    axis = np.linspace(0, 1, 7)
    rgb = source.astype(np.float64)/65535
    prediction = RegularGridInterpolator((axis, axis, axis), values.reshape(7, 7, 7, 3))(rgb)
    expected = deltaE_ciede2000(rgb2lab(rgb), rgb2lab(np.clip(prediction, 0, 1)))
    np.testing.assert_allclose(actual['mean_delta_e2000'], np.mean(expected, dtype=np.float64), rtol=1e-12, atol=1e-12)
    assert actual['pixel_count'] == shape[0]*shape[1]
    assert actual['chunks'] == (actual['pixel_count']+32767)//32768


def test_identity_and_no_uint16_prediction_quantization():
    axis = np.linspace(0, 1, 7)
    identity = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(343, 3)
    source = np.full((1, 1, 3), 32123, dtype=np.uint16)
    assert core.mean_native_delta_e(source, identity)['mean_delta_e2000'] < 1e-12
    tiny = identity+1e-7
    mean = core.mean_native_delta_e(source, tiny)['mean_delta_e2000']
    assert mean > 0
    rgb = source.astype(np.float64)/65535
    rounded = np.rint(np.clip(rgb+1e-7, 0, 1)*65535)/65535
    assert np.array_equal(rounded, rgb)


def test_exact_four_threshold_without_epsilon():
    assert core.classify_mean(4.0)
    assert core.classify_mean(np.nextafter(4.0, np.inf))
    assert not core.classify_mean(np.nextafter(4.0, -np.inf))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, -1.0])
def test_nonfinite_or_invalid_metric_rejected(bad):
    with pytest.raises(ValueError):
        core.classify_mean(bad)


def probes(split='fit'):
    return [core.VisibilityProbe(str(i), split, np.zeros((1, 1, 3), dtype=np.uint16)) for i in range(4)]


@pytest.mark.parametrize('means,expected_calls,possible', [([3., 3., 8., 9.], 2, False),
    ([4., np.nextafter(4., -np.inf), 4., 3.], 4, False), ([4., 4., 4., 0.], 4, True),
    ([3., 4., 4., 4.], 4, True), ([4., 3., 3., 8.], 3, False)])
def test_necessary_short_circuit_only_after_second_computed_failure(monkeypatch, means, expected_calls, possible):
    called = []

    def measure(_source, _grid):
        value = means[len(called)]
        called.append(value)
        return {'mean_delta_e2000': value, 'pixel_count': 1}

    monkeypatch.setattr(core, 'mean_native_delta_e', measure)
    result = core.screen_visibility(probes(), grid(), candidate_split='fit')
    assert len(called) == expected_calls and result['visibility_possible'] == possible
    assert len(result['probes']) == 4 and result['uncomputed_count'] == 4-expected_calls
    assert all(r['mean_delta_e2000'] is None and r['visible'] is None and r['status'] == 'NOT_COMPUTED_AFTER_NECESSARY_FAILURE' for r in result['probes'][expected_calls:])


def test_partition_identity_and_dtype_boundaries():
    with pytest.raises(ValueError, match='partition'):
        core.screen_visibility(probes('monitor'), grid(), candidate_split='fit')
    with pytest.raises(ValueError, match='four distinct'):
        core.screen_visibility([probes()[0]]*4, grid(), candidate_split='fit')
    with pytest.raises(ValueError, match='Unsupported'):
        core.screen_visibility(probes(), grid(), candidate_split='real')
    with pytest.raises(ValueError, match='uint16'):
        core.mean_native_delta_e(np.full((1, 1, 3), np.nan), grid())
    with pytest.raises(ValueError, match='C-contiguous'):
        core.mean_native_delta_e(np.zeros((4, 4, 3), dtype=np.uint16)[:, ::2], grid())


def test_nonfinite_grid_and_render_rejected(monkeypatch):
    for value in (np.nan, np.inf):
        invalid = grid()
        invalid[0, 0] = value
        with pytest.raises(ValueError, match='finite'):
            core.mean_native_delta_e(np.zeros((1, 1, 3), dtype=np.uint16), invalid)
    monkeypatch.setattr(core, 'render_absolute', lambda x, _v, _d: np.full_like(x, np.nan))
    with pytest.raises(ValueError, match='Nonfinite rendered'):
        core.mean_native_delta_e(np.zeros((1, 1, 3), dtype=np.uint16), grid())
