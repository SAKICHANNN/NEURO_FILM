import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('pilot', Path(__file__).parents[1]/'scripts/run_fable_scan_residual_pilot.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_ir_channel_does_not_change_rgb_codes():
    data = np.full((8, 8, 4), 32768, dtype=np.uint16)
    first = m.rgb_codes(data)
    data[..., 3] = 0
    assert np.array_equal(first, m.rgb_codes(data))
    assert first[0, 0, 0] == 32768/65535


@pytest.mark.parametrize('method', ['gaussian_sigma1', 'gaussian_sigma2', 'median_size3'])
def test_flat_is_not_false_texture_but_clean_fine_lines_expose_bias(method):
    clean, _ = m.controls(64, 1, .01)
    assert np.max(np.abs(m.residual(clean['flat'], method))) < 1e-12
    assert np.sqrt(np.mean(m.residual(clean['one_pixel_lines'], method)**2)) > .01


def test_known_injection_is_reproducible_zero_mean_and_rms_scaled():
    _, noise = m.controls(64, 1, .01)
    _, repeated = m.controls(64, 1, .01)
    assert np.array_equal(noise, repeated)
    np.testing.assert_allclose(noise.mean(axis=(0, 1)), 0, atol=1e-15)
    np.testing.assert_allclose(np.sqrt((noise**2).mean(axis=(0, 1))), .01, atol=1e-15)
