import numpy as np
import pytest

from src.eval.fable_cdfe_visual_panel import render_codes, render_comfort_panel
from src.eval.fable_reference_photometry import transform


def test_native_render_matches_direct_transform_and_gate_blocks(tmp_path):
    codes = np.random.default_rng(8).integers(0, 256, (13, 19, 3), dtype=np.uint8)
    u = np.array([.7, -.8, .5, -.4])
    expected = np.floor(255*transform(codes/255, u, slope_limit=1.25, offset_limit=.35)+.5).astype(np.uint8)
    np.testing.assert_array_equal(render_codes(codes, u), expected)
    with pytest.raises(ValueError, match='numeric gate'):
        render_comfort_panel(tmp_path/'panel', {}, {'gates': {'numeric_passed': False}}, {})
    assert not (tmp_path/'panel').exists()
