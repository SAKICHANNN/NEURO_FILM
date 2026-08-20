from __future__ import annotations

import numpy as np
import pytest

from src.preprocess.ocio_aces2_output import (
    CONFIG_CACHE_ID,
    OcioAces2RuntimeError,
    apply_aces2_output_packed,
    apply_aces2_output_scalar,
    build_aces2_numeric_fixture,
    load_aces2_config,
)


def test_pinned_config_and_fixture() -> None:
    config = load_aces2_config()
    assert config.getCacheID() == CONFIG_CACHE_ID
    fixture = build_aces2_numeric_fixture()
    assert fixture.shape == (986, 3)
    assert fixture.dtype == np.float32
    assert fixture.flags.c_contiguous


@pytest.mark.parametrize("target", ["sdr_rec709", "hdr_rec2020_pq"])
def test_scalar_and_packed_smoke_match(target: str) -> None:
    source = np.asarray([[0.18, 0.18, 0.18], [1.0, 0.2, 0.1]], dtype=np.float32)
    original = source.copy()
    scalar = apply_aces2_output_scalar(source, target)
    packed = apply_aces2_output_packed(source, target)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_allclose(scalar, packed, rtol=0.0, atol=2e-6)


def test_invalid_input_fails_closed() -> None:
    with pytest.raises(OcioAces2RuntimeError, match="Nx3 float32"):
        apply_aces2_output_packed(np.zeros((2, 3), dtype=np.float64), "sdr_rec709")
