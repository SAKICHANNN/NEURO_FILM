from __future__ import annotations

import numpy as np
import pytest

from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    apply_scanner_glare,
    compile_scanner_glare_kernel,
)
from src.film_physics.scanner_glare_block_fft import apply_scanner_glare_block_fft
from src.film_physics.scanner_glare_channel_serial_fft import (
    apply_scanner_glare_channel_serial_block_fft,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(6206105)
    source = rng.uniform(0.01, 0.99, size=(321, 513, 3))
    profile = MultiscaleScannerGlareProfile(
        components=(
            ScannerGlareComponent(weight=0.8, sigma_pixels=2.0),
            ScannerGlareComponent(weight=0.2, sigma_pixels=12.0),
        ),
        flare_fraction=0.08,
        truncate_sigma=6.0,
    )
    return source, compile_scanner_glare_kernel(profile, kernel_size=177), 0.08


@pytest.mark.parametrize("row_chunk", [257, 512, 769])
def test_channel_serial_is_exactly_p6ze_and_matches_reference(row_chunk: int) -> None:
    source, kernel, flare = _fixture()
    original = apply_scanner_glare_block_fft(
        source, kernel, flare_fraction=flare, row_chunk=row_chunk
    )
    candidate = apply_scanner_glare_channel_serial_block_fft(
        source, kernel, flare_fraction=flare, row_chunk=row_chunk
    )
    reference = apply_scanner_glare(source, kernel, flare_fraction=flare)
    assert np.array_equal(candidate, original)
    assert float(np.max(np.abs(candidate - reference))) <= 1e-12
