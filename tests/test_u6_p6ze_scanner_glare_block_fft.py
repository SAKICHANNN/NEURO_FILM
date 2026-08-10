from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    ScannerGlareDomainError,
    apply_scanner_glare,
    compile_scanner_glare_kernel,
)
from src.film_physics.scanner_glare_block_fft import (
    apply_scanner_glare_block_fft,
    apply_scanner_glare_channel_serial_block_fft,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6ze_scanner_glare_block_fft_v1.json"


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


def test_contract_freezes_materially_distinct_block_fft() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert (
        contract["schema"] == "neuro_film.u6_p6ze_scanner_glare_block_fft_contract.v1"
    )
    assert contract["mechanism"]["row_chunk"] == 512
    assert contract["mechanism"]["functional_row_partitions"] == [257, 512, 769]
    assert contract["production_integration_allowed"] is False


@pytest.mark.parametrize("row_chunk", [257, 512, 769])
def test_block_fft_matches_full_reference(row_chunk: int) -> None:
    source, kernel, flare = _fixture()
    reference = apply_scanner_glare(source, kernel, flare_fraction=flare)
    candidate = apply_scanner_glare_block_fft(
        source, kernel, flare_fraction=flare, row_chunk=row_chunk
    )
    difference = candidate - reference
    assert float(np.max(np.abs(difference))) <= 1e-12
    assert float(np.sqrt(np.mean(np.square(difference)))) <= 1e-13


@pytest.mark.parametrize("row_chunk", [257, 512, 769])
def test_channel_serial_block_fft_is_exactly_the_same_operator(row_chunk: int) -> None:
    source, kernel, flare = _fixture()
    original = apply_scanner_glare_block_fft(
        source, kernel, flare_fraction=flare, row_chunk=row_chunk
    )
    candidate = apply_scanner_glare_channel_serial_block_fft(
        source, kernel, flare_fraction=flare, row_chunk=row_chunk
    )
    assert np.array_equal(candidate, original)


def test_block_fft_preserves_constant_and_impulse_energy() -> None:
    _, kernel, flare = _fixture()
    constant = np.full((127, 139), 0.42, dtype=np.float64)
    constant_output = apply_scanner_glare_block_fft(
        constant, kernel, flare_fraction=flare, row_chunk=31
    )
    assert np.max(np.abs(constant_output - constant)) <= 1e-12
    impulse = np.zeros((257, 257), dtype=np.float64)
    impulse[128, 128] = 1.0
    impulse_output = apply_scanner_glare_block_fft(
        impulse, kernel, flare_fraction=1.0 - 1e-15, row_chunk=64
    )
    assert abs(float(np.sum(impulse_output)) - 1.0) <= 1e-12


@pytest.mark.parametrize("row_chunk", [0, True])
def test_block_fft_rejects_invalid_row_chunk(row_chunk: object) -> None:
    source, kernel, flare = _fixture()
    with pytest.raises(ScannerGlareDomainError, match="row chunk"):
        apply_scanner_glare_block_fft(
            source,
            kernel,
            flare_fraction=flare,
            row_chunk=row_chunk,  # type: ignore[arg-type]
        )
