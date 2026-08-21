from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples
from src.preprocess.rec2100_pq_transfer import (
    PQ_RGB16_MAXIMUM,
    absolute_rec2020_cdm2_to_pq,
    absolute_rec2020_cdm2_to_pq_rgb16,
    pq_to_absolute_rec2020_cdm2,
    save_absolute_rec2020_cdm2_to_pq_rgb16_png,
)


def test_bt2100_pq_known_values_and_dense_monotonicity() -> None:
    scalar = np.linspace(0.0, 10000.0, 65537, dtype=np.float64)
    probe = np.repeat(scalar[:, None], 3, axis=1)
    encoded = absolute_rec2020_cdm2_to_pq(probe)[:, 0]
    samples = absolute_rec2020_cdm2_to_pq_rgb16(probe)[:, 0]
    assert abs(encoded[0] - 7.309559025783966e-07) < 1e-21
    assert encoded[-1] == 1.0
    assert samples[0] == 0
    assert samples[-1] == 65535
    assert np.all(np.diff(encoded) > 0.0)
    assert np.all(np.diff(samples.astype(np.int64)) >= 0)
    assert abs(absolute_rec2020_cdm2_to_pq(np.full((1, 3), 100.0))[0, 0] - 0.508078421517399) < 1e-15
    assert abs(absolute_rec2020_cdm2_to_pq(np.full((1, 3), 203.0))[0, 0] - 0.5806888810416109) < 1e-15
    quantized = samples.astype(np.float64) / PQ_RGB16_MAXIMUM
    assert np.max(np.abs(encoded - quantized)) <= 0.5 / 65535.0 + 1e-12


def test_bt2100_pq_reference_inverse_is_finite_and_bounded() -> None:
    codes = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    probe = np.repeat(codes[:, None], 3, axis=1)
    absolute = pq_to_absolute_rec2020_cdm2(probe)
    assert absolute[0, 0] == 0.0
    assert abs(absolute[-1, 0] - 10000.0) < 1e-9
    assert np.all(np.diff(absolute[:, 0]) > 0.0)
    replay = absolute_rec2020_cdm2_to_pq(absolute)
    assert np.max(np.abs(replay[1:] - probe[1:])) < 3e-13
    assert replay[0, 0] < 0.5 / 65535.0


@pytest.mark.parametrize(
    "values,match",
    [
        (np.zeros((2, 2)), "three RGB"),
        (np.full((1, 1, 3), np.nan), "finite"),
        (np.full((1, 1, 3), -1.0), "inside"),
        (np.full((1, 1, 3), 10000.1), "inside"),
    ],
)
def test_bt2100_pq_rejects_invalid_absolute_input(values: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        absolute_rec2020_cdm2_to_pq(values)


def test_pq_png_publication_preserves_exact_samples(tmp_path: Path) -> None:
    values = np.linspace(0.0, 2000.0, 7 * 11 * 3, dtype=np.float64).reshape(7, 11, 3)
    path = tmp_path / "absolute.png"
    file_sha, samples = save_absolute_rec2020_cdm2_to_pq_rgb16_png(
        values,
        path,
        row_count=3,
    )
    assert file_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_rec2100_pq_rgb16_png_samples(
        path,
        width=11,
        height=7,
    ) == hashlib.sha256(samples.tobytes()).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        save_absolute_rec2020_cdm2_to_pq_rgb16_png(values, path)


def test_pq_png_invalid_input_is_failure_atomic(tmp_path: Path) -> None:
    path = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="inside"):
        save_absolute_rec2020_cdm2_to_pq_rgb16_png(
            np.full((2, 3, 3), -0.01),
            path,
        )
    assert not path.exists()
