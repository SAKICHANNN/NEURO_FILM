from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_p226_r1cv_rec2100_pq_runtime_compatibility import (
    build_p226_parity_fixture,
)
from src.preprocess.aces2_p3d65_canonical_pq_png import (
    publish_acescg_p3d65_1000nit_canonical_pq_png_v1,
)
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples


def _fixture() -> np.ndarray:
    return build_p226_parity_fixture().reshape(29, 34, 3)


def test_p238_samples_metadata_and_partition_are_exact(tmp_path: Path) -> None:
    source = _fixture()
    before = source.tobytes()
    forward = tmp_path / "forward.png"
    reverse = tmp_path / "reverse.png"
    digest, samples, encoded = publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
        source, forward, row_count=7
    )
    reverse_digest, reverse_samples, reverse_encoded = (
        publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
            source, reverse, row_count=7, reverse_partition=True
        )
    )
    expected = np.floor(encoded.astype(np.float64) * 65535.0 + 0.5).astype(np.uint16)
    assert source.tobytes() == before
    np.testing.assert_array_equal(samples, expected)
    np.testing.assert_array_equal(reverse_samples, samples)
    np.testing.assert_array_equal(reverse_encoded, encoded)
    assert forward.read_bytes() == reverse.read_bytes()
    assert digest == reverse_digest == hashlib.sha256(forward.read_bytes()).hexdigest()
    assert sha256_rec2100_pq_rgb16_png_samples(
        forward, width=34, height=29
    ) == hashlib.sha256(samples.tobytes()).hexdigest()
    assert b"cICP\x09\x10\x00\x01" in forward.read_bytes()


@pytest.mark.parametrize(
    "value",
    [
        np.empty((0, 1, 3), dtype=np.float32),
        np.zeros((1, 1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float32),
        np.full((1, 1, 3), np.nan, dtype=np.float32),
    ],
)
def test_p238_rejects_invalid_input_without_output(
    tmp_path: Path, value: np.ndarray
) -> None:
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError):
        publish_acescg_p3d65_1000nit_canonical_pq_png_v1(value, output)
    assert not output.exists()
    assert not output.with_suffix(".png.stream.tmp").exists()


def test_p238_is_create_only_and_validates_partition(tmp_path: Path) -> None:
    output = tmp_path / "existing.png"
    output.write_bytes(b"retained")
    with pytest.raises(FileExistsError, match="create-only"):
        publish_acescg_p3d65_1000nit_canonical_pq_png_v1(_fixture(), output)
    assert output.read_bytes() == b"retained"
    invalid = tmp_path / "rows.png"
    with pytest.raises(ValueError, match="row_count"):
        publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
            _fixture(), invalid, row_count=0
        )
    assert not invalid.exists()


def test_p238_writer_failure_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.preprocess import aces2_p3d65_canonical_pq_png as module

    original = module.CanonicalStreamingRec2100PqPngWriter.write_rows
    calls = 0

    def fail_second(self, row_start: int, samples: np.ndarray) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected write failure")
        original(self, row_start, samples)

    monkeypatch.setattr(
        module.CanonicalStreamingRec2100PqPngWriter, "write_rows", fail_second
    )
    output = tmp_path / "atomic.png"
    with pytest.raises(RuntimeError, match="injected"):
        publish_acescg_p3d65_1000nit_canonical_pq_png_v1(
            _fixture(), output, row_count=7
        )
    assert not output.exists()
    assert not output.with_suffix(".png.stream.tmp").exists()
