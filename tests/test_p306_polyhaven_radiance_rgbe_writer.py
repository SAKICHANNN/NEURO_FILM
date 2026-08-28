from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.preprocess.radiance_rgbe import (
    RadianceRgbeError,
    _decode_radiance_rgbe_codes_bytes,
    decode_radiance_rgbe_bytes,
    encode_radiance_rgbe_bytes,
    write_radiance_rgbe_create_only,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/p305_polyhaven_rgbe_v1/sunset_jhbcentral_1k.hdr"


def _probe() -> np.ndarray:
    row = np.array(
        [[1.5, 2.5, 3.5], [0.0, 0.0, 0.0]] * 4,
        dtype=np.float32,
    )
    return row[None, ...]


def test_p306_tiny_roundtrip_is_quantization_stable() -> None:
    values = _probe()
    payload = encode_radiance_rgbe_bytes(values)
    decoded = decode_radiance_rgbe_bytes(payload)
    assert np.array_equal(
        _decode_radiance_rgbe_codes_bytes(encode_radiance_rgbe_bytes(decoded)),
        _decode_radiance_rgbe_codes_bytes(payload),
    )
    assert np.array_equal(
        decoded[0, 0], np.array([1.4921875, 2.4921875, 3.4921875], np.float32)
    )
    assert _decode_radiance_rgbe_codes_bytes(payload).shape == (1, 8, 4)


def test_p306_exact_p305_source_codes_roundtrip() -> None:
    source = SOURCE.read_bytes()
    values = decode_radiance_rgbe_bytes(source)
    encoded = encode_radiance_rgbe_bytes(values)
    assert np.array_equal(
        _decode_radiance_rgbe_codes_bytes(encoded),
        _decode_radiance_rgbe_codes_bytes(source),
    )
    assert np.array_equal(decode_radiance_rgbe_bytes(encoded), values)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((0, 8, 3), np.float32),
        np.zeros((1, 7, 3), np.float32),
        np.zeros((1, 8, 4), np.float32),
        np.full((1, 8, 3), -1.0, np.float32),
        np.full((1, 8, 3), np.nan, np.float32),
        np.zeros((1, 8, 3), dtype=np.bool_),
    ],
)
def test_p306_invalid_values_fail_closed(values: np.ndarray) -> None:
    with pytest.raises(RadianceRgbeError):
        encode_radiance_rgbe_bytes(values)


def test_p306_create_only_publication(tmp_path: Path) -> None:
    output = tmp_path / "probe.hdr"
    write_radiance_rgbe_create_only(output.resolve(), _probe())
    before = output.read_bytes()
    with pytest.raises(RadianceRgbeError):
        write_radiance_rgbe_create_only(output.resolve(), _probe())
    assert output.read_bytes() == before
    assert not tuple(tmp_path.glob("*.stage"))
