from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.radiance_rgbe import (
    RadianceRgbeError,
    _decode_radiance_rgbe_codes_bytes,
    decode_radiance_rgbe_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p305_polyhaven_radiance_rgbe_intake_v1.json"


def _tiny_payload() -> bytes:
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n"
    marker = b"\x02\x02\x00\x08"
    channels = b"".join(
        bytes([128 + 8, value]) for value in (1, 2, 3, 136)
    )
    return header + marker + channels


def test_p305_tiny_scanline_decodes_half_bin_convention() -> None:
    output = decode_radiance_rgbe_bytes(_tiny_payload())
    assert output.shape == (1, 8, 3)
    assert output.dtype == np.float32
    assert output.flags.owndata and output.flags.c_contiguous and output.flags.writeable
    assert np.array_equal(output[0, 0], np.array([1.5, 2.5, 3.5], np.float32))


def test_p305_tiny_scanline_preserves_codes_for_independent_oracle() -> None:
    codes = _decode_radiance_rgbe_codes_bytes(_tiny_payload())
    assert codes.shape == (1, 8, 4)
    assert codes.dtype == np.uint8
    assert codes.flags.owndata and codes.flags.c_contiguous and codes.flags.writeable
    assert np.array_equal(codes[0, 0], np.array([1, 2, 3, 136], np.uint8))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: body.replace(b"-Y 1 +X 8", b"+Y 1 +X 8"),
        lambda body: body[:-1],
        lambda body: body + b"x",
        lambda body: body.replace(b"\x88\x01", b"\x00\x01", 1),
        lambda body: body.replace(b"\x02\x02\x00\x08", b"\x02\x02\x00\x09"),
    ],
)
def test_p305_invalid_payloads_fail_closed(mutation) -> None:
    with pytest.raises(RadianceRgbeError):
        decode_radiance_rgbe_bytes(mutation(_tiny_payload()))


def test_p305_contract_keeps_color_identity_unassigned() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["source"]["bytes"] == 1573705
    assert config["source"]["md5"] == "97335a81ba615beb6f6ae0da707ecd75"
    assert config["gates"]["maximum_absolute_official_semantics_residual"] == 1e-7
    claim = config["claim_ceiling"].casefold()
    assert "primaries" in claim and "workingimage" in claim and "candidate 3" in claim
