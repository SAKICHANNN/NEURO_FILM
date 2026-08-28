from __future__ import annotations

from pathlib import Path

import imagecodecs
import numpy as np
import pytest
import tifffile

from scripts.audit_u1_2e_high_precision_adobe_rgb_tiff import run
from src.preprocess import load_working_image
from src.preprocess.adobe_rgb_icc import (
    AdobeRGBICCError,
    adobe_rgb_icc_facts,
    decode_adobe_rgb16_to_linear_rec2020,
)
from src.preprocess.prophoto_icc import d50_xyz_to_linear_rec2020

ROOT = Path(__file__).resolve().parents[1]
CLAY_PROFILE = (
    ROOT
    / "data/physical_reference/spektrafilm_dev_2026_v1/src/spektrafilm/data/icc"
    / "ellelstone/ClayRGB-elle-V2-g22.icc"
)


def _generated_profile() -> bytes:
    return imagecodecs.cms_profile(
        "rgb",
        whitepoint=(0.3127, 0.3290, 1.0),
        primaries=(0.64, 0.33, 0.21, 0.71, 0.15, 0.06),
        gamma=563.0 / 256.0,
    )


def _samples() -> np.ndarray:
    return np.asarray(
        [
            [[0, 32768, 65535], [65535, 0, 4096], [1234, 23456, 45678]],
            [[8192, 16384, 32768], [60000, 40000, 20000], [1, 2, 3]],
        ],
        dtype=np.uint16,
    )


def _write(path: Path, samples: np.ndarray, profile: bytes, orientation: int = 1) -> None:
    tifffile.imwrite(
        path,
        samples,
        photometric="rgb",
        planarconfig="contig",
        metadata=None,
        extratags=[
            (274, "H", 1, orientation, False),
            (34675, "B", len(profile), profile, False),
        ],
    )


def _independent_oracle(samples: np.ndarray, profile: bytes) -> np.ndarray:
    facts = adobe_rgb_icc_facts(profile)
    encoded = samples.astype(np.float64) / 65535.0
    linear = np.power(encoded, float(facts["gamma"]))
    xyz_d50 = linear @ np.asarray(facts["matrix"], dtype=np.float64).T
    return np.asarray(d50_xyz_to_linear_rec2020(xyz_d50), dtype=np.float32)


@pytest.mark.parametrize("profile", [_generated_profile(), CLAY_PROFILE.read_bytes()])
def test_supported_profiles_decode_exactly_to_independent_oracle(
    tmp_path: Path, profile: bytes
) -> None:
    path = tmp_path / "adobe_rgb16.tiff"
    samples = _samples()
    _write(path, samples, profile)

    working = load_working_image(path)
    expected = _independent_oracle(samples, profile)

    assert working.working_space == "linear_rec2020"
    assert working.transfer_state == "display_linear"
    assert working.bit_depth_in == 16
    assert working.pixels.tobytes() == expected.tobytes()
    assert any(
        warning.code == "embedded_adobe_rgb_to_linear_rec2020"
        for warning in working.warnings
    )


def test_orientation_precedes_adobe_rgb_transform(tmp_path: Path) -> None:
    profile = _generated_profile()
    samples = _samples()
    path = tmp_path / "orientation6.tiff"
    _write(path, samples, profile, orientation=6)

    working = load_working_image(path)
    oriented = np.ascontiguousarray(np.rot90(samples, k=3, axes=(0, 1)))

    assert working.pixels.shape == oriented.shape
    assert working.pixels.tobytes() == _independent_oracle(oriented, profile).tobytes()


def test_direct_decoder_matches_working_image(tmp_path: Path) -> None:
    profile = _generated_profile()
    samples = _samples()
    path = tmp_path / "direct.tiff"
    _write(path, samples, profile)

    assert (
        load_working_image(path).pixels.tobytes()
        == decode_adobe_rgb16_to_linear_rec2020(samples, profile).tobytes()
    )


def _mutate_tag(profile: bytes, signature: bytes, delta: int) -> bytes:
    mutated = bytearray(profile)
    count = int.from_bytes(mutated[128:132], "big")
    for index in range(count):
        cursor = 132 + index * 12
        if mutated[cursor : cursor + 4] == signature:
            offset = int.from_bytes(mutated[cursor + 4 : cursor + 8], "big")
            value = int.from_bytes(mutated[offset + 12 : offset + 16], "big", signed=True)
            mutated[offset + 12 : offset + 16] = (value + delta).to_bytes(
                4, "big", signed=True
            )
            return bytes(mutated)
    raise AssertionError(f"missing tag {signature!r}")


@pytest.mark.parametrize(
    ("signature", "delta", "message"),
    [
        (b"gTRC", 1024, "shared Adobe RGB gamma"),
        (b"rXYZ", 6554, "colourants are not"),
    ],
)
def test_wrong_semantics_reject(
    signature: bytes, delta: int, message: str
) -> None:
    with pytest.raises(AdobeRGBICCError, match=message):
        adobe_rgb_icc_facts(_mutate_tag(_generated_profile(), signature, delta))


def test_wrong_profile_remains_fail_closed_at_raster_boundary(tmp_path: Path) -> None:
    profile = _mutate_tag(_generated_profile(), b"rXYZ", 6554)
    path = tmp_path / "wrong.tiff"
    _write(path, _samples(), profile)
    with pytest.raises(ValueError, match="ICC conversion is not implemented"):
        load_working_image(path)


def test_formal_audit_passes_and_cleans_scratch(tmp_path: Path) -> None:
    report = run(
        ROOT / "configs/u1_2e_high_precision_adobe_rgb_tiff_v1.json",
        tmp_path / "report.json",
    )
    assert report["status"] == "PASS_RGB16_ADOBE_RGB_TIFF_INGRESS"
    assert all(report["gates"].values())
