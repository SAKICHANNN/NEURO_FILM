from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.preprocess import inspect_input, load_working_image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/external/u7_21a_strict_sdr_heic_v1"
POSITIVE = SOURCE_ROOT / "heif_other__arrow.heic"


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_exact_display_p3_heic_enters_owned_linear_srgb() -> None:
    inspection = inspect_input(POSITIVE)
    assert inspection.source_kind == "raster"
    assert inspection.format_name == "HEIC"
    assert (inspection.width, inspection.height) == (3024, 4032)
    assert inspection.mode == "RGB"
    assert inspection.bit_depth == 8
    assert inspection.frame_count == 1
    assert inspection.has_alpha is False
    assert inspection.orientation == 1
    assert inspection.source_profile.kind == "icc"
    assert inspection.hdr_metadata["strict_sdr_heic"] == "accepted"

    working = load_working_image(POSITIVE)
    assert working.pixels.shape == (4032, 3024, 3)
    assert working.pixels.dtype == np.float32
    assert working.pixels.flags.owndata
    assert np.isfinite(working.pixels).all()
    assert working.working_space == "linear_srgb"
    assert working.transfer_state == "display_linear"
    assert working.orientation_applied is True
    assert _sha256(working.pixels) == (
        "068b9e4ed3b091be6ceefe3d75a814c5acaa3791f407bc705b6a201bcf52440a"
    )


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("heif__RGB_10__128x128.heif", "8-bit"),
        ("heif__RGBA_8__128x128.heif", "RGB"),
        ("heif__RGB_8__128x128.heif", "colour|ICC|NCLX"),
        ("heif_other__nokia__bird_burst.heic", "sequence|still-image|one"),
        ("heif_other__pug.heic", "auxiliary|depth|gain"),
        ("heif_special__aux_YCbCr.heic", "auxiliary|gain"),
        ("heif_corrupted__corrupted.heic", "invalid|corrupt|input"),
        ("heif_truncated__truncated.heic", "sequence|one|truncated"),
    ],
)
def test_non_strict_heic_rejects_before_sdr_fallback(name: str, reason: str) -> None:
    path = SOURCE_ROOT / name
    inspection = inspect_input(path)
    assert inspection.format_name == "HEIC"
    assert inspection.hdr_metadata["strict_sdr_heic"] == "rejected"
    assert any(
        warning.code == "unsupported_dynamic_range" for warning in inspection.warnings
    )
    with pytest.raises(ValueError, match=reason):
        load_working_image(path)


def test_product_requirements_choose_decode_only_heif_dependency() -> None:
    requirements = (ROOT / "requirements-product-v2.txt").read_text(encoding="utf-8")
    assert "pi-heif==1.4.0" in requirements
    assert "pillow-heif" not in requirements.casefold()
