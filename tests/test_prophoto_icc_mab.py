from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from src.preprocess import load_working_image
from src.preprocess.color_management import linear_rgb_matrix
from src.preprocess.prophoto_icc import (
    ProPhotoICCError,
    decode_prophoto_rgb16_to_linear_rec2020,
    prophoto_icc_facts,
    prophoto_matrix_shaper_facts,
)

_OFFICIAL_ROMM_PROFILE_B64 = (
    "AAADYG5vbmUEAAAAc3BhY1JHQiBYWVogB9YACwATAA8AEwA1YWNzcAAAAAAAAAAAbm9uZW5vbmU"
    "AAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1ub25lLJihZpUlfVITkG4EAsDqyQAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAGZGVzYwAAAMwAAABUQTJCMAAAASAAAADUQjJBMAAAAfQAAADUd3Rw"
    "dAAAAsgAAAAUY3BydAAAAtwAAABYY2hhZAAAAzQAAAAsbWx1YwAAAAAAAAABAAAADGVuVVMAAAA4"
    "AAAAHABJAFMATwAgADIAMgAwADIAOAAtADIAIABSAE8ATQBNACAAUgBHAEIAIABwAHIAbwBmAGkA"
    "bABlbUFCIAAAAAADAwAAAAAAIAAAAEQAAAB0AAAAAAAAAABjdXJ2AAAAAAAAAABjdXJ2AAAAAAAA"
    "AABjdXJ2AAAAAAAAAAAAAGYZAAARTgAABAMAACTeAABbHgAAACIAAAAAAAAAAAAAaXwAAABuAAAA"
    "cgAAAF5wYXJhAAAAAAADAAAAAczNAAEAAAAAAAAAABAAAAAIAHBhcmEAAAAAAAMAAAABzM0AAQAA"
    "AAAAAAAAEAAAAAgAcGFyYQAAAAAAAwAAAAHMzQABAAAAAAAAAAAQAAAACABtQkEgAAAAAAMDAAAA"
    "AAAgAAAARAAAAHQAAAAAAAAAAGN1cnYAAAAAAAAAAGN1cnYAAAAAAAAAAGN1cnYAAAAAAAAAAAAC"
    "sSj//30d///l9f/+6SsAAwQzAAAJoQAAAAAAAAAAAAJtSf///zX///81////NXBhcmEAAAAAAAMA"
    "AAAAjjkAAQAAAAAAAAAQAAAAAACAcGFyYQAAAAAAAwAAAACOOQABAAAAAAAAABAAAAAAAIBwYXJh"
    "AAAAAAADAAAAAI45AAEAAAAAAAAAEAAAAAAAgFhZWiAAAAAAAADbrAAA49cAALv1bWx1YwAAAAAA"
    "AAABAAAADGVuVVMAAAA8AAAAHABDAG8AcAB5AHIAaQBnAGgAdAAgADIAMAAwADYAIABIAGUAdwBs"
    "AGUAdAB0AC0AUABhAGMAawBhAHIAZHNmMzIAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA"
    "AAAAAAAAAQAA"
)
_OFFICIAL_ROMM_PROFILE = base64.b64decode(_OFFICIAL_ROMM_PROFILE_B64)
_D50 = np.asarray([0.9642, 1.0, 0.8249], dtype=np.float64)
_D50_TO_D65_BRADFORD = np.asarray(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float64,
)


def _product_lab_codes(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rec2020 = decode_prophoto_rgb16_to_linear_rec2020(
        (encoded.astype(np.uint16) * np.uint16(257)).reshape(1, -1, 3),
        _OFFICIAL_ROMM_PROFILE,
    ).reshape(-1, 3).astype(np.float64)
    linear_srgb = rec2020 @ linear_rgb_matrix(
        "linear_rec2020", "linear_srgb"
    ).astype(np.float64).T
    srgb_to_xyz_d65 = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz_d50 = (
        linear_srgb @ srgb_to_xyz_d65.T
    ) @ np.linalg.inv(_D50_TO_D65_BRADFORD).T
    ratio = xyz_d50 / _D50
    delta = 6.0 / 29.0
    f = np.where(
        ratio > delta**3,
        np.cbrt(ratio),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lab = np.stack(
        [
            116.0 * f[:, 1] - 16.0,
            500.0 * (f[:, 0] - f[:, 1]),
            200.0 * (f[:, 1] - f[:, 2]),
        ],
        axis=1,
    )
    codes = np.column_stack(
        [
            np.clip(np.rint(lab[:, 0] * 255.0 / 100.0), 0.0, 255.0),
            np.rint(lab[:, 1]).astype(np.int64) % 256,
            np.rint(lab[:, 2]).astype(np.int64) % 256,
        ]
    ).astype(np.uint8)
    return lab, codes


def test_official_romm_profile_is_exact_and_strictly_classified() -> None:
    assert len(_OFFICIAL_ROMM_PROFILE) == 864
    assert hashlib.sha256(_OFFICIAL_ROMM_PROFILE).hexdigest() == (
        "96b2f2987f83e2a545e607799fbfdff43ef8158fb9b215b187c574db8f145aaf"
    )
    facts = prophoto_icc_facts(_OFFICIAL_ROMM_PROFILE)
    assert facts["transform_kind"] == "mab-romm-type3-matrix-offset-pcsxyz"
    assert facts["transfer_kind"] == "parametric-type3"
    np.testing.assert_allclose(
        facts["transfer_parameters"],
        [1.8, 1.0, 0.0, 1 / 16, 1 / 32],
        atol=4e-6,
    )
    assert facts["maximum_prophoto_matrix_absolute_error"] < 0.002
    assert facts["maximum_reference_media_white_absolute_error"] < 5e-5
    assert facts["maximum_pcs_encoded_white_absolute_error"] < 0.005
    with pytest.raises(ProPhotoICCError, match="not a matrix-shaper"):
        prophoto_matrix_shaper_facts(_OFFICIAL_ROMM_PROFILE)


def test_product_decoder_matches_littlecms_lab_codes_on_official_profile() -> None:
    values = np.linspace(0, 255, 17, dtype=np.uint8)
    encoded = np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1)
    encoded = encoded.reshape(-1, 3)
    transform = ImageCms.buildTransformFromOpenProfiles(
        ImageCms.getOpenProfile(BytesIO(_OFFICIAL_ROMM_PROFILE)),
        ImageCms.createProfile("LAB", 5000),
        "RGB",
        "LAB",
        renderingIntent=1,
        flags=0,
    )
    littlecms = np.asarray(
        ImageCms.applyTransform(
            Image.fromarray(encoded.reshape(1, -1, 3), "RGB"), transform
        )
    ).reshape(-1, 3)
    lab, product = _product_lab_codes(encoded)
    representable = (
        (lab[:, 0] >= 0.0)
        & (lab[:, 0] <= 100.0)
        & (lab[:, 1] >= -127.5)
        & (lab[:, 1] <= 127.5)
        & (lab[:, 2] >= -127.5)
        & (lab[:, 2] <= 127.5)
    )
    difference = np.abs(
        littlecms[representable].astype(np.int16)
        - product[representable].astype(np.int16)
    )
    assert int(representable.sum()) == 4135
    assert int(difference.max()) <= 1
    assert int(np.count_nonzero(difference)) <= 250


def test_official_romm_profile_uses_high_precision_product_tiff_ingress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "official_romm.tiff"
    encoded = np.asarray(
        [
            [[0, 0, 0], [65535, 65535, 65535]],
            [[4096, 32768, 61440], [49152, 8192, 32768]],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(
        path,
        encoded,
        photometric="rgb",
        metadata=None,
        extratags=[
            (
                34675,
                "B",
                len(_OFFICIAL_ROMM_PROFILE),
                _OFFICIAL_ROMM_PROFILE,
                False,
            )
        ],
    )
    working = load_working_image(path)
    expected = decode_prophoto_rgb16_to_linear_rec2020(
        encoded, _OFFICIAL_ROMM_PROFILE
    )
    assert working.working_space == "linear_rec2020"
    assert working.transfer_state == "display_linear"
    assert working.pixels.tobytes() == expected.tobytes()
    assert [warning.code for warning in working.warnings] == [
        "embedded_prophoto_to_linear_rec2020"
    ]
