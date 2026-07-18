from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocess import (
    LINEAR_RGB_TRANSFORM_VERSION,
    DecodeWarning,
    SourceProfile,
    WorkingImage,
    convert_linear_rgb,
    convert_working_image_space,
    linear_rgb_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u1_4a_linear_rec2020_primitive_v1.json").read_text(
        encoding="utf-8"
    )
)


def _working(pixels: np.ndarray, *, transfer_state: str = "display_linear") -> WorkingImage:
    return WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state=transfer_state,
        source_transfer_state="display_referred",
        source_profile=SourceProfile("assumed_srgb", "test fixture"),
        hdr_metadata={"nested": {"headroom": "unknown"}},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("fixture.tiff"),
        warnings=[DecodeWarning("fixture", "original warning")],
    )


def test_derived_matrices_match_frozen_official_composition() -> None:
    expected_forward = np.asarray(
        CONFIG["column_vector_matrices"]["linear_srgb_to_linear_rec2020"],
        dtype=np.float64,
    )
    expected_inverse = np.asarray(
        CONFIG["column_vector_matrices"]["linear_rec2020_to_linear_srgb"],
        dtype=np.float64,
    )
    forward = linear_rgb_matrix("linear_srgb", "linear_rec2020")
    inverse = linear_rgb_matrix("linear_rec2020", "linear_srgb")
    gate = CONFIG["gates"]["derived_matrix_absolute_error_max"]
    assert LINEAR_RGB_TRANSFORM_VERSION == CONFIG["transform_version"]
    assert np.max(np.abs(forward - expected_forward)) <= gate
    assert np.max(np.abs(inverse - expected_inverse)) <= gate
    assert np.max(np.abs(inverse @ forward - np.eye(3))) <= CONFIG["gates"][
        "matrix_inverse_identity_error_max"
    ]


def test_primaries_white_and_extended_rec2020_green_match_golden_vectors() -> None:
    primaries_and_white = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float32
    ).reshape(1, 4, 3)
    converted = convert_linear_rgb(
        primaries_and_white,
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    ).reshape(4, 3)
    expected = np.vstack(
        [
            linear_rgb_matrix("linear_srgb", "linear_rec2020").T,
            np.ones((1, 3)),
        ]
    ).astype(np.float32)
    assert np.max(np.abs(converted - expected)) <= 2e-7

    rec2020_green = np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    srgb = convert_linear_rgb(
        rec2020_green,
        source_space="linear_rec2020",
        destination_space="linear_srgb",
    )[0, 0]
    assert srgb[0] < 0.0 and srgb[1] > 1.0 and srgb[2] < 0.0
    assert np.max(
        np.abs(
            srgb
            - linear_rgb_matrix("linear_rec2020", "linear_srgb")[:, 1].astype(
                np.float32
            )
        )
    ) <= 2e-7


def test_extended_float32_roundtrip_is_bounded_without_mutation_or_clipping() -> None:
    rng = np.random.default_rng(1401)
    source = rng.uniform(-0.25, 4.0, size=(73, 101, 3)).astype(np.float32)
    before = source.tobytes()
    writeable = source.flags.writeable
    wide = convert_linear_rgb(
        source,
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    )
    restored = convert_linear_rgb(
        wide,
        source_space="linear_rec2020",
        destination_space="linear_srgb",
    )
    assert source.tobytes() == before
    assert source.flags.writeable == writeable
    assert wide.dtype == np.float32 and restored.dtype == np.float32
    assert float(wide.min()) < 0.0 or float(wide.max()) > 1.0
    assert np.max(np.abs(restored - source)) <= CONFIG["gates"][
        "float32_extended_roundtrip_absolute_error_max"
    ]


def test_neutral_axis_and_same_space_copy() -> None:
    neutral = np.linspace(-0.25, 4.0, 257, dtype=np.float32)
    source = np.repeat(neutral[:, None], 3, axis=1).reshape(1, -1, 3)
    wide = convert_linear_rgb(
        source,
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    )
    assert np.max(np.abs(wide - source)) <= CONFIG["gates"][
        "neutral_axis_absolute_error_max"
    ]
    copied = convert_linear_rgb(
        source, source_space="linear_srgb", destination_space="linear_srgb"
    )
    assert copied.tobytes() == source.tobytes()
    assert copied is not source


def test_working_image_conversion_preserves_provenance_without_aliasing() -> None:
    source = _working(np.asarray([[[0.2, 0.4, 0.8]]], dtype=np.float32))
    converted = convert_working_image_space(source, "linear_rec2020")
    assert converted.working_space == "linear_rec2020"
    assert converted.transfer_state == source.transfer_state
    assert converted.source_transfer_state == source.source_transfer_state
    assert converted.source_profile == source.source_profile
    assert converted.orientation_applied == source.orientation_applied
    assert converted.alpha_policy == source.alpha_policy
    assert converted.bit_depth_in == source.bit_depth_in
    assert converted.source_path == source.source_path
    assert converted.hdr_metadata == source.hdr_metadata
    assert converted.hdr_metadata is not source.hdr_metadata
    assert converted.hdr_metadata["nested"] is not source.hdr_metadata["nested"]
    assert converted.warnings[:-1] == source.warnings
    assert converted.warnings[-1].code == "working_space_conversion"
    assert LINEAR_RGB_TRANSFORM_VERSION in converted.warnings[-1].message


@pytest.mark.parametrize(
    "pixels,source,destination,error",
    [
        (np.zeros((2, 3, 3), np.float64), "linear_srgb", "linear_rec2020", TypeError),
        (np.zeros((2, 3), np.float32), "linear_srgb", "linear_rec2020", ValueError),
        (np.zeros((2, 3, 3), np.float32), "unknown", "linear_rec2020", ValueError),
        (np.zeros((2, 3, 3), np.float32), "linear_srgb", "acescg", ValueError),
    ],
)
def test_invalid_linear_conversion_contract_fails_closed(
    pixels: np.ndarray, source: str, destination: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        convert_linear_rgb(
            pixels, source_space=source, destination_space=destination
        )


def test_nonfinite_and_nonlinear_state_fail_closed() -> None:
    pixels = np.zeros((2, 3, 3), np.float32)
    pixels[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        convert_linear_rgb(
            pixels,
            source_space="linear_srgb",
            destination_space="linear_rec2020",
        )
    nonlinear = _working(
        np.zeros((2, 3, 3), np.float32), transfer_state="display_referred"
    )
    with pytest.raises(ValueError, match="requires linear"):
        convert_working_image_space(nonlinear, "linear_rec2020")
