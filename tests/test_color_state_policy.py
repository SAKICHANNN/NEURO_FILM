from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.preprocess import SourceProfile, WorkingImage, resolve_look_approximation_claim
from src.preprocess.types import TransferState


def _working(source_state: TransferState) -> WorkingImage:
    return WorkingImage(
        pixels=np.zeros((2, 3, 3), dtype=np.float32),
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state=source_state,
        source_profile=SourceProfile("unknown", "test"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=8,
        source_path=Path("test.png"),
    )


def test_known_input_remains_uncalibrated_look_approximation() -> None:
    claim = resolve_look_approximation_claim(_working("display_referred"))
    assert claim["output_label"] == "film-inspired"
    assert claim["color_state_policy"] == "look_approximation_only"
    assert claim["calibrated_reference_allowed"] is False


def test_unknown_input_fails_closed_to_look_approximation() -> None:
    claim = resolve_look_approximation_claim(_working("unknown"))
    assert claim["input_color_state"] == "unknown"
    assert claim["color_state_policy"] == "look_approximation_fail_closed"
    assert claim["calibrated_reference_allowed"] is False


def test_working_image_rejects_nonfinite_pixels() -> None:
    pixels = np.zeros((2, 3, 3), dtype=np.float32)
    pixels[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        WorkingImage(
            pixels=pixels,
            working_space="linear_srgb",
            transfer_state="display_linear",
            source_transfer_state="display_referred",
            source_profile=SourceProfile("assumed_srgb", "test"),
            hdr_metadata={},
            orientation_applied=True,
            alpha_policy="absent",
            bit_depth_in=8,
            source_path=Path("test.png"),
        )
