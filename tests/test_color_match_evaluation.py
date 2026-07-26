from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.color_match import (
    ReferenceMatchContractError,
    evaluate_known_operator_batch,
)
from src.preprocess import SourceProfile, WorkingImage


def _working(
    pixels: np.ndarray,
    *,
    transfer_state: str = "display_linear",
) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space="linear_srgb",
        transfer_state=transfer_state,
        source_transfer_state="display_referred",
        source_profile=SourceProfile("assumed_srgb", "test fixture"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("fixture.png"),
    )


def test_known_operator_metrics_reward_target_recovery() -> None:
    source = _working(
        np.asarray(
            [
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]],
                [[0.5, 0.6, 0.7], [0.7, 0.8, 0.9]],
            ],
            dtype=np.float32,
        )
    )
    target = _working(np.clip(source.pixels * 0.8 + 0.05, 0.0, 1.0))
    result = evaluate_known_operator_batch(
        [source],
        [target],
        [target],
        sample_ids=["sample-1"],
    )

    assert result.improved_sample_count == 1
    assert result.regressed_sample_count == 0
    assert result.improvement_rate == 1.0
    assert result.samples[0].median_improvement_fraction == pytest.approx(1.0)
    assert result.samples[0].candidate_target_delta_e76_median == 0.0


def test_known_operator_metrics_expose_regression_and_new_boundaries() -> None:
    source = _working(np.full((3, 4, 3), 0.4, dtype=np.float32))
    target = _working(np.full((3, 4, 3), 0.45, dtype=np.float32))
    candidate = _working(np.zeros((3, 4, 3), dtype=np.float32))
    result = evaluate_known_operator_batch(
        [source],
        [target],
        [candidate],
        sample_ids=["regression"],
    )

    assert result.regressed_sample_count == 1
    assert result.worst_improvement_fraction < 0.0
    assert result.maximum_new_boundary_fraction == 1.0


def test_known_operator_contract_rejects_misaligned_or_unsupported_inputs() -> None:
    image = _working(np.full((2, 2, 3), 0.4, dtype=np.float32))
    different_shape = _working(np.full((3, 2, 3), 0.4, dtype=np.float32))
    with pytest.raises(ReferenceMatchContractError, match="shapes must match"):
        evaluate_known_operator_batch(
            [image],
            [different_shape],
            [image],
            sample_ids=["shape"],
        )
    with pytest.raises(ReferenceMatchContractError, match="display-linear"):
        evaluate_known_operator_batch(
            [_working(image.pixels, transfer_state="scene_linear")],
            [image],
            [image],
            sample_ids=["state"],
        )
    with pytest.raises(ReferenceMatchContractError, match="equal length"):
        evaluate_known_operator_batch(
            [image],
            [image],
            [image],
            sample_ids=[],
        )
