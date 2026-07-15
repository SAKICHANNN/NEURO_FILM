from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.roll2film.blueneg_eval import (
    BlueNegEvaluationError,
    align_blueneg_pair,
    load_development_roll_samples,
)


def test_alignment_intersects_negative_bbox_without_negative_indexing() -> None:
    preview = np.arange(8 * 9 * 3, dtype=np.float64).reshape(8, 9, 3)
    target = np.arange(7 * 8 * 3, dtype=np.float64).reshape(7, 8, 3)

    source_crop, target_crop = align_blueneg_pair(
        preview,
        target,
        [-2, -1, 6, 6],
        minimum_side=2,
    )

    assert source_crop.shape == target_crop.shape == (6, 6, 3)
    np.testing.assert_array_equal(source_crop, preview[0:6, 0:6])
    np.testing.assert_array_equal(target_crop, target[1:7, 2:8])


def test_alignment_rejects_target_shape_not_matching_bbox() -> None:
    with pytest.raises(BlueNegEvaluationError, match="shape does not match"):
        align_blueneg_pair(
            np.zeros((80, 80, 3)),
            np.zeros((70, 70, 3)),
            [0, 0, 80, 80],
        )


def test_development_loader_rejects_confirmatory_roll_before_decode(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "filename": "confirm-frame",
            "roll_id": "confirm-roll",
            "research_pool": "confirmatory_roll",
            "frame_role": "unpaired_support",
            "preview_path": "missing-preview.png",
            "pseudogt_path": "missing-target.png",
        }
    ]

    with pytest.raises(BlueNegEvaluationError, match="rejects non-development"):
        load_development_roll_samples(
            root=tmp_path,
            rows=rows,
            transformations={},
            roll_id="confirm-roll",
            support_frames=1,
            support_pixels=16,
            query_frames=1,
            query_pixels=16,
            support_seed=1,
            query_seed=2,
            minimum_side=4,
        )


def test_development_sampling_is_deterministic_and_pair_blind(tmp_path: Path) -> None:
    rows = []
    transformations = {}
    for index in range(4):
        filename = f"frame-{index}"
        preview = np.zeros((80, 80, 3), dtype=np.uint8)
        preview[..., 0] = np.arange(80, dtype=np.uint8)[None, :]
        target = np.roll(preview, index + 1, axis=2)
        preview_path = tmp_path / f"{filename}-preview.png"
        target_path = tmp_path / f"{filename}-target.png"
        Image.fromarray(preview).save(preview_path)
        Image.fromarray(target).save(target_path)
        rows.append(
            {
                "filename": filename,
                "roll_id": "dev-roll",
                "research_pool": "development_roll",
                "frame_role": "unpaired_support" if index < 2 else "hidden_aligned_query",
                "preview_path": preview_path.name,
                "pseudogt_path": target_path.name,
            }
        )
        transformations[filename] = {
            "matrix": np.eye(3),
            "bbox": np.array([0, 0, 80, 80]),
        }
    kwargs = dict(
        root=tmp_path,
        rows=rows,
        transformations=transformations,
        roll_id="dev-roll",
        support_frames=2,
        support_pixels=32,
        query_frames=2,
        query_pixels=32,
        support_seed=11,
        query_seed=12,
        minimum_side=64,
    )

    first = load_development_roll_samples(**kwargs)
    second = load_development_roll_samples(**kwargs)

    np.testing.assert_array_equal(first.support_source, second.support_source)
    np.testing.assert_array_equal(first.support_target, second.support_target)
    np.testing.assert_array_equal(first.query_source, second.query_source)
    np.testing.assert_array_equal(first.query_target, second.query_target)
    assert first.query_source.shape == (2, 32, 3)
