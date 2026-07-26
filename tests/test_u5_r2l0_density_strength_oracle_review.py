from __future__ import annotations

from pathlib import Path

from src.eval.density_strength_oracle_review import crop_boxes


def test_crop_boxes_cover_corners_and_centre_without_scaling() -> None:
    assert crop_boxes(1000, 800, 192) == [
        (0, 0, 192, 192),
        (808, 0, 1000, 192),
        (404, 304, 596, 496),
        (0, 608, 192, 800),
        (808, 608, 1000, 800),
    ]


def test_crop_boxes_shrink_only_for_small_images() -> None:
    assert crop_boxes(100, 80, 192) == [
        (0, 0, 80, 80),
        (20, 0, 100, 80),
        (10, 0, 90, 80),
        (0, 0, 80, 80),
        (20, 0, 100, 80),
    ]


def test_review_runner_normalizes_explicit_relative_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/make_u5_r2l0_density_strength_oracle_review.py"
    ).read_text(encoding="utf-8")
    assert "if args.config.is_absolute()" in source
    assert "if args.selected_manifest.is_absolute()" in source
    assert "if args.e1_config.is_absolute()" in source
    assert "if args.output_dir.is_absolute()" in source
