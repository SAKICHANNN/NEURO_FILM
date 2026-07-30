from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.fivek_fresh_normalization_support import (
    _alignment,
    center_crop_to_aspect,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay3s2_fivek_fresh_normalization_support_v1.json"
)


def test_contract_preserves_failed_parent_and_forbids_fitting() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["acquisition"]["rows"]) == 64
    assert config["operator_fitting_allowed"] is False


def test_center_crop_landscape_to_narrower_landscape() -> None:
    source = np.zeros((100, 180, 3), dtype=np.float32)
    cropped = center_crop_to_aspect(source, 100, 150)
    assert cropped.shape == (100, 150, 3)
    assert cropped.flags.c_contiguous


def test_center_crop_portrait_to_shorter_portrait() -> None:
    source = np.zeros((180, 100, 3), dtype=np.float32)
    cropped = center_crop_to_aspect(source, 150, 100)
    assert cropped.shape == (150, 100, 3)
    assert cropped.flags.c_contiguous


def test_alignment_repeat_is_exact_with_optimized_kernels_disabled() -> None:
    rng = np.random.default_rng(20260730)
    source = rng.random((96, 128, 3), dtype=np.float32)
    target = np.roll(source, shift=(1, -2), axis=(0, 1))
    first = _alignment(source, target, 96, 9)
    assert all(
        _alignment(source, target, 96, 9) == first for _ in range(8)
    )
