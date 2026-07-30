from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.bw_uniform_grain_preflight import (
    BWUniformGrainPreflightError,
    _crop_summary,
    _fixed_crops,
    run_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4z1_bw_uniform_grain_preflight_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_fixed_crops_are_deterministic_and_shape_exact() -> None:
    array = np.arange(100 * 120, dtype=np.uint16).reshape(100, 120)
    centers = [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]
    first = _fixed_crops(array, crop_size=32, centers_yx=centers)
    second = _fixed_crops(array, crop_size=32, centers_yx=centers)
    assert len(first) == 3
    assert all(crop.shape == (32, 32) for crop in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_crop_summary_uses_uint16_full_scale() -> None:
    crop = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)
    summary = _crop_summary(crop)
    assert summary["mean"] == pytest.approx(
        np.mean(crop.astype(np.float64) / 65535.0)
    )
    assert summary["standard_deviation"] > 0.0


def test_real_frozen_preflight_passes_and_repeats() -> None:
    contract = _config()
    report, sheet = run_preflight(root=ROOT, contract=contract)
    repeat, repeat_sheet = run_preflight(root=ROOT, contract=contract)
    assert report == repeat
    assert sheet.tobytes() == repeat_sheet.tobytes()
    assert report["automatic_pass"] is True
    assert report["visual_review_permitted"] is True
    assert report["source_count"] == 3
    assert report["operator_fitting_executed"] is False


def test_preflight_rejects_parent_drift_and_fit_authority() -> None:
    contract = deepcopy(_config())
    contract["parents"]["acquisition_manifest"]["sha256"] = "0" * 64
    with pytest.raises(BWUniformGrainPreflightError, match="parent hash"):
        run_preflight(root=ROOT, contract=contract)
    contract = deepcopy(_config())
    contract["operator_fitting_allowed"] = True
    with pytest.raises(BWUniformGrainPreflightError, match="cannot fit"):
        run_preflight(root=ROOT, contract=contract)
