from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u6_p6r_kodak_digital_lad_density_primitive import build_report
from src.film_physics.digital_lad import (
    DIGITAL_LAD_AIM_SCHEMA,
    CineonRecordingMode,
    DigitalLadAim,
    aim_from_config,
    code_to_printing_density,
    raw_printing_density_to_code,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p6r_kodak_digital_lad_density_primitive_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_all_codes_formula_roundtrip_monotonicity_and_partition() -> None:
    codes = np.arange(1024, dtype=np.int64)
    for mode, direction in (
        (CineonRecordingMode.NEGATIVE, 1.0),
        (CineonRecordingMode.INTERPOSITIVE, -1.0),
    ):
        result = code_to_printing_density(codes, mode)
        replay = raw_printing_density_to_code(result.raw_printing_density, mode)
        partitioned = np.concatenate(
            [
                code_to_printing_density(codes[:271], mode).raw_printing_density,
                code_to_printing_density(codes[271:], mode).raw_printing_density,
            ]
        )
        assert np.max(np.abs(replay - codes)) <= 1e-12
        assert np.all(direction * np.diff(result.raw_printing_density) > 0.0)
        assert np.array_equal(partitioned, result.raw_printing_density)
        assert np.min(result.physical_nonnegative_printing_density) >= 0.0


def test_interpositive_raw_tail_is_not_silently_clamped() -> None:
    result = code_to_printing_density(np.array([965, 1023]), "interpositive")
    assert np.allclose(result.raw_printing_density, [0.0, -0.116], atol=1e-12)
    assert np.array_equal(result.physical_nonnegative_printing_density, [0.0, 0.0])


def test_lad_aims_keep_status_m_dmin_and_printing_density_separate() -> None:
    for payload in _config()["lad_aims"]:
        aim = aim_from_config(payload)
        assert np.max(
            np.abs(aim.status_m_above_dmin + aim.dmin - aim.status_m_total)
        ) <= 1e-12
        wire = aim.to_dict()
        assert wire["schema"] == DIGITAL_LAD_AIM_SCHEMA
        assert DigitalLadAim.from_dict(wire).to_dict() == wire
        with pytest.raises(ValueError):
            aim.dmin[0] = 0.0


def test_invalid_code_mode_density_and_aim_fail_closed() -> None:
    for bad in (-1, 1024, 1.5, np.nan):
        with pytest.raises(ValueError):
            code_to_printing_density(bad, "negative")
    with pytest.raises(ValueError, match="mode"):
        code_to_printing_density(445, "slide")
    with pytest.raises(ValueError, match="outside"):
        raw_printing_density_to_code(-0.001, "negative")
    bad_aim = dict(_config()["lad_aims"][0])
    bad_aim["status_m_total"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="must equal"):
        aim_from_config(bad_aim)


def test_formal_report_passes_without_rgb_transform() -> None:
    report = build_report(CONFIG)
    assert report["decision"] == "PASS_TYPED_DIGITAL_LAD_DENSITY_PRIMITIVE"
    assert report["all_gates_passed"]
    assert report["metrics"]["code_count"] == 1024
    assert report["metrics"]["rgb_image_transform_count"] == 0
    assert report["metrics"]["recommended_lad_code"] == 445
