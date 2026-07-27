from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.dorf_film_response import (
    DorfArchiveError,
    audit_archive,
    inventory,
    parse_curves,
    read_archive,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ab0_dorf_source_audit_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_archive_and_inventory_are_exact() -> None:
    config = _config()
    result = audit_archive(ROOT / config["source"]["path"], config)
    assert result["archive"]["member_bytes"] == 6_595_757
    assert result["inventory"]["record_count"] == 201
    assert result["inventory"]["samples_per_curve"] == [1024]
    assert result["inventory"]["complete_rgb_triplet_count"] == 46
    assert result["inventory"]["complete_triplet_scale_mismatches"] == {}
    assert result["inventory"]["duplicate_source_names"] == {
        "FP2900ZB": 2,
        "FP2900ZG": 2,
        "FP2900ZR": 2,
    }


def test_every_curve_is_monotone_normalized_and_finite() -> None:
    config = _config()
    payload, _ = read_archive(ROOT / config["source"]["path"])
    curves = parse_curves(payload)
    for curve in curves:
        assert np.all(np.isfinite(curve.irradiance))
        assert np.all(np.isfinite(curve.brightness))
        assert np.all(np.diff(curve.irradiance) > 0)
        assert np.all(np.diff(curve.brightness) >= 0)
        assert curve.irradiance[[0, -1]].tolist() == [0.0, 1.0]
        assert curve.brightness[[0, -1]].tolist() == [0.0, 1.0]


def test_source_names_are_not_silently_repaired() -> None:
    config = _config()
    payload, _ = read_archive(ROOT / config["source"]["path"])
    result = inventory(parse_curves(payload))
    assert result["incomplete_rgb_groups"]["Kodachrome-25"] == ["green", "red"]
    assert result["incomplete_rgb_groups"]["Kodachrome-25CD"] == ["blue"]


def test_parser_rejects_bad_record_shape() -> None:
    with pytest.raises(DorfArchiveError, match="six-line"):
        parse_curves(b"name\nscale\nI =\n0\nB =\n")
