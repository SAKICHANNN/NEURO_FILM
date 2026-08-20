from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.real_film.ppisp_capture_pair_source_lock import ZipMember
from src.real_film.rgb2raw_metadata_explicit_isp import (
    RGB2RAWMetadataISPError,
    build_role_rows,
    decode_metadata,
    evaluate_gates,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/sf3_a0t_rgb2raw_metadata_explicit_isp_d0_v1.json").read_text(
        encoding="utf-8"
    )
)


def _member(name: str) -> ZipMember:
    return ZipMember(name, 0, 8, 0, 1, 1, 0, 0, 0)


def test_role_builder_is_group_disjoint_and_exact() -> None:
    members = []
    for camera in ("iphone-x", "samsung-s9"):
        for group in range(70):
            members.extend(
                (
                    _member(f"train/{camera}/{group}_0.png"),
                    _member(f"train/{camera}/{group}_0.npy"),
                    _member(f"train/{camera}/{group}.pkl"),
                )
            )
    rows = build_role_rows(members, CONFIG)
    assert len(rows) == 128
    for camera in ("iphone-x", "samsung-s9"):
        camera_rows = [row for row in rows if row.camera_group == camera]
        assert sum(row.role == "fit" for row in camera_rows) == 40
        assert sum(row.role == "calibration" for row in camera_rows) == 12
        assert sum(row.role == "sealed_confirmation" for row in camera_rows) == 12
        assert len({row.canonical_capture_group for row in camera_rows}) == 64


def _metadata_payload() -> bytes:
    value = {
        "white_level": 4095,
        "black_levels": [0, 0, 0, 0],
        "camera_whitebalance": [2.0, 1.0, 1.5, 0.0],
        "daylight_whitebalance": [2.2, 1.0, 1.4, 0.0],
        "color_matrix": np.eye(3, 4, dtype=np.float32),
        "cfa_pattern": np.asarray([[0, 1], [3, 2]], dtype=np.uint8),
        "sizes": {"width": 4032, "height": 3024},
        "tone_curve": np.arange(65536, dtype=np.uint16),
    }
    return pickle.dumps(value, protocol=4)


def test_restricted_metadata_decoder_accepts_exact_numpy_schema() -> None:
    value = decode_metadata(_metadata_payload(), CONFIG)
    assert value["white_level"] == 4095
    assert value["color_matrix"].shape == (3, 4)
    assert value["tone_curve"].dtype == np.uint16


def test_restricted_metadata_decoder_rejects_arbitrary_global() -> None:
    payload = pickle.dumps(Path("forbidden"), protocol=4)
    with pytest.raises(RGB2RAWMetadataISPError, match="restricted decode"):
        decode_metadata(payload, CONFIG)


def test_gate_evaluator_fails_scientific_tail_independently() -> None:
    rows = []
    for camera in ("iphone-x", "samsung-s9"):
        for index in range(12):
            rows.append(
                {
                    "camera_group": camera,
                    "errors": {"capture_wb": 0.05, "static": 0.06},
                    "candidate_gradient_p999_ratio": 1.0,
                    "candidate_new_exact_boundary_fraction": 0.0,
                }
            )
    score = {
        "summary": {
            "row_count": 24,
            "improvement_rate_vs_strongest_legitimate": 1.0,
            "median_relative_error_reduction_vs_strongest_legitimate": 0.2,
            "worst_relative_error_reduction_vs_strongest_legitimate": -0.2,
            "each_camera_improvement_rate_vs_strongest_legitimate": {
                "iphone-x": 1.0,
                "samsung-s9": 1.0,
            },
            "improvement_rate_vs_permuted_metadata": 1.0,
            "median_relative_error_reduction_vs_permuted_metadata": 0.2,
            "median_mean_oklab_error": 0.05,
            "p95_mean_oklab_error": 0.06,
            "maximum_gradient_p999_ratio": 1.0,
            "maximum_new_exact_boundary_fraction": 0.0,
            "maximum_fresh_reload_error": 0.0,
        },
        "rows": rows,
    }

    class Operator:
        matrix = np.eye(3)
        bias = np.zeros(3)
        dose = 1.0

    gates = evaluate_gates(
        score,
        {"capture_wb": {"iphone-x": Operator(), "samsung-s9": Operator()}},
        CONFIG["calibration_gates"],
    )
    assert gates["worst_tail_vs_strongest_legitimate"] is False
    assert gates["new_exact_boundary"] is True
