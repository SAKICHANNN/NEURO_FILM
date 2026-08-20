from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.eval.repid_reference_operator_predictor import (
    decode_safe_effect,
    encode_effect,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "parameter_lower_bounds": [-0.75] * 9 + [-2.0] * 3,
    "parameter_upper_bounds": [0.75] * 9 + [2.0] * 3,
    "safe_dose_grid": [1.0, 0.75, 0.5, 0.25, 0.0],
    "matrix_gates": {
        "determinant_min": 0.05,
        "condition_number_max": 4.0,
        "minimum_singular_value": 0.2,
    },
}


def test_identity_roundtrip_is_exact_and_safe() -> None:
    operator, clips = decode_safe_effect(np.zeros(12), CONFIG)
    assert clips == 0
    np.testing.assert_array_equal(encode_effect(operator), np.zeros(12))
    assert operator.dose == 1.0


def test_unsafe_prediction_shrinks_on_fixed_grid() -> None:
    parameters = np.zeros(12)
    parameters[:9] = -0.75
    operator, _ = decode_safe_effect(parameters, CONFIG)
    assert operator.dose < 1.0
    assert np.linalg.det(operator.matrix) >= 0.05


def test_contract_binds_implementation_and_runner_has_no_drive_literal() -> None:
    config = json.loads(
        (ROOT / "configs/u5_r2repid9c_reference_operator_predictor_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for binding in config["implementation"].values():
        payload = (ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding["sha256"]
    source = (ROOT / config["implementation"]["runner"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "D:\\" not in source
    assert "P:\\" not in source
    assert source.index("prediction_freeze_sha256 =") < source.index(
        "original = load_original_srgb(", source.index("prediction_freeze_sha256 =")
    )
