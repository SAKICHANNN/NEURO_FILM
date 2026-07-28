from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.eval.global_frontier import new_hard_clipping_fraction
from src.roll2film.factorized_boundary_guard import (
    apply_factorized_boundary_guard,
)
from src.roll2film.positive_film import positive_film_operator_from_config
from scripts.run_u5_r2ao3_factorized_chart_boundary_frontier import (
    CONFIG_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_CONFIG = (
    ROOT / "configs/u5_r2ao2_velvia_chart_one_matrix_operator_v1.json"
)
AO3_CONFIG = (
    ROOT / "configs/u5_r2ao3_factorized_chart_boundary_frontier_v1.json"
)


def _operator_and_controls():
    operator_payload = json.loads(OPERATOR_CONFIG.read_text(encoding="utf-8"))
    config = json.loads(AO3_CONFIG.read_text(encoding="utf-8"))
    operator = positive_film_operator_from_config(
        operator_payload["witnesses"][config["operator_witness_id"]],
        exposure_floor=float(operator_payload["exposure_floor"]),
        matrix_minimum_determinant=float(
            operator_payload["parameter_bounds"]["matrix_minimum_determinant"]
        ),
        minimum_endpoint_span=float(
            operator_payload["parameter_bounds"]["minimum_endpoint_span"]
        ),
    )
    return operator, config["factorization"]


def test_ao3_frozen_contract_hash_and_factor_bank() -> None:
    raw = AO3_CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256
    assert len(config["factor_bank"]) == config["candidate_count"] == 4
    assert not config["factorization"]["hard_clip_allowed"]
    assert not config["operator_refit_allowed"]
    assert not config["production_integration_allowed"]


def test_factorized_guard_is_repeat_exact_and_stays_in_cube() -> None:
    operator, controls = _operator_and_controls()
    source = np.random.default_rng(20260728).random((1024, 3))
    arguments = {
        "tone_strength": 0.35,
        "chroma_strength": 1.0,
        "luma_weights": np.asarray(controls["luma_weights"]),
        "hard_boundary_epsilon_encoded_srgb": controls[
            "hard_boundary_epsilon_encoded_srgb"
        ],
        "guard_boundary_epsilon_encoded_srgb": controls[
            "guard_boundary_epsilon_encoded_srgb"
        ],
    }
    first = apply_factorized_boundary_guard(operator, source, **arguments)
    second = apply_factorized_boundary_guard(operator, source, **arguments)
    np.testing.assert_array_equal(first.output, second.output)
    np.testing.assert_array_equal(first.tone_scale, second.tone_scale)
    np.testing.assert_array_equal(first.chroma_scale, second.chroma_scale)
    assert float(np.min(first.output)) >= 0.0
    assert float(np.max(first.output)) <= 1.0


def test_factorized_guard_creates_no_new_encoded_hard_boundary() -> None:
    operator, controls = _operator_and_controls()
    encoded = np.random.default_rng(41).random((64, 64, 3))
    linear = np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )
    result = apply_factorized_boundary_guard(
        operator,
        linear,
        tone_strength=0.5,
        chroma_strength=1.0,
        luma_weights=np.asarray(controls["luma_weights"]),
        hard_boundary_epsilon_encoded_srgb=controls[
            "hard_boundary_epsilon_encoded_srgb"
        ],
        guard_boundary_epsilon_encoded_srgb=controls[
            "guard_boundary_epsilon_encoded_srgb"
        ],
    )
    output_encoded = linear_srgb_to_encoded(result.output)
    assert (
        new_hard_clipping_fraction(
            encoded,
            output_encoded,
            controls["hard_boundary_epsilon_encoded_srgb"],
        )
        == 0.0
    )


def test_factorized_guard_rejects_invalid_strengths() -> None:
    operator, controls = _operator_and_controls()
    with pytest.raises(ValueError, match="invalid"):
        apply_factorized_boundary_guard(
            operator,
            np.full((2, 3), 0.5),
            tone_strength=1.1,
            chroma_strength=1.0,
            luma_weights=np.asarray(controls["luma_weights"]),
            hard_boundary_epsilon_encoded_srgb=controls[
                "hard_boundary_epsilon_encoded_srgb"
            ],
            guard_boundary_epsilon_encoded_srgb=controls[
                "guard_boundary_epsilon_encoded_srgb"
            ],
        )
