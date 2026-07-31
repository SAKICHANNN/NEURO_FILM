from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from skimage.color import rgb2lab

from src.roll2film.factorized_ao6_perceptual_residual import (
    FactorizedAO6PerceptualResidual,
)

ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT
    / "configs/u5_r2bk16_factorized_ao6_perceptual_residual_decision_v1.json"
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260731)
    source = rng.random((19, 17, 3), dtype=np.float32)
    source[0, 0] = 0.0
    source[-1, -1] = 1.0
    base = np.clip(0.04 + 0.9 * source, 0.0, 1.0).astype(np.float32)
    candidate = np.stack(
        (
            np.sqrt(source[..., 0]),
            np.power(source[..., 1], 1.4),
            np.clip(source[..., 2] * 1.25, 0.0, 1.0),
        ),
        axis=-1,
    ).astype(np.float32)
    return source, base, candidate


def test_factorized_ao6_residual_is_exact_bounded_and_partition_stable() -> None:
    operator = FactorizedAO6PerceptualResidual(row_chunk=7)
    source, base, candidate = _inputs()
    output = operator.apply(source, base, candidate)
    delta = np.linalg.norm(rgb2lab(output) - rgb2lab(base), axis=-1)
    assert float(np.max(delta)) <= (
        operator.maximum_final_delta_e76 + operator.perceptual_tolerance
    )
    assert np.array_equal(
        operator.apply(source, base, candidate, strength=0.0),
        base,
    )
    assert np.array_equal(output[0, 0], source[0, 0])
    assert np.array_equal(output[-1, -1], source[-1, -1])
    partitioned = np.concatenate(
        (
            operator.apply(source[:8], base[:8], candidate[:8]),
            operator.apply(source[8:], base[8:], candidate[8:]),
        )
    )
    assert np.array_equal(partitioned, output)
    assert np.all(np.isfinite(output))
    assert np.all((output >= 0.0) & (output <= 1.0))


def test_factorized_ao6_residual_soft_bounds_components() -> None:
    operator = FactorizedAO6PerceptualResidual()
    source, base, candidate = _inputs()
    _, residual = operator._factorized_residual(source, base, candidate)
    assert float(np.max(np.abs(residual[..., 0]))) <= (
        operator.lightness_soft_cap
    )
    assert float(np.max(np.linalg.norm(residual[..., 1:], axis=-1))) <= (
        operator.chroma_soft_cap
    )


def test_factorized_ao6_residual_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError):
        FactorizedAO6PerceptualResidual(maximum_final_delta_e76=0.0)
    operator = FactorizedAO6PerceptualResidual()
    source = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        operator.apply(source, source[..., :2], source)
    with pytest.raises(ValueError):
        operator.apply(source, source, source, strength=1.01)


def test_bk16_decision_opens_only_sixth_fresh_source_preflight() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    result = decision["result"]
    assert decision["status"] == "primitive_and_known_failure_regression_pass"
    assert decision["decision"] == (
        "retain_fixed_bk16_open_sixth_fresh_source_preflight"
    )
    assert result["automatic_pass"]
    assert result["repeat_file_count"] == 6
    assert result["repeat_file_differences"] == 0
    assert result["confirmed_severe_artifact_count"] == 0
    assert not result["known_phaseone_failure_reproduced"]
    assert result["cube_new_uint16_boundary_fraction"] == 0.0
    assert result["neutral_ramp_minimum_lightness_step"] > 0.0
    assert decision["evidence"]["repeat_report_sha256_exact"]
    assert decision["evidence"]["repeat_visual_sheet_sha256_exact"]
    assert decision["evidence"]["repeat_output_sha256_exact"]
    assert "sixth leakage-clean" in decision["next_leaf"]
    assert not decision["training_allowed"]
    assert not decision["operator_fitting_allowed"]
    assert not decision["selector_training_allowed"]
    assert not decision["strength_retuning_allowed"]
    assert not decision["production_default_changed"]
