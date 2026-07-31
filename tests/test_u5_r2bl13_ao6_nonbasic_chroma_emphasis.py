from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ao6_nonbasic_chroma_emphasis import (
    compose_ao6_nonbasic_chroma_emphasis,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bl13_ao6_nonbasic_chroma_emphasis_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _compose(source: np.ndarray, ao6: np.ndarray) -> dict[str, np.ndarray]:
    spec = _config()["candidate"]
    return compose_ao6_nonbasic_chroma_emphasis(
        source,
        ao6,
        fit_pixel_budget=int(spec["fit_pixel_budget"]),
        chroma_residual_strength=float(spec["chroma_residual_strength"]),
        hard_boundary_epsilon_encoded_srgb=float(
            spec["hard_boundary_epsilon_encoded_srgb"]
        ),
        guard_boundary_epsilon_encoded_srgb=float(
            spec["guard_boundary_epsilon_encoded_srgb"]
        ),
    )


def test_contract_binds_exact_bl8_evidence() -> None:
    validated = validate_contract(ROOT, _config())
    assert len(validated["eligible_ids"]) == 17


def test_neutral_basic_look_has_no_spurious_residual() -> None:
    axis = np.linspace(0.1, 0.9, 16)
    source = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    source = np.concatenate((source, source[..., :1]), axis=-1)
    ao6 = np.clip(source * 0.9 + 0.03, 0.0, 1.0)
    result = _compose(source, ao6)
    assert np.max(np.abs(result["output"] - ao6)) < 2e-3
    assert np.all(np.isfinite(result["output"]))


def test_nonbasic_colour_residual_changes_output_without_clipping() -> None:
    rng = np.random.default_rng(2026080101)
    source = rng.uniform(0.08, 0.92, size=(24, 24, 3))
    ao6 = source.copy()
    ao6[..., 0] = np.clip(ao6[..., 0] + 0.12 * (ao6[..., 2] - 0.5), 0.02, 0.98)
    result = _compose(source, ao6)
    assert np.mean(np.abs(result["output"] - ao6)) > 1e-4
    assert np.all((result["output"] >= 0.0) & (result["output"] <= 1.0))


@pytest.mark.parametrize(
    "source,ao6",
    [
        (np.zeros((8, 8)), np.zeros((8, 8))),
        (np.zeros((8, 8, 3)), np.zeros((9, 8, 3))),
        (np.full((8, 8, 3), np.nan), np.zeros((8, 8, 3))),
        (np.full((8, 8, 3), 1.1), np.zeros((8, 8, 3))),
    ],
)
def test_invalid_inputs_fail_closed(source: np.ndarray, ao6: np.ndarray) -> None:
    with pytest.raises(ValueError, match="invalid AO6"):
        _compose(source, ao6)
