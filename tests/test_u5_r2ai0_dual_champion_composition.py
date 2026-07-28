from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.dual_champion_composition import (
    CompositionFrontierError,
    candidate_bank,
    compose_rgb,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ai0_dual_champion_global_composition_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_contract_and_bank_validate() -> None:
    config = _config()
    validated = validate_contract(ROOT, config)
    bank = candidate_bank(config)
    assert len(bank) == 6
    assert len(validated["samples"]) == 41
    assert {row["order"] for row in bank} == {
        "anchor_then_density",
        "density_then_anchor",
    }


def test_composition_uses_order_and_one_final_margin() -> None:
    source = np.linspace(0.0, 1.0, 3 * 5 * 7, dtype=np.float32).reshape(
        5, 7, 3
    )

    def anchor(value: np.ndarray) -> np.ndarray:
        return np.asarray(value) * 0.8 + 0.1

    def density(value: np.ndarray, strength: float) -> np.ndarray:
        return np.power(np.asarray(value), 1.0 + strength)

    first = compose_rgb(
        source,
        order="anchor_then_density",
        density_strength=0.5,
        apply_anchor=anchor,
        apply_density=density,
        output_margin=4,
    )
    second = compose_rgb(
        source,
        order="density_then_anchor",
        density_strength=0.5,
        apply_anchor=anchor,
        apply_density=density,
        output_margin=4,
    )
    assert first.dtype == np.float32
    assert float(first.min()) >= 4.0 / 255.0
    assert float(first.max()) <= 251.0 / 255.0
    assert not np.array_equal(first, second)


def test_composition_rejects_invalid_input_or_order() -> None:
    source = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(CompositionFrontierError, match="unsupported"):
        compose_rgb(
            source,
            order="unknown",
            density_strength=0.5,
            apply_anchor=lambda value: value,
            apply_density=lambda value, strength: value,
            output_margin=4,
        )
    source[0, 0, 0] = np.nan
    with pytest.raises(CompositionFrontierError, match="finite"):
        compose_rgb(
            source,
            order="anchor_then_density",
            density_strength=0.5,
            apply_anchor=lambda value: value,
            apply_density=lambda value, strength: value,
            output_margin=4,
        )


def test_contract_rejects_candidate_count_drift() -> None:
    config = _config()
    config["candidate_bank"]["candidate_count"] = 7
    with pytest.raises(CompositionFrontierError, match="count"):
        candidate_bank(config)
