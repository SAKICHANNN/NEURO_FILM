from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ao6_procedural_filmfx_value import (
    AO6FilmFXValueError,
    _effect_metrics,
    evaluate_report,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u5_r2bc0_ao6_procedural_filmfx_value_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_contract_binds_parent_population_and_parameters() -> None:
    rows = validate_contract(ROOT, _config())
    assert len(rows) == 16
    assert rows[0]["id"] == "canon_eos_1d_mark_iv"
    assert rows[-1]["id"] == "sony_nex_3n"


def test_contract_rejects_parameter_drift() -> None:
    config = _config()
    config["arms"][1]["grain_strength"] = 0.013
    with pytest.raises(AO6FilmFXValueError, match="parameter drift"):
        validate_contract(ROOT, config)


def test_effect_metrics_detect_new_boundary_and_colour_speckle() -> None:
    base = np.full((8, 8, 3), 0.5, dtype=np.float32)
    output = base.copy()
    output[3, 3] = [1.0, 0.5, 0.5]
    metrics = _effect_metrics(base, output)
    assert metrics["new_raw_clipping_fraction"] > 0.0
    assert metrics["high_frequency_chroma_p999"] > 0.0


def test_evaluator_requires_complete_arms() -> None:
    with pytest.raises(AO6FilmFXValueError, match="incomplete"):
        evaluate_report(_config(), {"records": [], "halation_supported_rows": 0})
