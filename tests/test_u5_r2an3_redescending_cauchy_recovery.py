from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2an3_redescending_cauchy_paired_recovery import (
    CONFIG_SHA256,
    load_config,
)
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2an3_redescending_cauchy_paired_recovery_v1.json"


def test_frozen_contract_has_one_candidate_and_no_sweep() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["fit"]["candidate_loss"] == "cauchy"
    assert config["fit"]["candidate_loss_scale"] == 0.005
    assert config["fit"]["loss_or_scale_sweep_allowed"] is False
    assert config["proxy_pair_allowed"] is False
    assert config["photographic_render_allowed"] is False


def test_existing_fitter_accepts_fixed_cauchy_and_preserves_inputs() -> None:
    source = np.linspace(0.01, 0.95, 36, dtype=np.float64).reshape(12, 3)
    target = source.copy()
    source_before = source.copy()
    target_before = target.copy()
    result = fit_positive_film_response_operator(
        source,
        target,
        model="two_matrix",
        loss="cauchy",
        loss_scale=0.005,
        restart_count=1,
        maximum_function_evaluations=50,
    )
    assert result.loss == "cauchy"
    assert result.loss_scale == 0.005
    assert np.array_equal(source, source_before)
    assert np.array_equal(target, target_before)


def test_contract_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["fit"]["candidate_loss_scale"] = 0.01
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(tampered, expected_sha256=CONFIG_SHA256)
