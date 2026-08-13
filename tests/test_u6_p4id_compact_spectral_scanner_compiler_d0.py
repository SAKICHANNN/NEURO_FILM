from pathlib import Path

import numpy as np
import pytest

from src.eval.compact_spectral_scanner_compiler_d0 import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4id_compact_spectral_scanner_compiler_d0_v1.json"


def test_formal_contract_is_deterministic_and_discriminating() -> None:
    config = load_contract(CONFIG)
    first = evaluate(config, ROOT)
    second = evaluate(config, ROOT)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["models"]["linear-density-affine"]["p95_absolute_error"] > 0.05
    assert first["models"]["beer-lambert-log-response-affine"]["p95_absolute_error"] <= 0.05


def test_parent_hash_drift_fails_closed() -> None:
    config = load_contract(CONFIG)
    config["parents"]["curve_data"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        evaluate(config, ROOT)


def test_nonfinite_contract_population_fails_closed() -> None:
    config = load_contract(CONFIG)
    config["population"]["fit_dye_amounts"][2] = np.nan
    with pytest.raises(ValueError):
        evaluate(config, ROOT)


def test_neutral_control_is_uniform_spectral_density_not_equal_dyes() -> None:
    report = evaluate(load_contract(CONFIG), ROOT)
    assert report["neutral_channel_spread_max"] <= 1e-12
    assert report["spectral_truth_maximum_positive_step"] <= 1e-12
