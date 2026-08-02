from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.eval.applause_effective_sfr_temporal import (
    ApplauseEffectiveSFRError,
    measure_edge_from_array,
    validate_config,
    validate_development_lock,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/u6_p6ai_applause_effective_sfr_temporal_v1.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _synthetic_edge(sigma: float, *, reverse_rows: bool = False) -> np.ndarray:
    rows = int(CONFIG["measurement"]["row_stop"])
    columns = 128
    y, x = np.mgrid[:rows, :columns]
    distance = x - (64.0 + 0.015 * (y - (rows - 1) / 2.0))
    values = np.asarray(
        [
            0.5 * (1.0 + math.erf(float(value) / (math.sqrt(2.0) * sigma)))
            for value in distance.ravel()
        ],
        dtype=np.float64,
    ).reshape(distance.shape)
    if reverse_rows:
        values = values[::-1].copy()
    return values * 50000.0


def test_contract_binds_exact_parents_and_keeps_claim_narrow() -> None:
    validate_config(CONFIG, ROOT)
    assert CONFIG["training_allowed"] is False
    assert CONFIG["product_integration_allowed"] is False
    assert "not absolute scanner MTF" in CONFIG["claim_ceiling"]


@pytest.mark.parametrize("sigma", CONFIG["controls"]["synthetic_gaussian_sigma_px"])
def test_synthetic_gaussian_edge_recovers_curve_and_mtf50(sigma: float) -> None:
    result = measure_edge_from_array(
        _synthetic_edge(float(sigma)),
        nominal_edge_x=64,
        measurement=CONFIG["measurement"],
        raw_code_span=50000.0,
    )
    frequencies = np.asarray(
        CONFIG["measurement"]["curve_frequency_samples_cycles_per_pixel"],
        dtype=np.float64,
    )
    expected = np.exp(-2.0 * math.pi**2 * float(sigma) ** 2 * frequencies**2)
    assert np.max(np.abs(np.asarray(result["sfr_curve"]) - expected)) <= float(
        CONFIG["controls"]["maximum_synthetic_curve_abs_error"]
    )
    expected_mtf50 = math.sqrt(math.log(2.0)) / (
        math.sqrt(2.0) * math.pi * float(sigma)
    )
    assert abs(result["mtf50_cycles_per_pixel"] - expected_mtf50) <= float(
        CONFIG["controls"]["maximum_synthetic_mtf50_abs_error_cycles_per_pixel"]
    )


def test_row_order_control_preserves_effective_sfr() -> None:
    forward = measure_edge_from_array(
        _synthetic_edge(1.25),
        nominal_edge_x=64,
        measurement=CONFIG["measurement"],
        raw_code_span=50000.0,
    )
    reversed_rows = measure_edge_from_array(
        _synthetic_edge(1.25, reverse_rows=True),
        nominal_edge_x=64,
        measurement=CONFIG["measurement"],
        raw_code_span=50000.0,
    )
    assert np.max(
        np.abs(
            np.asarray(forward["sfr_curve"]) - np.asarray(reversed_rows["sfr_curve"])
        )
    ) <= float(CONFIG["controls"]["maximum_row_order_curve_abs_error"])


def test_contract_rejects_parent_and_role_drift() -> None:
    drifted = deepcopy(CONFIG)
    drifted["parents"]["p6k_contract"]["sha256"] = "0" * 64
    with pytest.raises(ApplauseEffectiveSFRError, match="parent identity"):
        validate_config(drifted, ROOT)
    drifted = deepcopy(CONFIG)
    drifted["source"]["confirmation_early"] = drifted["source"]["confirmation_early"][
        :-1
    ]
    with pytest.raises(ApplauseEffectiveSFRError, match="role support"):
        validate_config(drifted, ROOT)


def test_development_lock_rejects_tamper_before_confirmation() -> None:
    contract_sha = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    body = {
        "schema": "neuro_film.u6_p6ai_applause_effective_sfr_development_lock.v1",
        "experiment_id": CONFIG["experiment_id"],
        "contract_sha256": contract_sha,
        "frequency_samples_cycles_per_pixel": CONFIG["measurement"][
            "curve_frequency_samples_cycles_per_pixel"
        ],
        "selected_edge_indices": list(range(8)),
        "minimum_required_shared_edges": 8,
        "development_gate_pass": True,
        "development_scans": [],
        "prototypes": {},
        "algorithm": "test",
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    lock = {**body, "development_lock_id": hashlib.sha256(raw).hexdigest()}
    validate_development_lock(lock, CONFIG, contract_sha256=contract_sha)
    lock["selected_edge_indices"][0] = 12
    with pytest.raises(ApplauseEffectiveSFRError, match="lock hash"):
        validate_development_lock(lock, CONFIG, contract_sha256=contract_sha)
