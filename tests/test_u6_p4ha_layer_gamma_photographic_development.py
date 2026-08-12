import json
from pathlib import Path

import numpy as np

from src.eval.layer_gamma_photographic_development import (
    _high_frequency_chroma_p999,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p4ha_contract_keeps_exact_p4gz_mechanism_and_no_rescue():
    contract = load_contract(
        ROOT / "configs/u6_p4ha_layer_gamma_photographic_development_v1.json"
    )
    candidate = contract["candidate"]
    assert candidate["distribution"] == (
        "exact P4GZ independent per-layer support-matched Gamma density"
    )
    assert candidate["amplitude_multiplier"] == 1.0
    assert candidate["cohort_fitting_allowed"] is False
    assert candidate["posthoc_limiting_allowed"] is False
    assert contract["execution"]["same_cohort_rescue_allowed"] is False


def test_p4ha_chroma_gate_is_the_existing_procedural_effect_threshold():
    contract = json.loads(
        (
            ROOT / "configs/u6_p4ha_layer_gamma_photographic_development_v1.json"
        ).read_text()
    )
    assert contract["automatic_gates"]["maximum_high_frequency_chroma_p999"] == 0.004


def test_high_frequency_chroma_metric_ignores_shared_luminance_residual():
    residual = np.zeros((31, 47, 3), dtype=np.float32)
    residual[::2, ::2, :] = 0.01
    assert _high_frequency_chroma_p999(residual) == 0.0
    residual[::2, ::2, 0] += 0.01
    assert _high_frequency_chroma_p999(residual) > 0.004
