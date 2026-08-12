import json
from pathlib import Path

import numpy as np

from src.eval.cross_layer_thomas_gamma_photographic_development import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_p4hd_contract_reuses_exact_p4hc_correlation_and_photographic_gates():
    contract = load_contract(
        ROOT
        / "configs/u6_p4hd_cross_layer_thomas_gamma_photographic_development_v1.json"
    )
    p4hc = json.loads(
        (ROOT / "configs/u6_p4hc_cross_layer_thomas_gamma_copula_v1.json").read_text()
    )
    candidate = contract["candidate"]
    assert candidate["mechanism"] == (
        "exact P4HC fixed cross-layer Thomas/Gamma copula"
    )
    assert np.array_equal(
        np.asarray(candidate["correlation_matrix"]),
        np.asarray(p4hc["candidate"]["correlation_matrix"]),
    )
    assert candidate["amplitude_multiplier"] == 1.0
    assert candidate["cohort_fitting_allowed"] is False
    assert contract["automatic_gates"]["maximum_high_frequency_chroma_p999"] == 0.004
    assert contract["execution"]["same_cohort_rescue_allowed"] is False
