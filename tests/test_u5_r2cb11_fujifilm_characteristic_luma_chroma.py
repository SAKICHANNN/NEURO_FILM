from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_luma_chroma import (
    CONTRACT_SHA256,
    FujifilmCharacteristicLumaChromaError,
    apply_characteristic_luma_chroma,
    load_contract,
)
from src.eval.fujifilm_characteristic_photographic import _compiled_curve
from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _new_boundary_fraction

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb11_fujifilm_characteristic_luma_chroma_v1.json"
DECISION = ROOT / "configs/u5_r2cb11_fujifilm_characteristic_luma_chroma_decision_v1.json"


def _curve():
    c = load_contract(CONFIG)
    return _compiled_curve(load_cb6(ROOT / c["parents"]["cb6_contract_path"]))


def test_contract_is_frozen():
    assert hash_file(CONFIG) == CONTRACT_SHA256


def test_luminance_is_exact_and_boundaries_are_preserved():
    rng = np.random.default_rng(11)
    source = rng.random((191, 193, 3), dtype=np.float32)
    w = np.array([0.2126, 0.7152, 0.0722])
    e = 1 / 510
    output, scale, error = apply_characteristic_luma_chroma(
        source, _curve(), weights=w, strength=0.2, boundary_epsilon=e
    )
    assert np.max(np.abs(error)) < 1e-6
    assert _new_boundary_fraction(source, output, e) == 0
    assert scale.min() >= 0 and scale.max() <= 1


def test_invalid_source_fails():
    with pytest.raises(FujifilmCharacteristicLumaChromaError, match="invalid"):
        apply_characteristic_luma_chroma(
            np.array([[[-0.1, 0, 0]]], dtype=np.float32),
            _curve(),
            weights=np.array([0.2126, 0.7152, 0.0722]),
            strength=0.2,
            boundary_epsilon=1 / 510,
        )


def test_formal_decision_retains_only_the_mechanism():
    import json

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["decision"] == "retain_exact_luminance_bounded_chroma_mechanism"
    assert decision["repeat_report_byte_exact"] is True
    assert decision["automatic_pass"] is True
    assert "not Velvia stock response" in decision["claim_ceiling"]
