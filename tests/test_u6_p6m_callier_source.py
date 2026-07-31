from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from src.eval.physical_callier_source import (
    CallierSourceError,
    audit_source,
    evaluate_observations,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6m_callier_source_v1.json"


def _observation_fixture() -> str:
    return """
    Q = OD directed / OD diffuse as a function of OD diffuse.
    A completely transparent point is by definition equal to 1.
    Modern chromogenic monopack images are dyes that scarcely scatter light;
    Q is close to 1. The density reaches 2.2 and interpolation uses range [0, 2.4].
    31 images between 400 and 700 nm use a spectral step of 10 nm, and the
    effect is stronger for short wavelengths. The maximum Callier effect is
    between 1.3 and 1.5 OD diffuse. ODdye appears in [(ODfilm - ODdye) × Q]
    + ODdye. Residual discrepancies depend on different silver particles.
    """


def test_contract_is_bounded_and_fit_forbidden() -> None:
    contract = load_contract(CONTRACT)
    assert contract["source"]["expected_bytes"] == 6_128_203
    assert contract["source"]["expected_pages"] == 4
    assert contract["source_gate"]["no_numeric_curve_digitization"]
    assert contract["source_gate"]["no_parameter_fit"]


def test_observation_gate_requires_every_mechanism() -> None:
    observations = evaluate_observations(_observation_fixture())
    assert observations and all(observations.values())
    missing = evaluate_observations(
        _observation_fixture().replace("stronger for short wavelengths", "")
    )
    assert not missing["spectral_support"]


def test_contract_rejects_parameter_fit_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["source_gate"]["no_parameter_fit"] = False
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CallierSourceError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not (ROOT / "data/physical_scanner/callier_cic2017_v1/CIC_2017_art00031_Giorgio-Trumpy.pdf").is_file()
    or shutil.which("pdftotext.exe") is None
    or shutil.which("pdfinfo.exe") is None,
    reason="frozen P6M source or PDF tools unavailable",
)
def test_real_source_audit_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = audit_source(contract, ROOT)
    second = audit_source(contract, ROOT)
    assert first == second
    assert first["source_pass"]
    assert first["decision"] == "open_generic_bounded_callier_operator"
    assert first["source_sha256"] == (
        "c9513ecbad6704c482b87217374bf247cb165a6dcebbc15f967f011098f49223"
    )
