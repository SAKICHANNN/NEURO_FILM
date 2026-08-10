from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.scanner_log_oecf_profile import compile_profile, evaluate, load_contract
from src.film_physics.scanner_oecf import LogScannerOecfProfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6al_scanner_log_oecf_profile_v1.json"


def test_profile_exhaustive_verification_passes() -> None:
    report = evaluate(load_contract(CONTRACT, ROOT))
    assert report["status"] == "pass-exact-workflow-typed-log-oecf-profile"
    assert report["verification"]["integer_codes_preflighted"] == 65536
    assert all(report["verification"]["checks"].values())


def test_profile_exact_serialization_roundtrip() -> None:
    profile = compile_profile(load_contract(CONTRACT, ROOT))
    restored = LogScannerOecfProfile.from_dict(json.loads(profile.canonical_bytes()))
    assert restored == profile
    assert restored.profile_sha256 == profile.profile_sha256


def test_profile_known_density_roundtrip() -> None:
    profile = compile_profile(load_contract(CONTRACT, ROOT))
    density = np.asarray([0.06, 0.5, 1.5, 2.5, 3.08], dtype=np.float64)
    recovered = profile.code_to_density(profile.density_to_code(density))
    assert np.max(np.abs(recovered - density)) <= 1e-12


@pytest.mark.parametrize("code", [float("nan"), float("inf"), -1.0, 0.0, 2341.0])
def test_profile_rejects_invalid_code(code: float) -> None:
    profile = compile_profile(load_contract(CONTRACT, ROOT))
    with pytest.raises(ValueError):
        profile.code_to_density(code)


@pytest.mark.parametrize("density", [float("nan"), float("inf"), -0.1, 3.09])
def test_profile_rejects_uncalibrated_density(density: float) -> None:
    profile = compile_profile(load_contract(CONTRACT, ROOT))
    with pytest.raises(ValueError):
        profile.density_to_code(density)


def test_profile_rejects_claim_escalation() -> None:
    profile = compile_profile(load_contract(CONTRACT, ROOT)).to_dict()
    profile["production_eligible"] = True
    with pytest.raises(ValueError):
        LogScannerOecfProfile.from_dict(profile)


def test_contract_rejects_parent_evidence_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["parent"]["evidence_sha256"] = "0" * 64
    drifted = tmp_path / "contract.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="evidence binding mismatch"):
        load_contract(drifted, ROOT)
