from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.bw_process_response_profile import load_contract, run_audit
from src.film_physics.bw_process_response import BWProcessResponseProfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ae_bw_process_response_profile_v1.json"


def test_p2ae_profiles_pass_and_replay_exact() -> None:
    contract = load_contract(CONTRACT)
    assert run_audit(root=ROOT, contract=contract) == run_audit(root=ROOT, contract=contract)
    assert run_audit(root=ROOT, contract=contract)["automatic_pass"] is True


def test_bw_process_profile_rejects_domain_and_tamper() -> None:
    profile = BWProcessResponseProfile("stock", "developer", np.array([1.0, 2.0]), np.array([0.4, 0.6]), "a" * 64, {"equipment":"small_tank","temperature_c":20.0,"densitometry":"diffuse_visual"})
    before = profile.development_time_minutes.copy()
    with pytest.raises(ValueError, match="outside"):
        profile.contrast_index(np.array([0.9]))
    assert np.array_equal(before, profile.development_time_minutes)
    payload = profile.to_dict(); payload["interpolation"] = "cubic"
    with pytest.raises(ValueError, match="unsupported"):
        BWProcessResponseProfile.from_dict(payload)
