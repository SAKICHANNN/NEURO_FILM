from pathlib import Path

import numpy as np
import pytest

from src.eval.hybrid_density_amplitude_thomas import load_contract, run_audit
from src.film_physics.bw_hybrid_density_amplitude import (
    BWHybridDensityAmplitudeProfile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ar_hybrid_density_amplitude_thomas_v1.json"


def test_p2ar_hybrid_field_replays() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["gate_results"]["minimum_developed_density"] is False
    assert first["measurements"]["minimum_developed_density"] < 0.0
    assert first["measurements"]["repeat_byte_exact"] is True
    assert first["measurements"]["partition_exact"] is True


def test_p2ar_profile_rejects_expanded_density_domain() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    profile = BWHybridDensityAmplitudeProfile.from_dict(report["profile"])
    with pytest.raises(ValueError):
        profile.sigma_d(0.0)
    with pytest.raises(ValueError):
        profile.sigma_d(2.0)
    assert np.array_equal(
        profile.sigma_d(np.asarray(profile.densities)),
        np.asarray(profile.density_rms),
    )
