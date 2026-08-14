from pathlib import Path

import numpy as np
import pytest

from src.eval.historical_wiener_amplitude_compiler import load_contract, run_audit
from src.film_physics.bw_wiener_amplitude_compiler import (
    BWWienerAmplitudeCompiler,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2aq_historical_wiener_amplitude_compiler_v1.json"


def test_p2aq_compiler_replays_and_rejects_extrapolation() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["measurements"]["reference_scale"] == 1.0
    profile = BWWienerAmplitudeCompiler.from_dict(first["profile"])
    with pytest.raises(ValueError):
        profile.relative_standard_deviation(0.0)
    with pytest.raises(ValueError):
        profile.relative_standard_deviation(2.0)


def test_p2aq_node_scales_follow_source_wiener_amplitudes() -> None:
    report = run_audit(root=ROOT, contract=load_contract(CONTRACT))
    profile = BWWienerAmplitudeCompiler.from_dict(report["profile"])
    nodes = np.asarray(profile.densities)
    actual = profile.relative_standard_deviation(nodes)
    source = np.asarray(profile.wiener_granularity_spectrum_cm2)
    assert np.allclose(actual, np.sqrt(source / source[0]), rtol=0.0, atol=1e-15)
