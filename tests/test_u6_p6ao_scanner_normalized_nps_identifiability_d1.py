from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval import scanner_normalized_nps_identifiability_d1 as subject

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p6ao_scanner_normalized_nps_identifiability_d1_v1.json"


def test_contract_and_parents_are_frozen() -> None:
    contract = subject.load_contract(CONFIG)
    assert contract["execution"]["posthoc_nps_fit_allowed"] is False
    for binding in contract["parents"].values():
        assert subject.sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_nps_vector_is_deterministic_and_nonzero() -> None:
    values = np.linspace(0.1, 0.9, 32 * 32 * 3, dtype=np.float64).reshape(32, 32, 3)
    edges = np.asarray(subject.load_contract(CONFIG)["execution"]["radial_frequency_edges_cycles_per_pixel"])
    first, energy = subject._nps_vector(values, edges)
    second, _ = subject._nps_vector(values, edges)
    assert first.shape == (24,)
    assert np.array_equal(first, second)
    assert energy > 0.0


def test_contract_rejects_posthoc_fit() -> None:
    contract = subject.load_contract(CONFIG)
    contract["execution"]["posthoc_nps_fit_allowed"] = True
    path = ROOT / "tmp/p6ao-invalid.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(__import__("json").dumps(contract), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="unsupported"):
            subject.load_contract(path)
    finally:
        path.unlink()
