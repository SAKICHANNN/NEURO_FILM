from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.bw_structure_photographic_value import (
    BWStructurePhotographicValueError,
    _isolated_excursions,
    _load_manifest,
    _mean_density,
    _new_boundary_fraction,
    _scanner_profile,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2bb_bw_structure_photographic_value_v1.json"


def test_p2bb_contract_and_bound_source_manifest() -> None:
    contract = load_contract(CONTRACT)
    rows = _load_manifest(ROOT, contract["parents"]["source_manifest"])
    assert len(rows) == 11
    assert len({row["make"] for row in rows}) == 11


def test_p2bb_mean_density_mapping_is_global_and_bounded() -> None:
    contract = load_contract(CONTRACT)
    encoded = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float64)
    density = _mean_density(encoded, contract["mean_density_mapping"])
    assert np.array_equal(density, np.array([[0.06, 1.14]], dtype=np.float64))


def test_p2bb_scanner_is_neutral_mtf_only() -> None:
    profile = _scanner_profile(load_contract(CONTRACT))
    assert profile.scanner_mtf_sigma_um_rgb == pytest.approx((4.445, 4.445, 4.445))
    assert profile.forward_scatter_sigma_um_rgb == (0.0, 0.0, 0.0)
    assert profile.development_adjacency_gain_rgb == (0.0, 0.0, 0.0)


def test_p2bb_new_boundary_and_isolated_metrics() -> None:
    reference = np.full((7, 7, 3), 0.5, dtype=np.float64)
    candidate = reference.copy()
    candidate[3, 3] = 1.0
    assert _new_boundary_fraction(reference, candidate) == pytest.approx(1.0 / 49.0)
    assert _isolated_excursions(
        candidate - reference, threshold=0.05, radius=2, minimum_support=3
    ) == 1


def test_p2bb_rejects_source_manifest_hash_drift(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    copied = json.loads(json.dumps(contract["parents"]["source_manifest"]))
    copied["sha256"] = "0" * 64
    with pytest.raises(BWStructurePhotographicValueError, match="parent hash"):
        _load_manifest(ROOT, copied)
