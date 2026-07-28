from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_reference_scatter import (
    evaluate_reference_scatter,
    load_contract,
    write_report,
)
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.reference_scatter import (
    ReferenceScatterProfile,
    ScatterComponent,
    apply_reference_scatter,
    gaussian_kernel_2d,
    profile_from_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p1_reference_scatter_simulator_v1.json"


def _exposure(dtype: np.dtype = np.dtype(np.float64)) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        np.ones((9, 9, 3), dtype=dtype),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(8.0),
    )


def test_contract_profile_is_positive_energy_bounded_and_stable() -> None:
    profile = profile_from_contract(load_contract(CONTRACT))
    assert profile.direct_fraction.tolist() == pytest.approx([0.925, 0.955, 0.95])
    assert np.all(profile.total_scatter_fraction >= 0.0)
    assert len(profile.profile_sha256) == 64
    assert profile.profile_sha256 == profile_from_contract(
        load_contract(CONTRACT)
    ).profile_sha256
    for component in profile.components:
        kernel = gaussian_kernel_2d(component, profile.scale)
        assert np.all(kernel >= 0.0)
        assert float(np.sum(kernel)) == pytest.approx(1.0, abs=1e-15)


def test_profile_rejects_over_energy_and_bad_scale() -> None:
    component = ScatterComponent("too-much", 10.0, 6.0, (0.6, 0.6, 0.6))
    second = ScatterComponent("also-too-much", 20.0, 6.0, (0.6, 0.6, 0.6))
    with pytest.raises(ValueError, match="exceed"):
        ReferenceScatterProfile(8.0, (component, second))
    with pytest.raises(ValueError, match="pixel_pitch"):
        ReferenceScatterProfile(0.0, (component,))


def test_reference_requires_float64_layer_exposure_and_exact_scale() -> None:
    profile = profile_from_contract(load_contract(CONTRACT))
    with pytest.raises(TypeError, match="float64"):
        apply_reference_scatter(_exposure(np.dtype(np.float32)), profile)
    wrong_domain = PhysicalDomainArray(
        np.ones((9, 9, 3), dtype=np.float64),
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        ("red", "green", "blue"),
        PhysicalScale(8.0),
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        apply_reference_scatter(wrong_domain, profile)
    wrong_scale = PhysicalDomainArray(
        np.ones((9, 9, 3), dtype=np.float64),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(10.0),
    )
    with pytest.raises(ValueError, match="scale"):
        apply_reference_scatter(wrong_scale, profile)


def test_full_reference_report_passes_and_is_byte_repeatable(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_reference_scatter(contract)
    second = evaluate_reference_scatter(contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
    assert first["claim_ceiling"] == "generic-physical-inspired-reference-only"
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(first, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_report_fails_if_physical_far_halo_order_is_mutated() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["components"][1]["energy_fraction_rgb"] = [0.01, 0.02, 0.06]
    report = evaluate_reference_scatter(contract)
    assert report["decisions"]["far_halo_order"] is False
    assert report["automatic_pass"] is False
