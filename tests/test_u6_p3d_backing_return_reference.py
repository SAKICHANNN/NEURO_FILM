from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_backing_return import (
    evaluate_backing_return,
    load_contract,
    load_legacy_control,
    write_report,
)
from src.film_physics import (
    BackingReturnComponent,
    BackingReturnProfile,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_reference_backing_return,
    backing_return_kernel_2d,
    backing_return_profile_from_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p3d_backing_return_reference_v1.json"
LEGACY = ROOT / "configs" / "u6_p1_reference_scatter_simulator_v1.json"
COUPLING = (
    (0.96, 0.03, 0.01),
    (0.03, 0.94, 0.03),
    (0.01, 0.03, 0.92),
)


def _exposure(dtype: np.dtype = np.dtype(np.float64)) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        np.ones((9, 9, 3), dtype=dtype),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(8.0),
    )


def test_contract_profile_is_positive_bounded_and_stable() -> None:
    profile = backing_return_profile_from_contract(load_contract(CONTRACT))
    assert profile.maximum_return_fraction_by_source.tolist() == pytest.approx(
        [0.065, 0.018, 0.00576]
    )
    assert len(profile.profile_sha256) == 64
    assert profile.profile_sha256 == backing_return_profile_from_contract(
        load_contract(CONTRACT)
    ).profile_sha256
    for component in profile.components:
        assert np.all(component.return_weights >= 0.0)
        kernel = backing_return_kernel_2d(component, profile.scale)
        assert np.all(kernel >= 0.0)
        assert float(np.sum(kernel)) == pytest.approx(1.0, abs=1e-15)


def test_profile_rejects_invalid_energy_coupling_and_scale() -> None:
    component = BackingReturnComponent(
        "valid", 10.0, 6.0, (0.6, 0.6, 0.6), COUPLING
    )
    second_component = BackingReturnComponent(
        "second", 20.0, 6.0, (0.6, 0.6, 0.6), COUPLING
    )
    with pytest.raises(ValueError, match="aggregate"):
        BackingReturnProfile(8.0, (component, second_component))
    bad_coupling = (
        (1.0, 0.0, 0.0),
        (0.1, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="column"):
        BackingReturnComponent(
            "bad-coupling", 10.0, 6.0, (0.1, 0.1, 0.1), bad_coupling
        )
    with pytest.raises(ValueError, match="finite"):
        BackingReturnComponent(
            "nonfinite", 10.0, 6.0, (float("nan"), 0.1, 0.1), COUPLING
        )
    with pytest.raises(ValueError, match="pixel_pitch"):
        BackingReturnProfile(0.0, (component,))


def test_reference_requires_float64_layer_exposure_and_exact_scale() -> None:
    profile = backing_return_profile_from_contract(load_contract(CONTRACT))
    with pytest.raises(TypeError, match="float64"):
        apply_reference_backing_return(_exposure(np.dtype(np.float32)), profile)
    wrong_domain = PhysicalDomainArray(
        np.ones((9, 9, 3), dtype=np.float64),
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        ("red", "green", "blue"),
        PhysicalScale(8.0),
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        apply_reference_backing_return(wrong_domain, profile)
    wrong_scale = PhysicalDomainArray(
        np.ones((9, 9, 3), dtype=np.float64),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(10.0),
    )
    with pytest.raises(ValueError, match="scale"):
        apply_reference_backing_return(wrong_scale, profile)


def test_full_reference_report_passes_and_is_byte_repeatable(
    tmp_path: Path,
) -> None:
    contract = load_contract(CONTRACT)
    legacy = load_legacy_control(LEGACY)
    first = evaluate_backing_return(contract, legacy)
    second = evaluate_backing_return(contract, legacy)
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
    assert first["metrics"]["additive_energy_above_legacy"] > 0.0
    assert first["claim_ceiling"].startswith(
        "generic physical-inspired float64 reference"
    )
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(first, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_report_fails_if_far_halo_spectral_order_is_reversed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for component in contract["components"]:
        red, green, blue = component["return_fraction_rgb"]
        component["return_fraction_rgb"] = [blue, green, red]
    report = evaluate_backing_return(contract, load_legacy_control(LEGACY))
    assert report["decisions"]["far_halo_order"] is False
    assert report["automatic_pass"] is False
