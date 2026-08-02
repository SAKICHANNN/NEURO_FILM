from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.typed_thomas_image_formation import (
    evaluate_typed_thomas_image_formation,
)
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    effective_spectrum_identity_scanner,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4by_typed_thomas_image_formation_v1.json"


def test_identity_observer_has_no_spatial_or_noise_stage() -> None:
    profile = effective_spectrum_identity_scanner("test-identity")
    assert profile.spectral_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert profile.mtf_sigma_um_rgb == (0.0, 0.0, 0.0)
    assert profile.local_flare_fraction == 0.0
    assert profile.global_flare_fraction == 0.0
    assert profile.shot_noise_variance_scale == 0.0
    assert profile.read_noise_variance == 0.0


def test_physical_domain_array_rejects_domain_relabelling() -> None:
    values = np.ones((2, 3, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        PhysicalDomainArray(
            values,
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
            ("red", "green", "blue"),
            PhysicalScale(6.35),
        )


def test_formal_evaluation_is_exact_and_passes() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_typed_thomas_image_formation(contract, ROOT)
    second = evaluate_typed_thomas_image_formation(contract, ROOT)
    assert first == second
    assert first["automatic_pass"]
    assert first["fixture_count"] == 6
    assert all(first["gate_results"].values())


def test_frozen_contract_rejects_scanner_double_counting() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["candidate"]["scanner_stages"] = ["spectral", "mtf"]
    with pytest.raises(RuntimeError, match="contract drift"):
        evaluate_typed_thomas_image_formation(contract, ROOT)
