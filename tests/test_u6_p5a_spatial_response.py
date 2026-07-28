from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from src.eval.physical_spatial_response import (
    _slanted_edge,
    evaluate_spatial_response,
    load_contract,
)
from src.film_physics import (
    SpatialResponseProfile,
    apply_development_adjacency,
    apply_dye_diffusion,
    apply_forward_scatter,
    apply_scanner_mtf,
    density_to_scan_transmittance,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p5a_spatial_response_primitives_v1.json"


def _profile() -> SpatialResponseProfile:
    return SpatialResponseProfile(
        1.0,
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.1, 0.1, 0.1),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    )


def test_contract_and_slanted_edge_are_valid() -> None:
    contract = load_contract(CONTRACT)
    values, distance = _slanted_edge((32, 64), 5.0, 0.1, 0.8)
    assert contract["node"] == "U6.P5A"
    assert values.shape == (32, 64, 3)
    assert distance.shape == (32, 64)


def test_spatial_primitives_preserve_domains() -> None:
    profile = _profile()
    values, _ = _slanted_edge((32, 64), 5.0, 0.1, 0.8)
    exposure = apply_forward_scatter(values, profile)
    density = apply_development_adjacency(exposure + 0.2, profile)
    diffused = apply_dye_diffusion(density, profile)
    transmission = density_to_scan_transmittance(diffused)
    scanned = apply_scanner_mtf(transmission, profile)
    assert np.all(exposure >= 0.0)
    assert np.all(density >= 0.0)
    assert np.all(diffused >= 0.0)
    assert np.all((scanned > 0.0) & (scanned <= 1.0))


def test_frozen_spatial_response_report_passes_representation_gates() -> None:
    sensitometry = json.loads(
        (ROOT / "configs" / "u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    report = evaluate_spatial_response(load_contract(CONTRACT), sensitometry)
    assert report["stable_evidence_id"] == (
        "08689b8b7a7c06ea55fbc3cb7f68b213a820a93bea5f46dddbf7b4e4783eb596"
    )
    assert report["automatic_pass"] is True
    assert all(report["decisions"].values())
    assert max(
        item["undershoot"] for item in report["adjacency_edge_metrics"]
    ) > 0.3
