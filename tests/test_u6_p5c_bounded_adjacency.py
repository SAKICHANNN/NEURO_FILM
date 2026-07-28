from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_bounded_adjacency import (
    evaluate_bounded_adjacency,
    load_contract,
)
from src.eval.physical_spatial_response import _slanted_edge
from src.film_physics import (
    SpatialResponseProfile,
    apply_bounded_development_adjacency,
    density_to_scan_transmittance,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p5c_bounded_adjacency_v1.json"
P5B = ROOT / "configs" / "u6_p5b_spatial_halo_audit_v1.json"
P5A = ROOT / "configs" / "u6_p5a_spatial_response_primitives_v1.json"
SENSITOMETRY = ROOT / "configs" / "u2_2a_sensitometry_primitive_v1.json"


def _profile(gain: float = 0.25) -> SpatialResponseProfile:
    return SpatialResponseProfile(
        pixel_pitch_um=1.0,
        forward_scatter_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_sigma_um_rgb=(1.0, 1.0, 1.0),
        development_adjacency_gain_rgb=(gain, gain, gain),
        dye_diffusion_sigma_um_rgb=(0.0, 0.0, 0.0),
        scanner_mtf_sigma_um_rgb=(0.0, 0.0, 0.0),
    )


def test_bounded_adjacency_respects_density_and_transmittance_limits() -> None:
    density, _ = _slanted_edge((64, 128), 5.0, 0.05, 3.0)
    output = apply_bounded_development_adjacency(
        density,
        _profile(),
        maximum_absolute_transmittance_delta=0.008,
        maximum_absolute_density_delta=0.08,
    )
    assert np.max(np.abs(output - density)) <= 0.08 + 1e-12
    assert (
        np.max(
            np.abs(
                density_to_scan_transmittance(output)
                - density_to_scan_transmittance(density)
            )
        )
        <= 0.008 + 1e-12
    )


def test_bounded_adjacency_zero_gain_is_exact_identity() -> None:
    density, _ = _slanted_edge((32, 64), 5.0, 0.05, 3.0)
    output = apply_bounded_development_adjacency(
        density,
        _profile(gain=0.0),
        maximum_absolute_transmittance_delta=0.008,
        maximum_absolute_density_delta=0.08,
    )
    assert np.array_equal(output, density)


def test_bounded_adjacency_rejects_invalid_bounds() -> None:
    density = np.ones((4, 4, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        apply_bounded_development_adjacency(
            density,
            _profile(),
            maximum_absolute_transmittance_delta=0.0,
            maximum_absolute_density_delta=0.08,
        )


def test_frozen_p5c_candidate_is_evaluated(tmp_path: Path) -> None:
    report = evaluate_bounded_adjacency(
        load_contract(CONTRACT),
        json.loads(P5B.read_text(encoding="utf-8")),
        json.loads(P5A.read_text(encoding="utf-8")),
        json.loads(SENSITOMETRY.read_text(encoding="utf-8")),
        diagnostic_path=tmp_path / "diagnostic.png",
    )
    assert report["automatic_pass"] is True
    assert report["stable_evidence_id"] == (
        "7d5a9bb40a6446996022b0892850f45a221837920d987cdb2df7ced40ed9c8f8"
    )
    assert all(report["p5b_halo_metrics"]["decisions"].values())
    assert report["candidate_decisions"]["density_bound"] is True
    assert report["candidate_decisions"]["transmittance_bound"] is True
    assert report["candidate_decisions"]["sign_preserved"] is True
    assert (tmp_path / "diagnostic.png").is_file()
