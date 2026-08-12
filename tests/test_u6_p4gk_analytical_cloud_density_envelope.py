import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure_source_derived,
    render_physical_partition,
)
from src.film_physics.bounded_cloud_density import (
    apply_bounded_cloud_density_residual,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gk_analytical_cloud_density_envelope_v1.json"


def test_p4gk_contract_requires_analytical_bounds_before_chart():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["candidate"]["bounds"] == "exact-profile-black-and-white-reference-density"
    assert contract["execution"]["p4gh_chart_pixels_read"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert contract["gates"]["minimum_channel_scan_response_span"] == 0.5
    assert contract["gates"]["minimum_interior_cloud_density_residual_rms"] == 0.0001
    assert contract["candidate"]["sensitometry_endpoint_policy"].startswith("derive-exact")


def test_p4gk_scales_outward_density_residual_without_clipping():
    base = np.array([[[0.05, 0.65, 1.25]]], dtype=np.float64)
    candidate = np.array([[[-0.2, 0.8, 1.7]]], dtype=np.float64)
    bounded, diagnostics = apply_bounded_cloud_density_residual(
        base,
        candidate,
        black_reference_density=np.array([0.05, 0.05, 0.05]),
        white_reference_density=np.array([1.25, 1.25, 1.25]),
    )
    assert np.all(bounded >= 0.05) and np.all(bounded <= 1.25)
    assert bounded[0, 0, 0] == np.float32(0.05)
    assert bounded[0, 0, 1] > np.float32(0.65)
    assert diagnostics["limited_fraction"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["hard_clipping_used"] == 0.0


def test_p4gk_rejects_base_outside_profile_envelope():
    with pytest.raises(ValueError, match="invalid cloud density envelope"):
        apply_bounded_cloud_density_residual(
            np.array([[[0.049, 0.5, 0.5]]]),
            np.array([[[0.1, 0.5, 0.5]]]),
            black_reference_density=np.full(3, 0.05),
            white_reference_density=np.full(3, 1.25),
        )


def test_p4gk_uniform_endpoints_complete_but_span_gate_fails(tmp_path: Path):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fixture = contract["fixture"]
    physical = json.loads(
        (ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json").read_text(
            encoding="utf-8"
        )
    )
    physical["fixture"]["full_height"] = fixture["height"]
    physical["fixture"]["width"] = fixture["width"]
    library = _configure_source_derived(
        _build(
            ROOT,
            tmp_path / "build",
            None,
            bridge_source="nf_sensitometry_cloud_bridge_f32_v2.c",
        )
    )
    responses = []
    diagnostics = []
    for level in (0.0, 1.0):
        source = np.full(
            (fixture["height"], fixture["width"], 3), level, dtype=np.float32
        )
        output = render_physical_partition(
            library,
            physical,
            source,
            0,
            fixture["height"],
            source_derived_expected=True,
            enforce_density_envelope=True,
            exact_sensitometry_endpoints=True,
            density_envelope_diagnostics=diagnostics,
        )
        responses.append(np.median(output, axis=(0, 1)))
    assert np.min(responses[1] - responses[0]) < 0.5
    assert all(row["maximum_density_envelope_violation"] == 0.0 for row in diagnostics)


def test_p4gk_formal_result_stops_on_endpoint_span():
    result = json.loads(
        (ROOT / "docs/evidence/U6_P4GK_ANALYTICAL_CLOUD_DENSITY_ENVELOPE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["automatic_pass"] is False
    assert result["stable"]["gates"]["density_envelope"] is True
    assert result["stable"]["gates"]["response_span"] is False
    assert result["stable"]["scan_response_span_rgb"] == pytest.approx(
        [0.3608871101100853, 0.3702342210967949, 0.5029060243328026]
    )
    assert result["stable_evidence_id"] == "ac9a55fcbd0708a02e4bb00b5478683c21c9d4f1dfe532bcb32e8ca7364dcbef"
