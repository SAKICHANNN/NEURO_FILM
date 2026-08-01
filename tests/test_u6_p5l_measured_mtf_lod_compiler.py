from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_mtf_lod import (
    MeasuredMtfLodError,
    compile_and_evaluate_lod,
    load_contract,
)
from src.film_physics.measured_mtf import (
    ChannelPsf,
    PositivePsfComponent,
    apply_compiled_positive_psf,
    apply_zero_order_hold_reference,
    compile_channel_psf,
    compile_zero_order_hold_lod,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p5l_measured_mtf_lod_compiler_v1.json"
BUNDLE = ROOT / "outputs/u6_p5j_kodak_250d_positive_psf_compiler/bundle_run1.json"
P5K = ROOT / "outputs/u6_p5k_measured_mtf_kernel_conformance/report_run1.json"


def test_zero_order_hold_lod_matches_reference_for_random_array() -> None:
    model = ChannelPsf(
        "two_gaussian",
        (PositivePsfComponent(0.3, 2.0), PositivePsfComponent(0.7, 7.0)),
    )
    reference_channel = compile_channel_psf(
        model, pixel_pitch_um=1.5875, truncate_sigma=5.0
    )
    reference = (reference_channel, reference_channel, reference_channel)
    lod = compile_zero_order_hold_lod(reference, scale=4)
    values = np.random.default_rng(7).random((23, 31, 3))
    expected = apply_zero_order_hold_reference(values, reference, scale=4)
    actual = apply_compiled_positive_psf(values, lod)
    assert np.max(np.abs(expected - actual)) <= 1e-12


def test_contract_rejects_reconstruction_change(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["reference"]["input_reconstruction"] = "bicubic"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MeasuredMtfLodError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not BUNDLE.is_file() or not P5K.is_file(),
    reason="exact P5J/P5K evidence is unavailable",
)
def test_exact_lod_compiler_is_repeatable_and_passes(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first_bundle, first = compile_and_evaluate_lod(
        contract, ROOT, diagnostic_path=tmp_path / "first.png"
    )
    second_bundle, second = compile_and_evaluate_lod(
        contract, ROOT, diagnostic_path=tmp_path / "second.png"
    )
    assert first_bundle == second_bundle
    assert first == second
    assert first["automatic_pass"]
    assert all(first["gate_results"].values())
    assert first["reference_vs_lod_max_error"] <= 1e-12
    assert first["lod_vs_naive_error_reduction"] >= 0.99
    assert first["decision"] == "open_joint_spatial_budget_audit"
