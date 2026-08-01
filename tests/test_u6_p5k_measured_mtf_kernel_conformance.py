from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_mtf_kernel import (
    MeasuredMtfKernelError,
    evaluate_kernel,
    load_contract,
)
from src.film_physics.measured_mtf import (
    ChannelPsf,
    PositivePsfComponent,
    apply_compiled_positive_psf,
    compile_channel_psf,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p5k_measured_mtf_kernel_conformance_v1.json"
BUNDLE = ROOT / "outputs/u6_p5j_kodak_250d_positive_psf_compiler/bundle_run1.json"


def test_cell_integrated_kernel_is_positive_and_constant_preserving() -> None:
    model = ChannelPsf(
        "delta_plus_gaussian",
        (PositivePsfComponent(0.3, 0.0), PositivePsfComponent(0.7, 5.0)),
    )
    channel = compile_channel_psf(model, pixel_pitch_um=1.5875, truncate_sigma=5.0)
    values = np.full((31, 47, 3), 0.4)
    output = apply_compiled_positive_psf(values, (channel, channel, channel))
    assert np.min(output) >= 0.0
    assert np.max(np.abs(output - values)) <= 1e-12


def test_contract_rejects_lower_reference_sampling(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["compiler"]["reference_dpi"] = 4000.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MeasuredMtfKernelError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(not BUNDLE.is_file(), reason="exact P5J bundle is unavailable")
def test_exact_kernel_audit_is_repeatable_and_detects_4000dpi_gap(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_kernel(contract, ROOT, diagnostic_path=tmp_path / "first.png")
    second = evaluate_kernel(contract, ROOT, diagnostic_path=tmp_path / "second.png")
    assert first == second
    assert first["automatic_pass"]
    assert all(first["gate_results"].values())
    assert first["reference_max_error"] < 0.01
    assert first["direct_4000dpi_max_error"] > 0.02
    assert first["decision"] == "retain_reference_open_lod_compiler"
