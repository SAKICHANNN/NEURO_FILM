from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_mtf import (
    MeasuredMtfError,
    compile_and_evaluate,
    load_contract,
)
from src.film_physics.measured_mtf import ChannelPsf, PositivePsfComponent

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p5j_kodak_250d_positive_psf_compiler_v1.json"
PARENT = ROOT / "outputs/u6_p5i_kodak_250d_mtf_source/report_run1.json"
GRAPH = ROOT / "outputs/u5_r2aa0_source_audit/extracted/250d_mtf.png"


def test_positive_psf_preserves_dc_and_is_monotone() -> None:
    model = ChannelPsf(
        "two_gaussian",
        (PositivePsfComponent(0.3, 2.0), PositivePsfComponent(0.7, 7.0)),
    )
    response = model.response(np.linspace(0.0, 65.0, 1025))
    assert response[0] == 1.0
    assert np.all(response > 0.0)
    assert np.all(np.diff(response) <= 0.0)


def test_positive_psf_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        ChannelPsf("single_gaussian", (PositivePsfComponent(0.9, 3.0),))


def test_contract_rejects_relaxed_confirmation_gate(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["confirmation"]["all_channel_rmse_ratio_max"] = 2.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MeasuredMtfError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not PARENT.is_file() or not GRAPH.is_file(),
    reason="exact P5I evidence is unavailable",
)
def test_exact_compiler_is_repeatable_and_passes_confirmation(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first_bundle, first = compile_and_evaluate(
        contract, ROOT, overlay_path=tmp_path / "first.png"
    )
    second_bundle, second = compile_and_evaluate(
        contract, ROOT, overlay_path=tmp_path / "second.png"
    )
    assert first_bundle == second_bundle
    assert first == second
    assert first["automatic_pass"]
    assert all(first["gate_results"].values())
    assert [
        first["channel_results"][channel]["selected_family"]
        for channel in ("blue", "green", "red")
    ] == ["single_gaussian", "delta_plus_gaussian", "two_gaussian"]
    assert first["all_channel_confirmation_rmse_ratio"] < 0.25
    assert first["wrong_channel_mean_rmse_ratio"] > 5.0
    assert first_bundle["measured_frequency_interval_cycles_per_mm"] == [25.0, 65.0]
