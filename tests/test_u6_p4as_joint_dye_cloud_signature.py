from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.joint_dye_cloud_signature import (
    evaluate_joint_dye_cloud_signature,
    gaussian_mtf,
    load_contract,
    predicted_nps_shape,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4as_joint_dye_cloud_mtf_nps_v1.json"


def test_gaussian_signal_and_noise_transfers_share_one_kernel() -> None:
    frequencies = np.asarray([0.0, 0.05, 0.1, 0.2])
    mtf = gaussian_mtf(frequencies, 1.1, 4.0)
    assert mtf[0] == pytest.approx(1.0, abs=1e-15)
    assert np.all(np.diff(mtf) < 0.0)
    edges = np.asarray([0.015, 0.05, 0.1, 0.2, 0.27])
    nps = predicted_nps_shape((128, 128), edges, 1.1, 4.0)
    assert np.sum(nps) == pytest.approx(1.0, abs=1e-15)
    assert np.all(nps > 0.0)


def test_joint_evaluator_repeats_and_keeps_confirmation_separate() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_joint_dye_cloud_signature(contract)
    second = evaluate_joint_dye_cloud_signature(contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert first["stable_evidence_id"] == (
        "6adb42caf988810a8a4cd885e2e793b7300edf23978b3090bc6037f02dc4b09d"
    )
    assert (
        first["bundle"]["development_frequencies_cycles_per_pixel"]
        == (contract["split"]["development_mtf_frequencies_cycles_per_pixel"])
    )
    assert "confirmation_mtf_frequencies_cycles_per_pixel" not in first["bundle"]
    assert first["decisions"]["confirmation_after_bundle_freeze"] is True
    assert len(first["layer_metrics"]) == 3
    assert all(len(item["densities"]) == 3 for item in first["layer_metrics"])


def test_contract_rejects_wrong_layer_nonpermutation() -> None:
    contract = load_contract(CONTRACT)
    contract["controls"]["wrong_layer_mapping"] = [0, 0, 1]
    with pytest.raises(ValueError, match="permutation"):
        evaluate_joint_dye_cloud_signature(contract)
