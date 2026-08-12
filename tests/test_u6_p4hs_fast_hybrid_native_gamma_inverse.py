from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4hs_contract_preserves_accuracy_and_adds_runtime_gate() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4hs_fast_hybrid_native_gamma_inverse_v1.json").read_text(
            "utf-8"
        )
    )
    assert (
        contract["automatic_gates"]["maximum_absolute_developed_density_error"] == 2e-8
    )
    assert (
        contract["automatic_gates"]["maximum_kernel_seconds_per_131793_samples"] == 2.0
    )
    assert contract["candidate"]["direct_shape_upper_exclusive"] == 50.0
    assert contract["candidate"]["newton_shape_upper_exclusive"] == 10000.0
