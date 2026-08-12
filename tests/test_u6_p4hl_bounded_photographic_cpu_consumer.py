from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.bounded_photographic_runtime import (
    BoundedPhotographicCpuRuntime,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4hl_bounded_photographic_cpu_consumer_v1.json"
PROFILE = (
    ROOT
    / "outputs/experiments/u6_p4hk_bounded_photographic_profile_bundle_v1/run_a/profile.json"
)


def test_p4hl_contract_binds_exact_profile_and_expected_outputs() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["parents"]["profile"]["bundle_sha256"] == (
        "8d20ce2874036befe78657b059ffafeb24e43d2e5bc1000512092a753e1e75df"
    )
    assert len(payload["fixture"]["expected_scanner_output_sha256"]) == 64


def test_cpu_runtime_rejects_wrong_bundle_before_source() -> None:
    with pytest.raises(ValueError, match="expected identity drift"):
        BoundedPhotographicCpuRuntime.load(
            PROFILE, expected_bundle_sha256="0" * 64
        )


def test_cpu_runtime_rejects_non_float32_source() -> None:
    runtime = BoundedPhotographicCpuRuntime.load(
        PROFILE,
        expected_bundle_sha256=(
            "8d20ce2874036befe78657b059ffafeb24e43d2e5bc1000512092a753e1e75df"
        ),
    )
    with pytest.raises(ValueError, match="CPU source"):
        runtime.render(np.zeros((3, 3, 3), dtype=np.float64), source_index=0)
