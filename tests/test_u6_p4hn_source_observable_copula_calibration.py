from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.calibrated_native_histogram_copula import (
    apply_source_observable_calibrated_copula,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p4hn_contract_binds_failed_p4hm_and_disjoint_fixtures() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4hn_source_observable_copula_calibration_v1.json").read_text(
            "utf-8"
        )
    )
    assert contract["parents"]["p4hm_evidence"]["required_decision"] == (
        "retain_python_bounded_consumer_without_native_transport"
    )
    seeds = [row["field_seed"] for row in contract["confirmation_fixtures"]]
    assert len(seeds) == len(set(seeds)) == 2
    assert 2026081311 not in seeds
    assert contract["candidate"]["reference_or_target_pixel_reads"] == 0
    assert contract["candidate"]["native_source_change_allowed"] is False


def test_p4hn_rejects_zero_iterations_before_native_call() -> None:
    with pytest.raises(ValueError, match="iterations"):
        apply_source_observable_calibrated_copula(
            object(),  # type: ignore[arg-type]
            np.zeros((4, 3), dtype=np.float32),
            profile_correlation=np.eye(3),
            rank_bins=65536,
            iterations=0,
        )
