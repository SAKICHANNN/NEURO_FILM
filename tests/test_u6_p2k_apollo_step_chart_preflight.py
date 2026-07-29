from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.apollo_step_chart_preflight import (
    ApolloStepChartPreflightError,
    high_confidence_negative_edges,
)


ROOT = Path(__file__).resolve().parents[1]


def test_contract_stops_before_response_fitting() -> None:
    config = json.loads(
        (
            ROOT / "configs/u6_p2k_apollo_step_chart_preflight_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["curve_fitting_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
    assert config["development_disclosure"][
        "untouched_confirmation_claim_forbidden"
    ]


def test_separator_detection_keeps_separated_downward_edges() -> None:
    profile = np.concatenate(
        (
            np.full(20, 1000.0),
            np.linspace(1000.0, 9000.0, 20),
            np.full(4, 100.0),
            np.full(20, 18000.0),
            np.full(4, 100.0),
            np.full(20, 40000.0),
        )
    )
    edges = high_confidence_negative_edges(
        profile,
        smoothing_rows=3,
        minimum_drop_fraction=0.05,
        minimum_separation_rows=8,
    )
    assert len(edges) == 2
    assert edges[0]["sample_row_index"] < edges[1]["sample_row_index"]
    assert all(row["normalized_drop"] >= 0.05 for row in edges)


def test_close_edges_collapse_to_strongest_witness() -> None:
    profile = np.full(64, 30000.0)
    profile[20:22] = 10000.0
    profile[24:26] = 5000.0
    edges = high_confidence_negative_edges(
        profile,
        smoothing_rows=1,
        minimum_drop_fraction=0.1,
        minimum_separation_rows=8,
    )
    assert len(edges) == 1


@pytest.mark.parametrize(
    "profile,smoothing,drop,separation",
    [
        (np.asarray([]), 3, 0.01, 12),
        (np.asarray([0.0, np.nan, 1.0]), 3, 0.01, 12),
        (np.asarray([0.0, 1.0, 2.0]), 2, 0.01, 12),
        (np.asarray([0.0, 1.0, 2.0]), 3, 0.0, 12),
        (np.asarray([0.0, 1.0, 2.0]), 3, 0.01, 0),
    ],
)
def test_invalid_separator_inputs_fail_closed(
    profile: np.ndarray, smoothing: int, drop: float, separation: int
) -> None:
    with pytest.raises(ApolloStepChartPreflightError):
        high_confidence_negative_edges(
            profile,
            smoothing_rows=smoothing,
            minimum_drop_fraction=drop,
            minimum_separation_rows=separation,
        )
