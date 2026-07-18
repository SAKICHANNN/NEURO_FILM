from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.rec2020_visual_ood import (
    anonymous_candidate_order,
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
    risk_rank,
    validate_filmr_selection,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "u1_4c2_rec2020_visual_ood_v1.json").read_text(encoding="utf-8"))


def test_frozen_filmr_selection_matches_manifest_and_payloads() -> None:
    paths = validate_filmr_selection(CONFIG, root=ROOT)
    assert len(paths) == 8
    assert len(set(paths)) == 8


def test_srgb_preview_transfer_roundtrip_is_bounded() -> None:
    encoded = np.random.default_rng(14037).random((9, 11, 3), dtype=np.float32)
    restored = linear_srgb_to_encoded(encoded_srgb_to_linear(encoded))
    assert float(np.max(np.abs(restored - encoded))) <= 2e-6


def test_anonymous_candidate_order_is_deterministic_and_round_specific() -> None:
    candidates = [f"candidate_{index}" for index in range(12)]
    first = anonymous_candidate_order("sample", 1, candidates)
    repeated = anonymous_candidate_order("sample", 1, list(reversed(candidates)))
    second = anonymous_candidate_order("sample", 2, candidates)
    assert first == repeated
    assert first != second
    assert sorted(first) == sorted(candidates)


def test_risk_rank_uses_only_frozen_metrics_and_stable_ties() -> None:
    records = [
        {
            "logical_id": logical_id,
            "new_rec2020_boundary_fraction": boundary,
            "srgb_preview_clip_fraction": clipping,
            "residual_gradient_q999": gradient,
        }
        for logical_id, boundary, clipping, gradient in [
            ("a", 0.0, 0.1, 0.3),
            ("b", 0.2, 0.0, 0.1),
            ("c", 0.1, 0.3, 0.0),
            ("d", 0.0, 0.0, 0.0),
        ]
    ]
    ranked = risk_rank(records, 3)
    assert [row["logical_id"] for row in ranked] == ["a", "b", "c"]
    assert all("risk_rank" in row and "rank_sum" in row for row in ranked)
