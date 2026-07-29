from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.scene_linear_backing_return_attribution import (
    _classify,
    _edge_distance,
    _isolated_mask,
    load_contract,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/u6_p3m_scene_linear_backing_return_attribution_v1.json"
)
DECISION = (
    ROOT
    / "configs/u6_p3m_scene_linear_backing_return_attribution_decision_v1.json"
)


def test_contract_is_read_only_and_pins_closed_parent() -> None:
    contract = load_contract(CONTRACT)
    _, manifest, _, _, _, _ = validate_contract(ROOT, contract)
    assert len(manifest) == 9
    assert "change any P3L" in contract["forbidden_actions"][0]
    assert "generate or inspect" in contract["forbidden_actions"][1]


def test_edge_distance_has_exact_frame_geometry() -> None:
    distance = _edge_distance((5, 7))
    assert distance.shape == (5, 7)
    assert np.all(distance[[0, -1], :] == 0)
    assert np.all(distance[:, [0, -1]] == 0)
    assert distance[2, 3] == 2


def test_isolated_mask_matches_frozen_support_semantics() -> None:
    delta = np.zeros((9, 9, 3), dtype=np.float64)
    delta[4, 4, 0] = 0.02
    isolated = _isolated_mask(
        delta, threshold=0.01, radius=2, minimum_support=3
    )
    assert int(np.count_nonzero(isolated)) == 1
    delta[4, 5, 0] = 0.02
    delta[5, 4, 0] = 0.02
    isolated = _isolated_mask(
        delta, threshold=0.01, radius=2, minimum_support=3
    )
    assert int(np.count_nonzero(isolated)) == 0


def test_classification_uses_frozen_precedence_and_fraction() -> None:
    base = {
        "bound_fail_within_halo_fraction": 0.1,
        "bound_fail_exact_zero_fraction": 0.2,
        "bound_fail_highlight_fraction": 0.3,
        "bound_fail_shadow_fraction": 0.4,
    }
    assert _classify(base, dominance_fraction=0.9) == (
        "distributed-or-content-associated"
    )
    assert _classify(
        {**base, "bound_fail_highlight_fraction": 0.91},
        dominance_fraction=0.9,
    ) == "highlight-associated"
    assert _classify(
        {
            **base,
            "bound_fail_within_halo_fraction": 0.95,
            "bound_fail_highlight_fraction": 0.99,
        },
        dominance_fraction=0.9,
    ) == "frame-boundary-associated"


def test_decision_preserves_closed_parent_and_forbids_simple_rescue() -> None:
    import json

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    run_a = ROOT / decision["repeat_evidence"]["run_a"]
    run_b = ROOT / decision["repeat_evidence"]["run_b"]
    assert run_a.read_bytes() == run_b.read_bytes()
    assert decision["classification"] == "distributed-or-content-associated"
    assert decision["aggregate"]["bound_fail_within_halo_fraction"] < (
        decision["aggregate"]["dominance_fraction"]
    )
    assert "do not create a P3L" in decision["next_leaf"]
