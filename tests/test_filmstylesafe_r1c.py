from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.filmstylesafe_r1c import (
    analyze_metric_separation,
    conventional_pair_metrics,
    load_r1c_contract,
    proxy_severe_label,
    run_a0_metric_pilot,
    scis_v0,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_r1c_contract() -> None:
    contract = load_r1c_contract(ROOT / "configs" / "filmstylesafe_r1c_contract_v1.json")
    assert contract["authorization"]["model_training"] is False
    assert contract["authorization"]["hidden_split_population"] is False


def test_scis_v0_higher_on_chroma_island_than_mild_shift() -> None:
    base = np.full((64, 64, 3), 0.85, dtype=np.float32)
    mild = np.clip(base * 1.05, 0.0, 1.0)
    island = base.copy()
    island[20:40, 20:40, 0] = 1.0
    island[20:40, 20:40, 1] = 0.05
    island[20:40, 20:40, 2] = 0.85
    mild_score = scis_v0(base, mild)["scis_v0_score"]
    island_score = scis_v0(base, island)["scis_v0_score"]
    assert island_score > mild_score


def test_conventional_metrics_deterministic() -> None:
    a = np.full((32, 32, 3), 0.5, dtype=np.float32)
    b = np.clip(a + 0.1, 0.0, 1.0)
    m1 = conventional_pair_metrics(a, b)
    m2 = conventional_pair_metrics(a, b)
    assert m1 == m2
    assert m1["psnr"] > 0
    assert 0.0 <= m1["ssim"] <= 1.0


def test_proxy_severe_labels() -> None:
    assert proxy_severe_label("synthetic_failure") is True
    assert proxy_severe_label("legitimate_local_hard_negative") is False
    assert proxy_severe_label("real_failure") is None


def test_analyze_metric_separation_reports_gap() -> None:
    rows = [
        {
            "member_id": "pos",
            "role": "synthetic_failure",
            "proxy_severe": True,
            "hard_negative": False,
            "mean_delta_e76": 5.0,
            "mean_abs_rgb": 0.1,
            "psnr": 20.0,
            "ssim": 0.8,
            "scis_v0_score": 50.0,
        },
        {
            "member_id": "neg_style",
            "role": "strength_control",
            "proxy_severe": False,
            "hard_negative": False,
            "mean_delta_e76": 8.0,
            "mean_abs_rgb": 0.2,
            "psnr": 15.0,
            "ssim": 0.7,
            "scis_v0_score": 5.0,
        },
        {
            "member_id": "neg_hard",
            "role": "legitimate_local_hard_negative",
            "proxy_severe": False,
            "hard_negative": True,
            "mean_delta_e76": 0.1,
            "mean_abs_rgb": 0.01,
            "psnr": 40.0,
            "ssim": 0.99,
            "scis_v0_score": 1.0,
        },
    ]
    result = analyze_metric_separation(rows)
    assert result["conventional_metric_gap_at_zero_fpr"] is True
    assert result["scis_v0_perfect_sensitivity_at_zero_fpr"] is True
    assert result["scis_v0_perfect_vs_hardneg_external"] is True


def test_a0_pilot_runs_on_bound_inventory() -> None:
    contract = load_r1c_contract(ROOT / "configs" / "filmstylesafe_r1c_contract_v1.json")
    inventory = json.loads(
        (ROOT / "configs" / "filmstylesafe_r1b6_a0_inventory_v1.json").read_text(encoding="utf-8")
    )
    for path in contract["parent_sources"].values():
        if not (ROOT / path).is_file():
            pytest.skip("local A0 parent sources unavailable")
    for member in inventory["members"]:
        if member.get("binding_status") == "bound" and member["role"] != "source_scene":
            if not (ROOT / member["artifact_path"]).is_file():
                pytest.skip("local A0 artifacts unavailable")
    result = run_a0_metric_pilot(
        inventory,
        root=ROOT,
        parent_sources=contract["parent_sources"],
        scis_params=contract["scis_v0"],
    )
    assert result["n_positives"] >= 2
    assert result["n_negatives"] >= 2
    assert "conventional_metric_gap_at_zero_fpr" in result
