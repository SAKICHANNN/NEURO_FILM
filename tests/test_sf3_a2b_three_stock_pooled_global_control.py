from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.real_film.three_stock_k1_baseline import StockFrameSamples
from src.real_film.three_stock_pooled_global_control import evaluate

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a2b_three_stock_pooled_global_control_v1.json"
STOCKS = (
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
)


def _population(
    gains: dict[str, np.ndarray],
) -> tuple[dict[str, list[StockFrameSamples]], dict[str, list[StockFrameSamples]]]:
    rng = np.random.default_rng(91827)
    development = {stock: [] for stock in STOCKS}
    confirmation = {stock: [] for stock in STOCKS}
    for index in range(4):
        source = rng.uniform(0.08, 0.62, size=(96, 3))
        for stock in STOCKS:
            development[stock].append(
                StockFrameSamples(
                    scene_id=f"dev-{index}",
                    frame_id=f"{stock}-dev-{index}",
                    roll_id=f"roll-{index % 2}",
                    source=source,
                    target=source * gains[stock],
                )
            )
    for index in range(4):
        source = rng.uniform(0.08, 0.62, size=(96, 3))
        for stock in STOCKS:
            confirmation[stock].append(
                StockFrameSamples(
                    scene_id=f"confirmation-{index}",
                    frame_id=f"{stock}-confirmation-{index}",
                    roll_id=f"held-roll-{stock}",
                    source=source,
                    target=source * gains[stock],
                )
            )
    return development, confirmation


def test_distinct_stock_operators_beat_one_pooled_global_control() -> None:
    development, confirmation = _population(
        {
            STOCKS[0]: np.array([1.18, 0.78, 0.68]),
            STOCKS[1]: np.array([0.82, 1.10, 0.88]),
            STOCKS[2]: np.array([0.68, 0.86, 1.22]),
        }
    )
    report = evaluate(
        CONTRACT, root=ROOT, development=development, confirmation=confirmation
    )
    assert report["automatic_pass"] is True
    assert report["confirmation_target_fit_or_selection_reads"] == 0
    assert report["adaptive_or_retrieval_models_fitted"] == 0
    assert all(
        row["metrics"]["stock_specific_win_rate_over_pooled_global"] == 1.0
        for row in report["stock_results"].values()
    )


def test_common_effect_fails_stock_specific_incremental_value() -> None:
    gain = np.array([0.92, 1.04, 0.81])
    development, confirmation = _population({stock: gain for stock in STOCKS})
    report = evaluate(
        CONTRACT, root=ROOT, development=development, confirmation=confirmation
    )
    assert report["automatic_pass"] is False
    assert report["decision"].startswith("RETAIN_POOLED_OR_IDENTITY")
    assert not any(row["automatic_pass"] for row in report["stock_results"].values())


def test_confirmation_targets_cannot_change_frozen_selections() -> None:
    gains = {
        STOCKS[0]: np.array([1.18, 0.78, 0.68]),
        STOCKS[1]: np.array([0.82, 1.10, 0.88]),
        STOCKS[2]: np.array([0.68, 0.86, 1.22]),
    }
    development, confirmation = _population(gains)
    first = evaluate(
        CONTRACT, root=ROOT, development=development, confirmation=confirmation
    )
    changed = {
        stock: [
            StockFrameSamples(
                scene_id=row.scene_id,
                frame_id=row.frame_id,
                roll_id=row.roll_id,
                source=row.source,
                target=np.clip(1.0 - row.target, 0.0, 1.0),
            )
            for row in rows
        ]
        for stock, rows in confirmation.items()
    }
    second = evaluate(
        CONTRACT, root=ROOT, development=development, confirmation=changed
    )
    assert (
        first["selected_pooled_global_operator"]
        == second["selected_pooled_global_operator"]
    )
    assert first["pooled_operator"] == second["pooled_operator"]
    assert {
        stock: row["selected_stock_specific_operator"]
        for stock, row in first["stock_results"].items()
    } == {
        stock: row["selected_stock_specific_operator"]
        for stock, row in second["stock_results"].items()
    }


def test_contract_binds_exact_parent_and_three_stocks() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["required_stocks"] == list(STOCKS)
    assert (
        contract["pooled_selection"][
            "confirmation_target_access_before_all_selections_frozen"
        ]
        == 0
    )
    parent = ROOT / contract["parent"]["k1_contract"]["path"]
    import hashlib

    assert (
        hashlib.sha256(parent.read_bytes()).hexdigest()
        == contract["parent"]["k1_contract"]["sha256"]
    )
