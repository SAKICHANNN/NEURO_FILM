from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.real_film.three_stock_k1_baseline import (
    SINGLE_STOCK_REPORT_SCHEMA,
    StockFrameSamples,
    ThreeStockK1BaselineError,
    evaluate,
    evaluate_single_stock,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sf3_a2_three_stock_k1_baseline_v1.json"
STOCKS = (
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
)


def _population(*, identity_targets: bool = False):
    transforms = {
        "fujifilm_velvia_50": (
            np.array([1.12, 0.98, 0.90]),
            np.array([0.01, 0.01, 0.03]),
        ),
        "kodak_portra_400": (
            np.array([0.94, 1.03, 1.06]),
            np.array([0.04, 0.02, 0.00]),
        ),
        "kodak_ektar_100": (np.array([1.07, 0.93, 1.11]), np.array([0.00, 0.04, 0.01])),
    }
    development = {stock: [] for stock in STOCKS}
    confirmation = {stock: [] for stock in STOCKS}
    for roll_index in range(2):
        for scene_index in range(4):
            source = np.random.default_rng(
                1000 + 10 * roll_index + scene_index
            ).uniform(0.12, 0.72, size=(512, 3))
            for stock in STOCKS:
                gain, bias = transforms[stock]
                target = source if identity_targets else source * gain + bias
                development[stock].append(
                    StockFrameSamples(
                        scene_id=f"dev-scene-{scene_index}",
                        frame_id=f"{stock}-dev-{roll_index}-{scene_index}",
                        roll_id=f"{stock}-dev-roll-{roll_index}",
                        source=source,
                        target=target,
                    )
                )
    for scene_index in range(4):
        source = np.random.default_rng(2000 + scene_index).uniform(
            0.12, 0.72, size=(512, 3)
        )
        for stock in STOCKS:
            gain, bias = transforms[stock]
            target = source if identity_targets else source * gain + bias
            confirmation[stock].append(
                StockFrameSamples(
                    scene_id=f"confirmation-scene-{scene_index}",
                    frame_id=f"{stock}-confirmation-{scene_index}",
                    roll_id=f"{stock}-confirmation-roll",
                    source=source,
                    target=target,
                )
            )
    return development, confirmation


def test_contract_binds_integrity_parent() -> None:
    _, contract = load_contract(CONFIG, root=ROOT)
    assert contract["status"] == "FROZEN_BEFORE_PHYSICAL_TARGET_ADMISSION"
    assert contract["required_stocks"] == list(STOCKS)


def test_three_independent_k1_affine_baselines_pass_synthetic_truth() -> None:
    development, confirmation = _population()
    first = evaluate(
        CONFIG,
        root=ROOT,
        development=development,
        confirmation=confirmation,
    )
    second = evaluate(
        CONFIG,
        root=ROOT,
        development=development,
        confirmation=confirmation,
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert first["all_three_stocks_pass"] is True
    assert first["all_three_stock_pairs_distinguishable"] is True
    assert first["selection_used_confirmation_targets"] is False
    assert (
        first["outer_runner_must_decode_confirmation_targets_after_selection_freeze"]
        is True
    )
    assert first["confirmation_frames_evaluated_after_all_selections_frozen"] == 12
    assert first["adaptive_or_retrieval_models_fitted"] == 0
    assert first["latent_modes_fitted"] == 0
    assert all(
        row["selected_operator"] == "bounded_per_channel_affine"
        for row in first["stocks"].values()
    )


def test_identity_targets_fail_stock_signal_without_capacity_rescue() -> None:
    development, confirmation = _population(identity_targets=True)
    report = evaluate(
        CONFIG,
        root=ROOT,
        development=development,
        confirmation=confirmation,
    )
    assert report["automatic_pass"] is False
    assert report["all_three_stock_pairs_distinguishable"] is False
    assert report["decision"].startswith("RETAIN_PER_STOCK_SIMPLEST")


def test_single_stock_entry_passes_without_cross_stock_claim() -> None:
    development, confirmation = _population()
    stock = "kodak_ektar_100"
    first = evaluate_single_stock(
        CONFIG,
        root=ROOT,
        stock=stock,
        development=development[stock],
        confirmation=confirmation[stock],
    )
    second = evaluate_single_stock(
        CONFIG,
        root=ROOT,
        stock=stock,
        development=development[stock],
        confirmation=confirmation[stock],
    )
    assert first == second
    assert first["schema"] == SINGLE_STOCK_REPORT_SCHEMA
    assert first["automatic_pass"] is True
    assert first["wrong_stock_control_evaluated"] is False
    assert first["cross_stock_distinguishability_evaluated"] is False
    assert first["selected_operator"] == "bounded_per_channel_affine"
    assert first["decision"] == (
        "RETAIN_SINGLE_STOCK_K1_CANDIDATE_PENDING_THREE_STOCK_CONTROLS"
    )


def test_single_stock_entry_fails_identity_truth_without_rescue() -> None:
    development, confirmation = _population(identity_targets=True)
    stock = "kodak_portra_400"
    report = evaluate_single_stock(
        CONFIG,
        root=ROOT,
        stock=stock,
        development=development[stock],
        confirmation=confirmation[stock],
    )
    assert report["automatic_pass"] is False
    assert report["decision"] == (
        "RETAIN_SINGLE_STOCK_IDENTITY_BASELINE_WITHOUT_CAPACITY_OR_ROUTER_RESCUE"
    )


def test_single_stock_entry_rejects_unknown_stock_and_scene_duplicates() -> None:
    development, confirmation = _population()
    stock = "kodak_ektar_100"
    with pytest.raises(ThreeStockK1BaselineError, match="unsupported stock"):
        evaluate_single_stock(
            CONFIG,
            root=ROOT,
            stock="kodak_gold_200",
            development=development[stock],
            confirmation=confirmation[stock],
        )
    duplicate_scene = list(confirmation[stock])
    victim = duplicate_scene[1]
    duplicate_scene[1] = StockFrameSamples(
        scene_id=duplicate_scene[0].scene_id,
        frame_id=victim.frame_id,
        roll_id=victim.roll_id,
        source=victim.source,
        target=victim.target,
    )
    with pytest.raises(ThreeStockK1BaselineError, match="scene identity"):
        evaluate_single_stock(
            CONFIG,
            root=ROOT,
            stock=stock,
            development=development[stock],
            confirmation=duplicate_scene,
        )


def test_confirmation_source_mismatch_is_invalid() -> None:
    development, confirmation = _population()
    victim = confirmation["kodak_ektar_100"][0]
    confirmation["kodak_ektar_100"][0] = StockFrameSamples(
        scene_id=victim.scene_id,
        frame_id=victim.frame_id,
        roll_id=victim.roll_id,
        source=np.clip(victim.source + 0.01, 0.0, 1.0),
        target=victim.target,
    )
    with pytest.raises(ThreeStockK1BaselineError, match="differ across stocks"):
        evaluate(
            CONFIG,
            root=ROOT,
            development=development,
            confirmation=confirmation,
        )


def test_development_confirmation_frame_leakage_is_invalid() -> None:
    development, confirmation = _population()
    victim = confirmation["kodak_portra_400"][0]
    confirmation["kodak_portra_400"][0] = StockFrameSamples(
        scene_id=victim.scene_id,
        frame_id=development["kodak_portra_400"][0].frame_id,
        roll_id=victim.roll_id,
        source=victim.source,
        target=victim.target,
    )
    with pytest.raises(ThreeStockK1BaselineError, match="frame leakage"):
        evaluate(
            CONFIG,
            root=ROOT,
            development=development,
            confirmation=confirmation,
        )
