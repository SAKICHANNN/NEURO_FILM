from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.real_film.three_stock_k1_baseline import evaluate, load_contract
from src.real_film.three_stock_paired_sampling import (
    AlignedScanRow,
    ThreeStockPairedSamplingError,
    extract_common_paired_samples,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sf3_a2_three_stock_k1_baseline_v1.json"
STOCKS = (
    "fujifilm_velvia_50",
    "kodak_portra_400",
    "kodak_ektar_100",
)


def _rows(*, mismatch: bool = False, translation: float = 0.0) -> list[AlignedScanRow]:
    transforms = {
        STOCKS[0]: (np.array([1.12, 0.98, 0.90]), np.array([0.01, 0.01, 0.03])),
        STOCKS[1]: (np.array([0.94, 1.03, 1.06]), np.array([0.04, 0.02, 0.00])),
        STOCKS[2]: (np.array([1.07, 0.93, 1.11]), np.array([0.00, 0.04, 0.01])),
    }
    rows: list[AlignedScanRow] = []
    for role, scene_count, roll_count in (
        ("development", 4, 2),
        ("confirmation", 4, 1),
    ):
        for scene_index in range(scene_count):
            source = np.random.default_rng(
                scene_index + (0 if role == "development" else 100)
            ).integers(30, 170, size=(80, 96, 3), dtype=np.uint8)
            for stock_index, stock in enumerate(STOCKS):
                for roll_index in range(roll_count):
                    gain, bias = transforms[stock]
                    target = np.rint(
                        np.clip(
                            source.astype(np.float64) / 255.0 * gain + bias, 0.0, 1.0
                        )
                        * 255.0
                    ).astype(np.uint8)
                    row_source = source.copy()
                    if (
                        mismatch
                        and role == "confirmation"
                        and scene_index == 0
                        and stock_index == 2
                    ):
                        row_source[0, 0, 0] ^= 1
                    rows.append(
                        AlignedScanRow(
                            row_id=f"{stock}-{role}-{roll_index}-{scene_index}",
                            stock_id=stock,
                            role=role,
                            scene_id=f"{role}-scene-{scene_index}",
                            film_frame_id=f"{stock}-{role}-{roll_index}-{scene_index}",
                            roll_id=f"{stock}-{role}-roll-{roll_index}",
                            source_rgb=row_source,
                            scan_rgb=target,
                            homography_source_to_scan=np.array(
                                [
                                    [1.0, 0.0, translation],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                ]
                            ),
                        )
                    )
    return rows


def test_common_sampling_preserves_identical_sources_and_feeds_k1() -> None:
    _, contract = load_contract(CONFIG, root=ROOT)
    development, confirmation, facts = extract_common_paired_samples(
        _rows(), contract["paired_sampling"]
    )
    assert facts["row_count"] == 36
    assert facts["scene_count"] == 8
    assert facts["minimum_observed_shared_valid_fraction"] == 1.0
    for scene_index in range(4):
        samples = [
            next(
                frame.source
                for frame in confirmation[stock]
                if frame.scene_id == f"confirmation-scene-{scene_index}"
            )
            for stock in STOCKS
        ]
        assert np.array_equal(samples[0], samples[1])
        assert np.array_equal(samples[0], samples[2])
    report = evaluate(
        CONFIG,
        root=ROOT,
        development=development,
        confirmation=confirmation,
    )
    assert report["automatic_pass"] is True


def test_source_pixel_drift_is_invalid() -> None:
    _, contract = load_contract(CONFIG, root=ROOT)
    with pytest.raises(ThreeStockPairedSamplingError, match="source pixels differ"):
        extract_common_paired_samples(_rows(mismatch=True), contract["paired_sampling"])


def test_insufficient_shared_scan_support_is_invalid() -> None:
    _, contract = load_contract(CONFIG, root=ROOT)
    with pytest.raises(ThreeStockPairedSamplingError, match="shared valid support"):
        extract_common_paired_samples(
            _rows(translation=80.0), contract["paired_sampling"]
        )


def test_sampling_is_exactly_replayable() -> None:
    _, contract = load_contract(CONFIG, root=ROOT)
    first = extract_common_paired_samples(
        _rows(translation=2.0), contract["paired_sampling"]
    )
    second = extract_common_paired_samples(
        _rows(translation=2.0), contract["paired_sampling"]
    )
    assert first[2] == second[2]
    for partition in (0, 1):
        for stock in STOCKS:
            for left, right in zip(
                first[partition][stock], second[partition][stock], strict=True
            ):
                assert np.array_equal(left.source, right.source)
                assert np.array_equal(left.target, right.target)
