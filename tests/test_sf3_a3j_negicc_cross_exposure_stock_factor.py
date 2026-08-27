from __future__ import annotations

import json

import numpy as np

from src.real_film.negicc_cross_exposure_factor import diagnose_contrasts

STOCKS = ["ektar", "portra"]
EXPOSURES = [-1, 0, 1, 2]


def _observations(*, reverse_last: bool = False) -> dict[tuple[str, int], np.ndarray]:
    base = np.asarray(
        [[-4.0, -3.0, -2.0], [-2.0, -2.5, -3.0], [-1.0, -1.5, -2.0]],
        dtype=np.float64,
    )
    contrast = np.asarray(
        [[0.4, -0.2, 0.1], [0.2, -0.1, 0.3], [-0.1, 0.3, 0.2]],
        dtype=np.float64,
    )
    rows = {}
    for exposure in EXPOSURES:
        drift = exposure * np.asarray([0.2, 0.1, 0.15])
        rows[("portra", exposure)] = base + drift
        sign = -1.0 if reverse_last and exposure == 2 else 1.0
        rows[("ektar", exposure)] = base + drift + sign * contrast
    return rows


def test_stable_stock_contrast_is_exact_across_exposure() -> None:
    report = diagnose_contrasts(
        _observations(), stocks=STOCKS, exposures=EXPOSURES, sign_floor=1e-6
    )
    assert len(report["pairs"]) == 6
    assert all(row["raw_cosine"] == 1.0 for row in report["pairs"])
    assert all(row["raw_sign_agreement"] == 1.0 for row in report["pairs"])
    assert report["raw_contrast_rms_coefficient_of_variation"] < 1e-12


def test_reversed_held_exposure_is_detected() -> None:
    report = diagnose_contrasts(
        _observations(reverse_last=True),
        stocks=STOCKS,
        exposures=EXPOSURES,
        sign_floor=1e-6,
    )
    held_pairs = [row for row in report["pairs"] if row["right_ev"] == 2]
    assert len(held_pairs) == 3
    assert all(row["raw_cosine"] < -0.999999 for row in held_pairs)
    assert all(row["raw_sign_agreement"] == 0.0 for row in held_pairs)


def test_diagnostic_payload_is_strict_json_serializable() -> None:
    report = diagnose_contrasts(
        _observations(), stocks=STOCKS, exposures=EXPOSURES, sign_floor=1e-6
    )
    assert json.loads(json.dumps(report, allow_nan=False)) == report
