"""Cross-exposure stock-factor diagnostic for the frozen NegICC observations."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from src.real_film.negicc_it8_source import _canonical_bytes, _sha256
from src.real_film.negicc_stock_k1 import _load_train, _target


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left.ravel(), right.ravel()) / denominator)


def _sign_agreement(left: np.ndarray, right: np.ndarray, floor: float) -> float:
    mask = np.maximum(np.abs(left), np.abs(right)) >= floor
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.signbit(left[mask]) == np.signbit(right[mask])))


def diagnose_contrasts(
    observations: dict[tuple[str, int], np.ndarray],
    *,
    stocks: list[str],
    exposures: list[int],
    sign_floor: float,
) -> dict[str, Any]:
    if len(stocks) != 2:
        raise ValueError("cross-exposure diagnostic requires exactly two stocks")
    first, second = stocks
    contrasts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    exposure_rows = []
    for exposure in exposures:
        first_rgb = observations[(first, exposure)]
        second_rgb = observations[(second, exposure)]
        if first_rgb.shape != second_rgb.shape or first_rgb.ndim != 2:
            raise ValueError("stock observations must have identical 2-D shapes")
        raw = first_rgb - second_rgb
        first_centered = first_rgb - np.mean(first_rgb, axis=0, keepdims=True)
        second_centered = second_rgb - np.mean(second_rgb, axis=0, keepdims=True)
        centered = first_centered - second_centered
        contrasts[exposure] = (raw, centered)
        exposure_rows.append(
            {
                "exposure_ev": exposure,
                "patch_rows": int(first_rgb.shape[0]),
                "raw_contrast_rms": float(np.sqrt(np.mean(raw * raw))),
                "centered_contrast_rms": float(np.sqrt(np.mean(centered * centered))),
            }
        )

    pair_rows = []
    for left_ev, right_ev in itertools.combinations(exposures, 2):
        left_raw, left_centered = contrasts[left_ev]
        right_raw, right_centered = contrasts[right_ev]
        pair_rows.append(
            {
                "left_ev": left_ev,
                "right_ev": right_ev,
                "raw_cosine": _cosine(left_raw, right_raw),
                "centered_cosine": _cosine(left_centered, right_centered),
                "raw_sign_agreement": _sign_agreement(left_raw, right_raw, sign_floor),
                "centered_sign_agreement": _sign_agreement(
                    left_centered, right_centered, sign_floor
                ),
            }
        )
    raw_rms = np.asarray(
        [row["raw_contrast_rms"] for row in exposure_rows], dtype=np.float64
    )
    return {
        "exposures": exposure_rows,
        "pairs": pair_rows,
        "raw_contrast_rms_coefficient_of_variation": float(
            np.std(raw_rms) / max(float(np.mean(raw_rms)), 1e-12)
        ),
    }


def run_diagnostic(
    config: dict[str, Any], source_config: dict[str, Any], *, reverse: bool = False
) -> dict[str, Any]:
    stocks = sorted(config["stocks"])
    exposures = sorted(config["exposures_ev"])
    fetch_stocks = list(reversed(stocks)) if reverse else stocks
    fetch_exposures = list(reversed(exposures)) if reverse else exposures
    observations: dict[tuple[str, int], np.ndarray] = {}
    source_facts = []
    canonical_patches: list[str] | None = None
    canonical_xyz: np.ndarray | None = None
    patch_identity_exact = True
    reference_xyz_exact = True
    for stock_id in fetch_stocks:
        for exposure in fetch_exposures:
            patches, scanner_rgb, reference_xyz, fact = _load_train(
                source_config, stock_id, exposure
            )
            if canonical_patches is None:
                canonical_patches = patches
                canonical_xyz = reference_xyz
            else:
                patch_identity_exact &= patches == canonical_patches
                reference_xyz_exact &= np.array_equal(reference_xyz, canonical_xyz)
            observations[(stock_id, exposure)] = _target(scanner_rgb)
            source_facts.append(fact)

    diagnostic = diagnose_contrasts(
        observations,
        stocks=stocks,
        exposures=exposures,
        sign_floor=float(config["representation"]["sign_agreement_floor"]),
    )
    gates = config["gates"]
    exposure_rows = diagnostic["exposures"]
    pair_rows = diagnostic["pairs"]
    finite = all(
        np.isfinite(value)
        for row in exposure_rows + pair_rows
        for key, value in row.items()
        if key not in {"exposure_ev", "left_ev", "right_ev", "patch_rows"}
    ) and np.isfinite(diagnostic["raw_contrast_rms_coefficient_of_variation"])
    gate_results = {
        "patch_identity_exact": bool(patch_identity_exact),
        "reference_xyz_exact": bool(reference_xyz_exact),
        "patch_rows": all(
            row["patch_rows"] == gates["required_patch_rows_per_stock_exposure"]
            for row in exposure_rows
        ),
        "exposure_pairs": len(pair_rows) == gates["required_exposure_pairs"],
        "raw_pairwise_cosine": min(row["raw_cosine"] for row in pair_rows)
        >= gates["raw_pairwise_cosine_min"],
        "centered_pairwise_cosine": min(row["centered_cosine"] for row in pair_rows)
        >= gates["centered_pairwise_cosine_min"],
        "raw_sign_agreement": min(row["raw_sign_agreement"] for row in pair_rows)
        >= gates["raw_sign_agreement_min"],
        "centered_sign_agreement": min(
            row["centered_sign_agreement"] for row in pair_rows
        )
        >= gates["centered_sign_agreement_min"],
        "contrast_rms": min(row["raw_contrast_rms"] for row in exposure_rows)
        >= gates["contrast_rms_min"],
        "contrast_rms_coefficient_of_variation": diagnostic[
            "raw_contrast_rms_coefficient_of_variation"
        ]
        <= gates["contrast_rms_coefficient_of_variation_max"],
        "finite": finite,
        "image_or_tiff_requests": gates["image_or_tiff_requests_max"] == 0,
    }
    source_facts.sort(key=lambda fact: fact["path"])
    scientific_payload = {
        "source_lock_scientific_identity": config["source_lock"]["scientific_identity"],
        "source_facts": source_facts,
        "diagnostic": diagnostic,
        "gates": gate_results,
    }
    passed = all(gate_results.values())
    return {
        "schema": "neuro-film.sf3-a3j-negicc-cross-exposure-stock-factor-result.v1",
        "experiment_id": config["experiment_id"],
        "status": "PASS_RETROSPECTIVE_STOCK_FACTOR_STABILITY"
        if passed
        else "FAIL_CLOSED_RETROSPECTIVE_STOCK_FACTOR_STABILITY",
        "decision": "REQUIRE_INDEPENDENT_SOURCE_CONFIRMATION"
        if passed
        else "REQUIRE_NEW_CONTROLLED_OBSERVATIONS_NOT_OPERATOR_CAPACITY",
        "scientific_payload": scientific_payload,
        "scientific_identity": _sha256(_canonical_bytes(scientific_payload)),
        "network_reads": len(source_facts),
        "image_or_tiff_requests": 0,
        "pixel_decodes": 0,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["diagnose_contrasts", "run_diagnostic"]
