"""Attribute the generic reversal appearance across fixed scanner stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.generic_reversal_development import _file_sha256
from src.eval.negative_route_photographic_stress import (
    _array_sha256,
    _load_source,
)
from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    GenericReversalDevelopment,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    develop_reversal_layer_exposure,
    prepare_interpretation_medium,
    reversal_development_contract,
    scan_interpretation_medium,
)


SCHEMA = "neuro_film.u6_p2h_reversal_scanner_stage_attribution.v1"
_EXPECTED_CHAIN = (
    ("direct_transmittance", ()),
    ("spectral", ("spectral",)),
    ("spectral_flare", ("spectral", "flare")),
    ("spectral_flare_dmax", ("spectral", "flare", "dmax")),
    (
        "spectral_flare_dmax_mtf",
        ("spectral", "flare", "dmax", "mtf"),
    ),
    (
        "spectral_flare_dmax_mtf_noise",
        ("spectral", "flare", "dmax", "mtf", "noise"),
    ),
)


def _statistics(values: np.ndarray, coefficients: np.ndarray) -> dict[str, Any]:
    mean_rgb = np.mean(values, axis=(0, 1), dtype=np.float64)
    luma = np.sum(values * coefficients, axis=-1)
    p05, p95 = np.percentile(luma, [5.0, 95.0])
    return {
        "output_sha256": _array_sha256(values),
        "mean_rgb": mean_rgb.tolist(),
        "mean_luma": float(np.mean(luma)),
        "median_luma": float(np.median(luma)),
        "luma_p95_minus_p05": float(p95 - p05),
        "mean_channel_spread": float(np.max(mean_rgb) - np.min(mean_rgb)),
    }


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2H contract")
    parents = contract["parents"]
    p2g_contract_path = root / parents["p2g_contract_path"]
    p2g_decision_path = root / parents["p2g_decision_path"]
    if _file_sha256(p2g_contract_path) != parents["p2g_contract_sha256"]:
        raise ValueError("P2G contract hash drift")
    if _file_sha256(p2g_decision_path) != parents["p2g_decision_sha256"]:
        raise ValueError("P2G decision hash drift")
    p2g = json.loads(p2g_contract_path.read_text(encoding="utf-8"))
    chain = tuple(
        (row["id"], tuple(row["scanner_stages"]))
        for row in contract["stage_chain"]
    )
    if chain != _EXPECTED_CHAIN:
        raise ValueError("scanner stage attribution chain drift")

    p2f = json.loads(
        (root / p2g["parents"]["p2f_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    template_path = root / p2f["parents"]["negative_sensitometry_path"]
    operator = GenericReversalDevelopment(
        build_operator(json.loads(template_path.read_text(encoding="utf-8"))),
        float(p2f["candidate"]["maximum_relative_layer_exposure"]),
    )
    route_contract = reversal_development_contract(operator)
    scanner_contract = json.loads(
        (root / p2g["parents"]["scanner_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    scanner = _profile(
        scanner_contract["profiles"][p2g["parents"]["scanner_profile"]]
    )
    manifest = json.loads(
        (root / p2g["input"]["manifest"]).read_text(encoding="utf-8")
    )
    if len(manifest) != int(contract["input"]["expected_rows"]):
        raise ValueError("attribution manifest row count drift")
    if len({row["make"] for row in manifest}) != int(
        contract["input"]["expected_camera_makes"]
    ):
        raise ValueError("attribution camera make count drift")
    coefficients = np.asarray(
        contract["metrics"]["luma_coefficients"], dtype=np.float32
    )
    pitch = float(p2g["pipeline"]["pixel_pitch_um"])
    scale = float(p2g["pipeline"]["pseudo_exposure_scale"])

    rows: list[dict[str, Any]] = []
    all_replays_exact = True
    all_finite_bounded = True
    for source_row in manifest:
        _, linear, input_sha = _load_source(source_row, root)
        exposure = PhysicalDomainArray(
            (linear * np.float32(scale)).astype(np.float32),
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            ("red", "green", "blue"),
        )
        developed = develop_reversal_layer_exposure(
            exposure, operator, route_contract
        )
        medium = prepare_interpretation_medium(developed)
        stage_rows: dict[str, dict[str, Any]] = {}
        stage_values: list[np.ndarray] = []
        for stage_id, stages in chain:
            if not stages:
                first = np.array(medium.values, copy=True)
                second = np.array(medium.values, copy=True)
            else:
                first = scan_interpretation_medium(
                    medium,
                    scanner,
                    pixel_pitch_um=pitch,
                    stages=stages,
                ).scan_linear.values
                second = scan_interpretation_medium(
                    medium,
                    scanner,
                    pixel_pitch_um=pitch,
                    stages=stages,
                ).scan_linear.values
            replay_exact = first.tobytes() == second.tobytes()
            finite_bounded = bool(
                np.all(np.isfinite(first))
                and np.all(first >= 0.0)
                and np.all(first <= 1.0)
            )
            statistics = _statistics(first, coefficients)
            statistics["replay_exact"] = replay_exact
            statistics["finite_bounded"] = finite_bounded
            stage_rows[stage_id] = statistics
            stage_values.append(first)
            all_replays_exact &= replay_exact
            all_finite_bounded &= finite_bounded
            del second
        successive = {}
        for index in range(1, len(chain)):
            previous_id = chain[index - 1][0]
            current_id = chain[index][0]
            previous = stage_values[index - 1]
            current = stage_values[index]
            previous_stats = stage_rows[previous_id]
            current_stats = stage_rows[current_id]
            mean_rgb_shift = (
                np.asarray(current_stats["mean_rgb"])
                - np.asarray(previous_stats["mean_rgb"])
            )
            successive[current_id] = {
                "from_stage": previous_id,
                "mean_abs_pixel_change": float(
                    np.mean(np.abs(current - previous))
                ),
                "mean_luma_shift": float(
                    current_stats["mean_luma"]
                    - previous_stats["mean_luma"]
                ),
                "mean_rgb_shift": mean_rgb_shift.tolist(),
                "channel_spread_shift": float(
                    current_stats["mean_channel_spread"]
                    - previous_stats["mean_channel_spread"]
                ),
            }
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "input_sha256": input_sha,
                "stages": stage_rows,
                "successive": successive,
            }
        )
        del stage_values

    stage_ids = tuple(item[0] for item in chain)
    population_stages = {
        stage_id: {
            "median_mean_rgb": np.median(
                np.asarray(
                    [row["stages"][stage_id]["mean_rgb"] for row in rows]
                ),
                axis=0,
            ).tolist(),
            "median_mean_luma": float(
                np.median(
                    [row["stages"][stage_id]["mean_luma"] for row in rows]
                )
            ),
            "median_median_luma": float(
                np.median(
                    [row["stages"][stage_id]["median_luma"] for row in rows]
                )
            ),
            "median_luma_p95_minus_p05": float(
                np.median(
                    [
                        row["stages"][stage_id]["luma_p95_minus_p05"]
                        for row in rows
                    ]
                )
            ),
            "median_mean_channel_spread": float(
                np.median(
                    [
                        row["stages"][stage_id]["mean_channel_spread"]
                        for row in rows
                    ]
                )
            ),
        }
        for stage_id in stage_ids
    }
    population_successive = {}
    for stage_id in stage_ids[1:]:
        population_successive[stage_id] = {
            "from_stage": rows[0]["successive"][stage_id]["from_stage"],
            "median_mean_abs_pixel_change": float(
                np.median(
                    [
                        row["successive"][stage_id][
                            "mean_abs_pixel_change"
                        ]
                        for row in rows
                    ]
                )
            ),
            "median_mean_luma_shift": float(
                np.median(
                    [
                        row["successive"][stage_id]["mean_luma_shift"]
                        for row in rows
                    ]
                )
            ),
            "median_mean_rgb_shift": np.median(
                np.asarray(
                    [
                        row["successive"][stage_id]["mean_rgb_shift"]
                        for row in rows
                    ]
                ),
                axis=0,
            ).tolist(),
            "median_channel_spread_shift": float(
                np.median(
                    [
                        row["successive"][stage_id]["channel_spread_shift"]
                        for row in rows
                    ]
                )
            ),
        }
    luma_owner = max(
        population_successive,
        key=lambda stage_id: abs(
            population_successive[stage_id]["median_mean_luma_shift"]
        ),
    )
    spread_owner = max(
        population_successive,
        key=lambda stage_id: abs(
            population_successive[stage_id][
                "median_channel_spread_shift"
            ]
        ),
    )
    gates = contract["automatic_gates"]
    minimum_material = float(
        gates["minimum_successive_stage_material_change"]
    )
    checks = {
        "parents_and_inputs": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "finite_bounded": all_finite_bounded,
        "replay": all_replays_exact,
        "stage_order": chain == _EXPECTED_CHAIN,
        "direct_bypass": all(
            row["stages"]["direct_transmittance"]["output_sha256"]
            == _array_sha256(
                prepare_interpretation_medium(
                    develop_reversal_layer_exposure(
                        PhysicalDomainArray(
                            _load_source(source, root)[1],
                            PhysicalDomain.LAYER_EXPOSURE,
                            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
                            ("red", "green", "blue"),
                        ),
                        operator,
                        route_contract,
                    )
                ).values
            )
            for row, source in zip(rows, manifest, strict=True)
        ),
        "successive_material": all(
            metrics["median_mean_abs_pixel_change"] >= minimum_material
            for metrics in population_successive.values()
        ),
    }
    passed = all(checks.values())
    report = {
        "schema": "neuro_film.u6_p2h_reversal_scanner_stage_attribution_report.v1",
        "node": contract["node"],
        "rows": rows,
        "population_stages": population_stages,
        "population_successive": population_successive,
        "attribution": {
            "largest_absolute_population_luma_shift": luma_owner,
            "largest_absolute_population_channel_spread_shift": spread_owner,
        },
        "checks": checks,
        "automatic_pass": passed,
        "branch": contract["branch_rules"][
            "automatic_pass" if passed else "automatic_fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return report


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
