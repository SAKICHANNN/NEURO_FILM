"""Photographic stress for normalized generic reversal interpretation."""

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
    _preview,
)
from src.eval.physical_scanner_profile import _profile
from src.eval.reversal_photographic_stress import _contact_sheet
from src.eval.sensitometry_primitive import build_operator
from src.eval.density_witness_frontier import linear_srgb_to_encoded
from src.film_physics import (
    GenericReversalDevelopment,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    derive_scan_signal_normalization,
    develop_reversal_layer_exposure,
    prepare_interpretation_medium,
    reversal_development_contract,
    scan_interpretation_medium,
    scan_signal_normalization_identity,
)


SCHEMA = "neuro_film.u6_p2j_normalized_reversal_photographic_stress.v1"


def evaluate(
    contract: dict[str, Any],
    *,
    root: Path,
    contact_sheet_path: Path,
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2J contract")
    parents = contract["parents"]
    for path_key, hash_key in (
        ("p2g_contract_path", "p2g_contract_sha256"),
        ("p2i_contract_path", "p2i_contract_sha256"),
        ("p2i_decision_path", "p2i_decision_sha256"),
    ):
        if _file_sha256(root / parents[path_key]) != parents[hash_key]:
            raise ValueError(f"{path_key} hash drift")
    p2g = json.loads(
        (root / parents["p2g_contract_path"]).read_text(encoding="utf-8")
    )
    p2i = json.loads(
        (root / parents["p2i_contract_path"]).read_text(encoding="utf-8")
    )
    p2f = json.loads(
        (root / p2g["parents"]["p2f_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    template_path = root / p2f["parents"]["negative_sensitometry_path"]
    reversal = GenericReversalDevelopment(
        build_operator(json.loads(template_path.read_text(encoding="utf-8"))),
        float(p2f["candidate"]["maximum_relative_layer_exposure"]),
    )
    route_contract = reversal_development_contract(reversal)
    scanner_contract = json.loads(
        (root / p2g["parents"]["scanner_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    scanner = _profile(
        scanner_contract["profiles"][p2g["parents"]["scanner_profile"]]
    )
    derivation = p2i["endpoint_derivation"]
    normalization = derive_scan_signal_normalization(
        reversal,
        scanner,
        flat_field_shape=tuple(derivation["flat_field_shape"]),
        scanner_stages=tuple(derivation["scanner_stages"]),
    )
    if (
        scan_signal_normalization_identity(normalization)
        != parents["normalization_sha256"]
    ):
        raise ValueError("scan normalization identity drift")
    stages = tuple(contract["pipeline"]["scanner_stages"])
    if stages != tuple(derivation["scanner_stages"]) or "noise" in stages:
        raise ValueError("photographic scanner stages drift")

    manifest = json.loads(
        (root / p2g["input"]["manifest"]).read_text(encoding="utf-8")
    )
    expected = contract["input"]
    if len(manifest) != int(expected["expected_rows"]):
        raise ValueError("photographic manifest row count drift")
    if len({row["make"] for row in manifest}) != int(
        expected["expected_camera_makes"]
    ):
        raise ValueError("photographic camera-make count drift")
    gates = contract["automatic_gates"]
    scale = float(contract["pipeline"]["pseudo_exposure_scale"])
    pitch = float(p2g["pipeline"]["pixel_pitch_um"])
    black = np.asarray(normalization.black_scan_rgb)
    white = np.asarray(normalization.white_scan_rgb)
    rows: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    for source_row in manifest:
        encoded, linear, input_sha = _load_source(source_row, root)
        exposure = PhysicalDomainArray(
            (linear * np.float32(scale)).astype(np.float32),
            PhysicalDomain.LAYER_EXPOSURE,
            PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
            ("red", "green", "blue"),
        )
        developed = develop_reversal_layer_exposure(
            exposure, reversal, route_contract
        )
        medium = prepare_interpretation_medium(developed)
        first_scan = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=pitch, stages=stages
        ).scan_linear
        second_scan = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=pitch, stages=stages
        ).scan_linear
        scan_replay_exact = (
            first_scan.values.tobytes() == second_scan.values.tobytes()
        )
        escape = (first_scan.values < black) | (first_scan.values > white)
        escape_fraction = float(np.mean(escape))
        if escape_fraction > 0.0:
            rows.append(
                {
                    "id": source_row["id"],
                    "make": source_row["make"],
                    "input_sha256": input_sha,
                    "shape": list(linear.shape),
                    "endpoint_domain_escape_fraction": escape_fraction,
                    "scan_replay_exact": scan_replay_exact,
                    "eligible": False,
                }
            )
            continue
        first = normalization.apply(first_scan).values
        second = normalization.apply(second_scan).values
        repeat_exact = first.tobytes() == second.tobytes()
        boundary = (
            ((first <= 0.0) | (first >= 1.0))
            & ~((linear <= 0.0) | (linear >= 1.0))
        )
        luma = (
            np.float32(0.2126) * first[..., 0]
            + np.float32(0.7152) * first[..., 1]
            + np.float32(0.0722) * first[..., 2]
        )
        p05, p95 = np.percentile(luma, [5.0, 95.0])
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "input_sha256": input_sha,
                "shape": list(linear.shape),
                "endpoint_domain_escape_fraction": escape_fraction,
                "scan_replay_exact": scan_replay_exact,
                "eligible": True,
                "output_sha256": _array_sha256(first),
                "repeat_exact": repeat_exact,
                "finite_bounded": bool(
                    np.all(np.isfinite(first))
                    and np.all(first >= 0.0)
                    and np.all(first <= 1.0)
                ),
                "mean_abs_change": float(np.mean(np.abs(first - linear))),
                "new_boundary_fraction": float(np.mean(boundary)),
                "near_black_fraction": float(
                    np.mean(luma <= float(gates["near_black_threshold"]))
                ),
                "near_white_fraction": float(
                    np.mean(luma >= float(gates["near_white_threshold"]))
                ),
                "luma_p95_minus_p05": float(p95 - p05),
            }
        )
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visuals.append(
                {
                    "id": source_row["id"],
                    "previews": (
                        _preview(encoded, (420, 276)),
                        _preview(
                            linear_srgb_to_encoded(first), (420, 276)
                        ),
                    ),
                }
            )
    eligible = [row for row in rows if row["eligible"]]
    all_eligible = len(eligible) == len(rows)
    if all_eligible:
        fixed_ids = contract["visual_protocol"]["fixed_ids"]
        if {row["id"] for row in visuals} != set(fixed_ids):
            raise ValueError("fixed visual IDs are incomplete")
        contact_sha: str | None = _contact_sheet(
            visuals, fixed_ids, contact_sheet_path
        )
        population = {
            "median_mean_abs_change": float(
                np.median([row["mean_abs_change"] for row in eligible])
            ),
            "maximum_new_boundary_fraction": max(
                row["new_boundary_fraction"] for row in eligible
            ),
            "maximum_near_black_fraction": max(
                row["near_black_fraction"] for row in eligible
            ),
            "maximum_near_white_fraction": max(
                row["near_white_fraction"] for row in eligible
            ),
            "median_near_black_fraction": float(
                np.median(
                    [row["near_black_fraction"] for row in eligible]
                )
            ),
            "median_near_white_fraction": float(
                np.median(
                    [row["near_white_fraction"] for row in eligible]
                )
            ),
            "minimum_luma_p95_minus_p05": min(
                row["luma_p95_minus_p05"] for row in eligible
            ),
        }
    else:
        contact_sha = None
        population = None
    maximum_escape = max(
        row["endpoint_domain_escape_fraction"] for row in rows
    )
    decisions = {
        "parents_and_inputs": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "endpoint_domain": maximum_escape
        <= float(gates["maximum_endpoint_domain_escape_fraction"]),
        "finite_bounded": all_eligible
        and all(row["finite_bounded"] for row in eligible),
        "repeat": all_eligible
        and all(
            row["scan_replay_exact"] and row["repeat_exact"]
            for row in eligible
        ),
        "material": all_eligible
        and population["median_mean_abs_change"]
        >= float(gates["minimum_population_median_mean_abs_change"]),
        "boundary": all_eligible
        and population["maximum_new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"]),
        "per_image_near_black": all_eligible
        and population["maximum_near_black_fraction"]
        <= float(gates["maximum_per_image_near_black_fraction"]),
        "per_image_near_white": all_eligible
        and population["maximum_near_white_fraction"]
        <= float(gates["maximum_per_image_near_white_fraction"]),
        "population_near_black": all_eligible
        and population["median_near_black_fraction"]
        <= float(gates["maximum_population_median_near_black_fraction"]),
        "population_near_white": all_eligible
        and population["median_near_white_fraction"]
        <= float(gates["maximum_population_median_near_white_fraction"]),
        "photographic_luma_range": all_eligible
        and population["minimum_luma_p95_minus_p05"]
        >= float(gates["minimum_per_image_luma_p95_minus_p05"]),
    }
    passed = all(decisions.values())
    report = {
        "schema": "neuro_film.u6_p2j_normalized_reversal_photographic_stress_report.v1",
        "node": contract["node"],
        "normalization_sha256": scan_signal_normalization_identity(
            normalization
        ),
        "rows": rows,
        "population": population,
        "maximum_endpoint_domain_escape_fraction": maximum_escape,
        "contact_sheet_sha256": contact_sha,
        "decisions": decisions,
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
