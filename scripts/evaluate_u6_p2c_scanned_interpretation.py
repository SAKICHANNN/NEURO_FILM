#!/usr/bin/env python
"""Evaluate the frozen U6.P2C route/scanner/polarity ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_scanner_profile import _profile  # noqa: E402
from src.eval.sensitometry_primitive import build_operator  # noqa: E402
from src.film_physics import (  # noqa: E402
    DevelopedExposureResult,
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationRoute,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    apply_post_scan_polarity,
    apply_scanner_profile,
    compile_print_interpretation,
    develop_layer_exposure,
    prepare_interpretation_medium,
    scan_interpretation_medium,
)


def _sha(values: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(values)).cast("B")
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    u2 = json.loads(
        (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    sensitometry = build_operator(u2)
    witness = json.loads(
        (
            ROOT / "configs/u5_r2e0_density_domain_operator_v1.json"
        ).read_text(encoding="utf-8")
    )["witnesses"]["cyan_shadow_warm_highlight_like"]
    print_operator = compile_print_interpretation(
        sensitometry,
        dye_absorption_matrix=np.asarray(witness["dye_absorption_matrix"]),
        print_matrix=np.asarray(witness["print_matrix"]),
        paper_midpoints=np.asarray(witness["paper_midpoints"]),
        paper_slopes=np.asarray(witness["paper_slopes"]),
        paper_maximum_densities=np.asarray(
            witness["paper_maximum_densities"]
        ),
        maximum_relative_layer_exposure=16.0,
    )
    scanner_contract = json.loads(
        (
            ROOT / config["parents"]["scanner_contract_path"]
        ).read_text(encoding="utf-8")
    )
    scanner = _profile(
        scanner_contract["profiles"][
            config["parents"]["scanner_profile"]
        ]
    )
    identity_scanner = _profile(scanner_contract["profiles"]["identity"])
    shape = tuple(config["synthetic_input"]["shape"])
    height, width, _ = shape
    y, x = np.mgrid[:height, :width]
    values = np.stack(
        (
            16.0 * x / (width - 1),
            16.0 * y / (height - 1),
            8.0 * (x / (width - 1) + y / (height - 1)),
        ),
        axis=-1,
    ).astype(np.float32)
    source = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )
    source_sha = _sha(source.values)
    pitch = float(config["synthetic_input"]["pixel_pitch_um"])

    specifications = (
        (
            "negative",
            EmulsionFamily.COLOR_NEGATIVE,
            InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
            None,
        ),
        (
            "print",
            EmulsionFamily.COLOR_NEGATIVE,
            InterpretationRoute.COLOR_NEGATIVE_PRINT,
            print_operator,
        ),
        (
            "slide",
            EmulsionFamily.SLIDE,
            InterpretationRoute.SLIDE_DIRECT_SCAN,
            None,
        ),
    )
    media = {}
    results = {}
    rows = []
    for name, family, route, print_arg in specifications:
        contract = DevelopmentInterpretationContract.for_operator(
            sensitometry,
            emulsion_family=family,
            interpretation_route=route,
        )
        developed = develop_layer_exposure(source, sensitometry, contract)
        medium = prepare_interpretation_medium(
            developed, print_interpretation=print_arg
        )
        first = scan_interpretation_medium(
            medium,
            scanner,
            pixel_pitch_um=pitch,
            expected_route=route,
        )
        second = scan_interpretation_medium(
            medium,
            scanner,
            pixel_pitch_um=pitch,
            expected_route=route,
        )
        media[name] = medium
        results[name] = first
        rows.append(
            {
                "name": name,
                "medium_sha256": _sha(medium.values),
                "output_sha256": _sha(first.scan_linear.values),
                "output_minimum": float(np.min(first.scan_linear.values)),
                "output_maximum": float(np.max(first.scan_linear.values)),
                "repeat_exact": bool(
                    first.scan_linear.values.tobytes()
                    == second.scan_linear.values.tobytes()
                    and first.descriptor() == second.descriptor()
                ),
                "order": list(first.order),
            }
        )

    wrong_order = apply_scanner_profile(
        apply_post_scan_polarity(
            media["negative"].values,
            media["negative"].post_scan_polarity,
        ),
        scanner,
        pixel_pitch_um=pitch,
    )
    correct_wrong_order_delta = float(
        np.max(
            np.abs(
                results["negative"].scan_linear.values - wrong_order
            )
        )
    )
    negative_slide_delta = float(
        np.max(
            np.abs(
                results["negative"].scan_linear.values
                - results["slide"].scan_linear.values
            )
        )
    )
    print_negative_delta = float(
        np.max(
            np.abs(
                results["print"].scan_linear.values
                - results["negative"].scan_linear.values
            )
        )
    )
    identity_negative = scan_interpretation_medium(
        media["negative"], identity_scanner, pixel_pitch_um=pitch
    ).scan_linear.values
    identity_slide = scan_interpretation_medium(
        media["slide"], identity_scanner, pixel_pitch_um=pitch
    ).scan_linear.values
    identity_complement_error = float(
        np.max(np.abs(identity_negative + identity_slide - 1.0))
    )

    bw_contract = DevelopmentInterpretationContract.for_operator(
        sensitometry,
        emulsion_family=EmulsionFamily.BLACK_AND_WHITE,
        interpretation_route=InterpretationRoute.BW_DEVELOPER_SCAN,
    )
    colour_bw = develop_layer_exposure(source, sensitometry, bw_contract)
    try:
        prepare_interpretation_medium(colour_bw)
        non_neutral_bw_rejected = False
    except ValueError:
        non_neutral_bw_rejected = True
    neutral = np.repeat(
        colour_bw.density.values[..., 1:2], 3, axis=-1
    ).astype(np.float32)
    neutral_density = PhysicalDomainArray(
        neutral,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        ("red", "green", "blue"),
    )
    bw_medium = prepare_interpretation_medium(
        DevelopedExposureResult(neutral_density, bw_contract)
    )
    bw_result = scan_interpretation_medium(
        bw_medium, scanner, pixel_pitch_um=pitch
    )
    rows.append(
        {
            "name": "bw_neutral",
            "medium_sha256": _sha(bw_medium.values),
            "output_sha256": _sha(bw_result.scan_linear.values),
            "output_minimum": float(np.min(bw_result.scan_linear.values)),
            "output_maximum": float(np.max(bw_result.scan_linear.values)),
            "repeat_exact": True,
            "order": list(bw_result.order),
        }
    )
    try:
        scan_interpretation_medium(
            media["print"],
            scanner,
            pixel_pitch_um=pitch,
            expected_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
        )
        wrong_route_rejected = False
    except ValueError:
        wrong_route_rejected = True

    output_boundary = np.concatenate(
        [
            result.scan_linear.values.reshape(-1, 3)
            for result in results.values()
        ],
        axis=0,
    )
    new_boundary_fraction = float(
        np.mean((output_boundary <= 0.0) | (output_boundary >= 1.0))
    )
    gates = config["automatic_gates"]
    checks = {
        "route_count": len(rows) == 4,
        "repeat": all(row["repeat_exact"] for row in rows),
        "input_preservation": _sha(source.values) == source_sha,
        "negative_slide_medium_equal": (
            media["negative"].values.tobytes()
            == media["slide"].values.tobytes()
        ),
        "negative_slide_final": negative_slide_delta
        >= float(gates["minimum_negative_vs_slide_final_max_abs"]),
        "print_negative_final": print_negative_delta
        >= float(gates["minimum_print_vs_negative_final_max_abs"]),
        "order_control": correct_wrong_order_delta
        >= float(
            gates[
                "minimum_correct_vs_polarity_before_scanner_max_abs"
            ]
        ),
        "boundary": new_boundary_fraction
        <= float(gates["maximum_new_boundary_fraction"]),
        "identity_complement": identity_complement_error
        <= float(
            gates["identity_scanner_negative_plus_slide_max_error"]
        ),
        "bw_guard": non_neutral_bw_rejected,
        "route_guard": wrong_route_rejected,
    }
    core = {
        "schema": config["schema"].replace("ablation.v1", "report.v1"),
        "node": config["node"],
        "source_sha256": source_sha,
        "routes": rows,
        "metrics": {
            "negative_vs_slide_final_max_abs": negative_slide_delta,
            "print_vs_negative_final_max_abs": print_negative_delta,
            "correct_vs_polarity_before_scanner_max_abs": (
                correct_wrong_order_delta
            ),
            "identity_negative_plus_slide_max_error": (
                identity_complement_error
            ),
            "new_boundary_fraction": new_boundary_fraction,
        },
        "guards": {
            "non_neutral_bw_rejected": non_neutral_bw_rejected,
            "wrong_route_rejected": wrong_route_rejected,
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }
    core["evidence_id"] = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p2c_scanned_interpretation_ablation_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.config.read_text(encoding="utf-8")))
    _atomic_json(args.output, report)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"evidence_id={report['evidence_id']}")
    print(json.dumps(report["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
