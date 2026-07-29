#!/usr/bin/env python
"""Evaluate neutral exposure polarity for the frozen U6.P2 routes."""

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
    compile_print_interpretation,
    develop_layer_exposure,
    prepare_interpretation_medium,
    scan_interpretation_medium,
)


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
            ROOT / config["scanner"]["contract_path"]
        ).read_text(encoding="utf-8")
    )
    scanner = _profile(
        scanner_contract["profiles"][config["scanner"]["profile"]]
    )
    ramp = config["neutral_exposure_ramp"]
    samples = np.linspace(
        float(ramp["minimum"]),
        float(ramp["maximum"]),
        int(ramp["samples"]),
        dtype=np.float32,
    )
    values = np.repeat(samples[:, None, None], 3, axis=2)
    source = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )
    gates = config["automatic_gates"]
    specifications = (
        (
            "color_negative_neutral_scan",
            EmulsionFamily.COLOR_NEGATIVE,
            InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
            None,
            False,
        ),
        (
            "color_negative_print",
            EmulsionFamily.COLOR_NEGATIVE,
            InterpretationRoute.COLOR_NEGATIVE_PRINT,
            print_operator,
            False,
        ),
        (
            "slide_direct_scan",
            EmulsionFamily.SLIDE,
            InterpretationRoute.SLIDE_DIRECT_SCAN,
            None,
            False,
        ),
        (
            "bw_developer_scan",
            EmulsionFamily.BLACK_AND_WHITE,
            InterpretationRoute.BW_DEVELOPER_SCAN,
            None,
            True,
        ),
    )
    rows = []
    non_neutral_bw_rejected = False
    for name, family, route, print_arg, force_neutral in specifications:
        contract = DevelopmentInterpretationContract.for_operator(
            sensitometry,
            emulsion_family=family,
            interpretation_route=route,
        )
        developed = develop_layer_exposure(source, sensitometry, contract)
        if force_neutral:
            try:
                prepare_interpretation_medium(developed)
            except ValueError:
                non_neutral_bw_rejected = True
            neutral = np.repeat(
                developed.density.values[..., 1:2], 3, axis=-1
            ).astype(np.float32)
            developed = DevelopedExposureResult(
                PhysicalDomainArray(
                    neutral,
                    PhysicalDomain.DEVELOPED_DENSITY,
                    PhysicalUnit.OPTICAL_DENSITY,
                    ("red", "green", "blue"),
                ),
                contract,
            )
        medium = prepare_interpretation_medium(
            developed, print_interpretation=print_arg
        )
        first = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=1.0
        ).scan_linear.values[:, 0, :]
        second = scan_interpretation_medium(
            medium, scanner, pixel_pitch_um=1.0
        ).scan_linear.values[:, 0, :]
        endpoint = first[-1] - first[0]
        differences = np.diff(first, axis=0)
        channel_spread = np.ptp(first, axis=1)
        checks = {
            "endpoint": bool(
                np.all(
                    endpoint
                    >= float(gates["minimum_endpoint_increase"])
                )
            ),
            "monotone": float(np.min(differences))
            >= float(gates["minimum_first_difference"]),
            "channel_spread": float(np.max(channel_spread))
            <= float(gates["maximum_channel_spread"]),
            "repeat": first.tobytes() == second.tobytes(),
        }
        rows.append(
            {
                "route": name,
                "endpoint_delta_rgb": endpoint.astype(
                    np.float64
                ).tolist(),
                "minimum_first_difference": float(np.min(differences)),
                "maximum_channel_spread": float(np.max(channel_spread)),
                "output_sha256": hashlib.sha256(
                    memoryview(np.ascontiguousarray(first)).cast("B")
                ).hexdigest(),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    failed = [row["route"] for row in rows if not row["passed"]]
    if not failed:
        branch = "all_pass"
    elif failed == ["slide_direct_scan"]:
        branch = "slide_only_fail"
    elif "bw_developer_scan" in failed and len(failed) == 1:
        branch = "bw_fail"
    else:
        branch = "negative_or_print_fail"
    core = {
        "schema": "neuro_film.u6_p2c1_route_exposure_polarity_report.v1",
        "node": config["node"],
        "rows": rows,
        "non_neutral_bw_rejected": non_neutral_bw_rejected,
        "failed_routes": failed,
        "automatic_pass": not failed and non_neutral_bw_rejected,
        "branch": branch,
        "branch_action": config["branch_rules"][branch],
        "claim_ceiling": config["claim_ceiling"],
    }
    core["stable_evidence_id"] = hashlib.sha256(
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
        default=ROOT
        / "configs/u6_p2c1_route_exposure_polarity_audit_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.config.read_text(encoding="utf-8")))
    _atomic_json(args.output, report)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"branch={report['branch']}")
    print(f"failed_routes={report['failed_routes']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")


if __name__ == "__main__":
    main()
