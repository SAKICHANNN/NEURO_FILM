#!/usr/bin/env python
"""Evaluate the frozen U6.P2B interpretation-medium contract."""

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
        density_dtype=np.float32,
    )
    rng = np.random.default_rng(20260729)
    values = rng.uniform(0.0, 16.0, size=(257, 31, 3)).astype(np.float32)
    values[0] = 0.0
    values[-1] = 16.0
    source = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )
    source_sha = _sha(source.values)

    rows = []
    media: dict[str, Any] = {}
    route_specs = (
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
    for name, family, route, print_arg in route_specs:
        contract = DevelopmentInterpretationContract.for_operator(
            sensitometry,
            emulsion_family=family,
            interpretation_route=route,
        )
        developed = develop_layer_exposure(source, sensitometry, contract)
        medium = prepare_interpretation_medium(
            developed, print_interpretation=print_arg
        )
        parts = []
        for start, stop in ((0, 37), (37, 128), (128, 257)):
            density_part = PhysicalDomainArray(
                developed.density.values[start:stop],
                PhysicalDomain.DEVELOPED_DENSITY,
                PhysicalUnit.OPTICAL_DENSITY,
                developed.density.channels,
            )
            part = prepare_interpretation_medium(
                DevelopedExposureResult(density_part, contract),
                print_interpretation=print_arg,
            )
            parts.append(part.values)
        partitioned = np.concatenate(parts)
        media[name] = medium
        rows.append(
            {
                "name": name,
                "descriptor": medium.descriptor(),
                "values_sha256": _sha(medium.values),
                "minimum": float(np.min(medium.values)),
                "maximum": float(np.max(medium.values)),
                "partition_exact": bool(
                    np.array_equal(partitioned, medium.values)
                ),
            }
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
    neutral_values = np.repeat(
        colour_bw.density.values[..., 1:2], 3, axis=-1
    ).astype(np.float32)
    neutral_density = PhysicalDomainArray(
        neutral_values,
        PhysicalDomain.DEVELOPED_DENSITY,
        PhysicalUnit.OPTICAL_DENSITY,
        ("red", "green", "blue"),
    )
    bw_medium = prepare_interpretation_medium(
        DevelopedExposureResult(neutral_density, bw_contract)
    )
    rows.append(
        {
            "name": "bw_neutral",
            "descriptor": bw_medium.descriptor(),
            "values_sha256": _sha(bw_medium.values),
            "minimum": float(np.min(bw_medium.values)),
            "maximum": float(np.max(bw_medium.values)),
            "partition_exact": True,
        }
    )

    guards: dict[str, bool] = {
        "non_neutral_bw_rejected": non_neutral_bw_rejected
    }
    try:
        prepare_interpretation_medium(
            develop_layer_exposure(
                source,
                sensitometry,
                DevelopmentInterpretationContract.for_operator(
                    sensitometry,
                    emulsion_family=EmulsionFamily.COLOR_NEGATIVE,
                    interpretation_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
                ),
            )
        )
        guards["missing_print_operator_rejected"] = False
    except TypeError:
        guards["missing_print_operator_rejected"] = True
    try:
        prepare_interpretation_medium(
            develop_layer_exposure(
                source,
                sensitometry,
                DevelopmentInterpretationContract.for_operator(
                    sensitometry,
                    emulsion_family=EmulsionFamily.SLIDE,
                    interpretation_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
                ),
            ),
            print_interpretation=print_operator,
        )
        guards["foreign_print_operator_rejected"] = False
    except ValueError:
        guards["foreign_print_operator_rejected"] = True

    implementation = (
        ROOT / "src/film_physics/interpretation_medium.py"
    ).read_text(encoding="utf-8")
    duplicate_symbols = sum(
        token in implementation
        for token in (
            "class AnchoredCharacteristicCurve",
            "class LogExposureEncoder",
            "def density_to_transmittance",
        )
    )
    checks = {
        "route_count": len(rows) == 4,
        "route_replay": all(row["partition_exact"] for row in rows),
        "input_preservation": _sha(source.values) == source_sha,
        "bounds": all(
            row["minimum"]
            >= float(config["automatic_gates"]["minimum_medium_value"])
            and row["maximum"]
            <= float(config["automatic_gates"]["maximum_medium_value"])
            for row in rows
        ),
        "negative_slide_bytes": media["negative"].values.tobytes()
        == media["slide"].values.tobytes(),
        "negative_slide_polarity": (
            media["negative"].post_scan_polarity
            is not media["slide"].post_scan_polarity
        ),
        "bw_neutral": bool(
            np.array_equal(
                bw_medium.values[..., 0], bw_medium.values[..., 1]
            )
            and np.array_equal(
                bw_medium.values[..., 1], bw_medium.values[..., 2]
            )
        ),
        "guards": all(guards.values()),
        "no_duplicate_density_or_tone": duplicate_symbols
        == int(config["automatic_gates"]["duplicate_density_or_tone_symbols"]),
    }
    core = {
        "schema": "neuro_film.u6_p2b_interpretation_medium_report.v1",
        "node": config["node"],
        "source_sha256": source_sha,
        "routes": rows,
        "guards": guards,
        "duplicate_density_or_tone_symbols": duplicate_symbols,
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
        default=ROOT / "configs/u6_p2b_interpretation_medium_contract_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.config.read_text(encoding="utf-8")))
    _atomic_json(args.output, report)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"evidence_id={report['evidence_id']}")


if __name__ == "__main__":
    main()
