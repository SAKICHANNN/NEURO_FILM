#!/usr/bin/env python
"""Evaluate the frozen U6.P2A typed development contract."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationEvidence,
    InterpretationRoute,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    ProcessCondition,
    develop_layer_exposure,
)


ROUTES = (
    (
        EmulsionFamily.COLOR_NEGATIVE,
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
    ),
    (
        EmulsionFamily.COLOR_NEGATIVE,
        InterpretationRoute.COLOR_NEGATIVE_PRINT,
    ),
    (EmulsionFamily.SLIDE, InterpretationRoute.SLIDE_DIRECT_SCAN),
    (
        EmulsionFamily.BLACK_AND_WHITE,
        InterpretationRoute.BW_DEVELOPER_SCAN,
    ),
)


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(array)).cast("B")
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
        (ROOT / config["reused_sensitometry"]["contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    operator = build_operator(u2)
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
    route_rows = []
    for family, route in ROUTES:
        contract = DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=family,
            interpretation_route=route,
        )
        replay = DevelopmentInterpretationContract.from_dict(
            json.loads(json.dumps(contract.to_dict()))
        )
        full = develop_layer_exposure(source, operator, replay).density.values
        pieces = []
        for start, stop in ((0, 37), (37, 128), (128, 257)):
            part = PhysicalDomainArray(
                source.values[start:stop],
                source.domain,
                source.unit,
                source.channels,
            )
            pieces.append(
                develop_layer_exposure(part, operator, replay).density.values
            )
        partitioned = np.concatenate(pieces)
        route_rows.append(
            {
                "emulsion_family": family.value,
                "interpretation_route": route.value,
                "contract": contract.to_dict(),
                "density_sha256": _sha(full),
                "partition_exact": bool(np.array_equal(partitioned, full)),
                "minimum_density": float(np.min(full)),
                "maximum_density": float(np.max(full)),
                "replay_exact": replay == contract,
            }
        )

    guards: dict[str, bool] = {}
    base = DevelopmentInterpretationContract.for_operator(
        operator,
        emulsion_family=EmulsionFamily.SLIDE,
        interpretation_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
    )
    try:
        DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily.SLIDE,
            interpretation_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
        )
        guards["wrong_route_family_rejected"] = False
    except ValueError:
        guards["wrong_route_family_rejected"] = True
    wrong_domain = PhysicalDomainArray(
        np.ones((2, 3, 3), np.float32),
        PhysicalDomain.SCENE_LINEAR,
        PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
        ("red", "green", "blue"),
    )
    try:
        develop_layer_exposure(wrong_domain, operator, base)
        guards["wrong_input_domain_rejected"] = False
    except ValueError:
        guards["wrong_input_domain_rejected"] = True
    try:
        develop_layer_exposure(
            source, operator, replace(base, sensitometry_sha256="0" * 64)
        )
        guards["operator_identity_mismatch_rejected"] = False
    except ValueError:
        guards["operator_identity_mismatch_rejected"] = True
    try:
        DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily.SLIDE,
            interpretation_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
            process_condition=ProcessCondition.PUSH,
            interpretation_evidence=InterpretationEvidence.UNKNOWN,
        )
        guards["process_evidence_mismatch_rejected"] = False
    except ValueError:
        guards["process_evidence_mismatch_rejected"] = True

    implementation = (
        ROOT / "src/film_physics/exposure_development.py"
    ).read_text(encoding="utf-8")
    duplicate_symbols = sum(
        token in implementation
        for token in (
            "class AnchoredCharacteristicCurve",
            "class LogExposureEncoder",
            "class RationalQuadraticSpline",
        )
    )
    checks = {
        "route_count": len(route_rows) == 4,
        "operator_replay": all(row["replay_exact"] for row in route_rows),
        "partition_replay": all(row["partition_exact"] for row in route_rows),
        "input_preservation": _sha(source.values) == source_sha,
        "density_nonnegative": min(
            row["minimum_density"] for row in route_rows
        )
        >= float(config["automatic_gates"]["minimum_density"]),
        "guards": all(guards.values()),
        "no_duplicate_curve_symbols": duplicate_symbols
        == int(config["automatic_gates"]["duplicate_tone_curve_symbols"]),
    }
    core = {
        "schema": "neuro_film.u6_p2a_exposure_development_report.v1",
        "node": config["node"],
        "source_sha256": source_sha,
        "routes": route_rows,
        "guards": guards,
        "duplicate_tone_curve_symbols": duplicate_symbols,
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
        default=ROOT
        / "configs/u6_p2a_exposure_development_interpretation_contract_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.config.read_text(encoding="utf-8")))
    _atomic_json(args.output, report)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"evidence_id={report['evidence_id']}")


if __name__ == "__main__":
    main()
