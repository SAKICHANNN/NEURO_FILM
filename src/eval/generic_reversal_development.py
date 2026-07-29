"""Evaluate the frozen U6.P2F generic reversal-development primitive."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    GenericReversalDevelopment,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    develop_reversal_layer_exposure,
    prepare_interpretation_medium,
    reversal_development_contract,
    reversal_development_identity,
)


SCHEMA = "neuro_film.u6_p2f_generic_reversal_development.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(values)).cast("B")
    ).hexdigest()


def _state(values: np.ndarray) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2F contract")
    parents = contract["parents"]
    p2c1_path = root / parents["p2c1_decision_path"]
    template_path = root / parents["negative_sensitometry_path"]
    if _file_sha256(p2c1_path) != parents["p2c1_decision_sha256"]:
        raise ValueError("P2C1 decision hash drift")
    if (
        _file_sha256(template_path)
        != parents["negative_sensitometry_config_sha256"]
    ):
        raise ValueError("sensitometry template hash drift")
    template = build_operator(
        json.loads(template_path.read_text(encoding="utf-8"))
    )
    candidate = contract["candidate"]
    operator = GenericReversalDevelopment(
        template, float(candidate["maximum_relative_layer_exposure"])
    )
    replay = GenericReversalDevelopment.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    serialized_exact = replay.to_dict() == operator.to_dict()
    identity = reversal_development_identity(operator)
    replay_identity = reversal_development_identity(replay)

    synthetic = contract["synthetic"]
    maximum = float(candidate["maximum_relative_layer_exposure"])
    neutral = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.geomspace(
                1e-6,
                maximum,
                int(synthetic["neutral_ramp_samples"]) - 1,
                dtype=np.float64,
            ),
        )
    )
    neutral_rgb = np.repeat(neutral[:, None], 3, axis=1)
    neutral_before = neutral_rgb.tobytes()
    developed = develop_reversal_layer_exposure(
        _state(neutral_rgb),
        operator,
        reversal_development_contract(operator),
    )
    density = developed.density.values
    density_difference = np.diff(density, axis=0)
    direct_medium = prepare_interpretation_medium(developed).values

    rng = np.random.default_rng(int(synthetic["random_seed"]))
    random_rgb = rng.uniform(
        0.0,
        maximum,
        size=(int(synthetic["random_rgb_samples"]), 3),
    )
    random_before = random_rgb.tobytes()
    first = operator.apply(random_rgb)
    second = operator.apply(random_rgb)
    restored = operator.inverse(first)
    stops = tuple(int(value) for value in synthetic["partition_stops"])
    if stops[0] != 0 or stops[-1] != len(random_rgb):
        raise ValueError("partition stops do not cover the random population")
    partitioned = np.concatenate(
        [
            operator.apply(random_rgb[start:stop])
            for start, stop in zip(stops[:-1], stops[1:], strict=True)
        ]
    )

    guards: dict[str, bool] = {}
    try:
        operator.apply(np.full((1, 3), maximum + 1e-6))
        guards["maximum"] = False
    except ValueError:
        guards["maximum"] = True
    wrong_domain = PhysicalDomainArray(
        np.full((1, 3), 0.18, np.float64),
        PhysicalDomain.SCENE_LINEAR,
        PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
        ("red", "green", "blue"),
    )
    try:
        develop_reversal_layer_exposure(
            wrong_domain, operator, reversal_development_contract(operator)
        )
        guards["domain"] = False
    except ValueError:
        guards["domain"] = True
    try:
        develop_reversal_layer_exposure(
            _state(np.full((1, 3), 0.18, np.float64)),
            operator,
            replace(
                reversal_development_contract(operator),
                sensitometry_sha256="0" * 64,
            ),
        )
        guards["identity"] = False
    except ValueError:
        guards["identity"] = True

    metrics = {
        "operator_sha256": identity,
        "template_endpoint_density_sum": (
            operator.endpoint_density_sum.tolist()
        ),
        "neutral_density_endpoint_change": (
            density[-1].astype(np.float64) - density[0].astype(np.float64)
        ).tolist(),
        "maximum_neutral_density_first_difference": float(
            np.max(density_difference)
        ),
        "maximum_negative_neutral_density_first_difference_magnitude": float(
            np.max(-density_difference)
        ),
        "direct_transmittance_endpoint_increase": (
            direct_medium[-1].astype(np.float64)
            - direct_medium[0].astype(np.float64)
        ).tolist(),
        "inverse_roundtrip_max_abs": float(
            np.max(np.abs(restored - random_rgb))
        ),
        "density_minimum": float(np.min(first)),
        "density_maximum": float(np.max(first)),
        "array_sha256": {
            "neutral_exposure": _array_sha256(neutral_rgb),
            "neutral_density": _array_sha256(density),
            "direct_transmittance": _array_sha256(direct_medium),
            "random_exposure": _array_sha256(random_rgb),
            "random_density": _array_sha256(first),
            "random_restored": _array_sha256(restored),
        },
        "replay_exact": first.tobytes() == second.tobytes(),
        "partition_exact": first.tobytes() == partitioned.tobytes(),
        "serialization_exact": serialized_exact and identity == replay_identity,
        "input_unchanged": (
            neutral_rgb.tobytes() == neutral_before
            and random_rgb.tobytes() == random_before
        ),
        "guards": guards,
    }
    gates = contract["automatic_gates"]
    checks = {
        "density": bool(
            np.all(np.isfinite(first))
            and np.min(first) >= 0.0
            and np.all(np.isfinite(density))
            and np.min(density) >= 0.0
        ),
        "decreasing": metrics[
            "maximum_neutral_density_first_difference"
        ]
        <= float(gates["maximum_neutral_density_first_difference"]),
        "material_decrease": metrics[
            "maximum_negative_neutral_density_first_difference_magnitude"
        ]
        >= float(
            gates[
                "minimum_negative_neutral_density_first_difference_magnitude"
            ]
        ),
        "direct_brightening": min(
            metrics["direct_transmittance_endpoint_increase"]
        )
        >= float(
            gates[
                "minimum_direct_transmittance_endpoint_increase_per_channel"
            ]
        ),
        "inverse": metrics["inverse_roundtrip_max_abs"]
        <= float(gates["maximum_inverse_roundtrip_absolute_error"]),
        "replay": bool(metrics["replay_exact"]),
        "partition": bool(metrics["partition_exact"]),
        "serialization": bool(metrics["serialization_exact"]),
        "input": bool(metrics["input_unchanged"]),
        "guards": all(guards.values()),
    }
    passed = all(checks.values())
    report = {
        "schema": "neuro_film.u6_p2f_generic_reversal_development_report.v1",
        "node": contract["node"],
        "operator": operator.to_dict(),
        "metrics": metrics,
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
