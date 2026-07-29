"""Evaluate frozen U6.P2I endpoint-derived scan normalization."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.generic_reversal_development import _file_sha256
from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    GenericReversalDevelopment,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    ScanSignalNormalization,
    derive_scan_signal_normalization,
    reversal_development_identity,
    scan_signal_normalization_identity,
)


SCHEMA = "neuro_film.u6_p2i_scan_signal_normalization.v1"


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        memoryview(np.ascontiguousarray(values)).cast("B")
    ).hexdigest()


def _display(values: np.ndarray) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.DISPLAY_LINEAR,
        PhysicalUnit.RELATIVE_DISPLAY_LIGHT,
        ("red", "green", "blue"),
    )


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2I contract")
    parents = contract["parents"]
    if (
        _file_sha256(root / parents["p2h_decision_path"])
        != parents["p2h_decision_sha256"]
    ):
        raise ValueError("P2H decision hash drift")
    if (
        _file_sha256(root / parents["scanner_contract_path"])
        != parents["scanner_contract_sha256"]
    ):
        raise ValueError("scanner contract hash drift")
    p2f = json.loads(
        (root / parents["p2f_contract_path"]).read_text(encoding="utf-8")
    )
    template_path = root / p2f["parents"]["negative_sensitometry_path"]
    reversal = GenericReversalDevelopment(
        build_operator(json.loads(template_path.read_text(encoding="utf-8"))),
        float(p2f["candidate"]["maximum_relative_layer_exposure"]),
    )
    if (
        reversal_development_identity(reversal)
        != parents["p2f_operator_sha256"]
    ):
        raise ValueError("reversal operator identity drift")
    scanner_contract = json.loads(
        (root / parents["scanner_contract_path"]).read_text(encoding="utf-8")
    )
    scanner = _profile(
        scanner_contract["profiles"][parents["scanner_profile"]]
    )
    derivation = contract["endpoint_derivation"]
    normalization = derive_scan_signal_normalization(
        reversal,
        scanner,
        flat_field_shape=tuple(derivation["flat_field_shape"]),
        scanner_stages=tuple(derivation["scanner_stages"]),
    )
    replay = ScanSignalNormalization.from_dict(
        json.loads(json.dumps(normalization.to_dict(), sort_keys=True))
    )
    endpoints = np.asarray(
        [
            normalization.black_scan_rgb,
            normalization.white_scan_rgb,
        ],
        dtype=np.float64,
    )
    endpoint_output = normalization.apply(
        PhysicalDomainArray(
            endpoints,
            PhysicalDomain.SCAN_LINEAR,
            PhysicalUnit.RELATIVE_SCAN_SIGNAL,
            ("red", "green", "blue"),
        )
    ).values
    target_endpoints = np.asarray([[0.0] * 3, [1.0] * 3])
    endpoint_error = float(
        np.max(np.abs(endpoint_output - target_endpoints))
    )

    synthetic = contract["synthetic"]
    rng = np.random.default_rng(int(synthetic["seed"]))
    random_display = rng.uniform(
        0.0, 1.0, size=(int(synthetic["random_samples"]), 3)
    )
    random_before = random_display.tobytes()
    scan = normalization.inverse(_display(random_display))
    restored = normalization.apply(scan).values
    replay_output = replay.apply(scan).values
    stops = tuple(int(value) for value in synthetic["partition_stops"])
    if stops[0] != 0 or stops[-1] != len(random_display):
        raise ValueError("partition stops do not cover synthetic samples")
    partitioned = np.concatenate(
        [
            normalization.apply(
                PhysicalDomainArray(
                    scan.values[start:stop],
                    PhysicalDomain.SCAN_LINEAR,
                    PhysicalUnit.RELATIVE_SCAN_SIGNAL,
                    ("red", "green", "blue"),
                )
            ).values
            for start, stop in zip(stops[:-1], stops[1:], strict=True)
        ]
    )
    neutral = np.linspace(
        0.0, 1.0, int(synthetic["neutral_samples"]), dtype=np.float64
    )
    neutral_rgb = np.repeat(neutral[:, None], 3, axis=1)
    neutral_scan = normalization.inverse(_display(neutral_rgb))
    neutral_output = normalization.apply(neutral_scan).values
    neutral_difference = np.diff(neutral_output, axis=0)

    guards: dict[str, bool] = {}
    black = np.asarray(normalization.black_scan_rgb)
    try:
        normalization.apply(
            PhysicalDomainArray(
                (black - 1e-6)[None, :],
                PhysicalDomain.SCAN_LINEAR,
                PhysicalUnit.RELATIVE_SCAN_SIGNAL,
                ("red", "green", "blue"),
            )
        )
        guards["outside"] = False
    except ValueError:
        guards["outside"] = True
    wrong = PhysicalDomainArray(
        np.full((1, 3), 0.5, np.float64),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )
    try:
        normalization.apply(wrong)
        guards["domain"] = False
    except ValueError:
        guards["domain"] = True
    try:
        replace(
            normalization,
            white_scan_rgb=normalization.black_scan_rgb,
        )
        guards["endpoint"] = False
    except ValueError:
        guards["endpoint"] = True
    try:
        replace(normalization, reversal_operator_sha256="0" * 64)
        guards["identity"] = (
            scan_signal_normalization_identity(normalization)
            != scan_signal_normalization_identity(
                replace(normalization, reversal_operator_sha256="0" * 64)
            )
        )
    except ValueError:
        guards["identity"] = True

    metrics = {
        "normalization_sha256": scan_signal_normalization_identity(
            normalization
        ),
        "black_scan_rgb": list(normalization.black_scan_rgb),
        "white_scan_rgb": list(normalization.white_scan_rgb),
        "endpoint_separation_rgb": normalization.separation_rgb.tolist(),
        "endpoint_mapping_max_abs_error": endpoint_error,
        "inverse_roundtrip_max_abs": float(
            np.max(np.abs(restored - random_display))
        ),
        "minimum_neutral_first_difference": float(
            np.min(neutral_difference)
        ),
        "maximum_positive_neutral_first_difference_magnitude": float(
            np.max(neutral_difference)
        ),
        "replay_exact": restored.tobytes() == replay_output.tobytes(),
        "partition_exact": restored.tobytes() == partitioned.tobytes(),
        "serialization_exact": replay == normalization,
        "input_unchanged": random_display.tobytes() == random_before,
        "guards": guards,
        "array_sha256": {
            "random_display": _array_sha256(random_display),
            "scan": _array_sha256(scan.values),
            "restored": _array_sha256(restored),
            "neutral_output": _array_sha256(neutral_output),
        },
    }
    gates = contract["automatic_gates"]
    checks = {
        "endpoint_separation": min(metrics["endpoint_separation_rgb"])
        >= float(gates["minimum_endpoint_separation_per_channel"]),
        "endpoint_mapping": endpoint_error
        <= float(gates["maximum_endpoint_mapping_absolute_error"]),
        "inverse": metrics["inverse_roundtrip_max_abs"]
        <= float(gates["maximum_float64_inverse_roundtrip_absolute_error"]),
        "monotone": metrics["minimum_neutral_first_difference"]
        >= float(gates["minimum_neutral_first_difference"]),
        "material": metrics[
            "maximum_positive_neutral_first_difference_magnitude"
        ]
        >= float(
            gates["minimum_positive_neutral_first_difference_magnitude"]
        ),
        "replay": bool(metrics["replay_exact"]),
        "partition": bool(metrics["partition_exact"]),
        "serialization": bool(metrics["serialization_exact"]),
        "input": bool(metrics["input_unchanged"]),
        "guards": all(guards.values()),
    }
    passed = all(checks.values())
    report = {
        "schema": "neuro_film.u6_p2i_scan_signal_normalization_report.v1",
        "node": contract["node"],
        "normalization": normalization.to_dict(),
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
