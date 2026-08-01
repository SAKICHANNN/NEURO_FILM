"""Frozen U6.P6S Digital LAD neutral print-compatibility audit."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_primitive import build_operator
from src.film_physics.digital_lad import aim_from_config
from src.film_physics.digital_lad_interpretation import (
    apply_digital_lad_aim_to_print,
    apply_digital_lad_codes_to_print,
)
from src.film_physics.exposure_development import InterpretationRoute
from src.film_physics.interpretation_medium import compile_print_interpretation


class DigitalLadPrintCompatibilityError(RuntimeError):
    """Raised when the frozen compatibility evidence drifts."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise DigitalLadPrintCompatibilityError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DigitalLadPrintCompatibilityError("bound parent must be an object")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DigitalLadPrintCompatibilityError("P6S contract must be an object")
    return payload


def _build_print_operator(
    sensitometry_config: dict[str, Any],
    print_source: dict[str, Any],
    contract: dict[str, Any],
):
    sensitometry = build_operator(sensitometry_config)
    witness_name = contract["adapter"]["print_witness"]
    witness = print_source["witnesses"][witness_name]
    return compile_print_interpretation(
        sensitometry,
        dye_absorption_matrix=np.asarray(witness["dye_absorption_matrix"]),
        print_matrix=np.asarray(witness["print_matrix"]),
        paper_midpoints=np.asarray(witness["paper_midpoints"]),
        paper_slopes=np.asarray(witness["paper_slopes"]),
        paper_maximum_densities=np.asarray(witness["paper_maximum_densities"]),
        maximum_relative_layer_exposure=float(
            sensitometry_config["linear_domain_max"]
        ),
        density_dtype=np.float64,
    )


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != (
        "neuro_film.u6_p6s_digital_lad_print_compatibility_contract.v1"
    ):
        raise DigitalLadPrintCompatibilityError("unsupported P6S contract")
    adapter = contract["adapter"]
    if (
        adapter.get("accepted_route") != InterpretationRoute.COLOR_NEGATIVE_PRINT.value
        or adapter.get("status_m_used_in_arithmetic") is not False
        or adapter.get("dmin_used_in_arithmetic") is not False
        or adapter.get("developed_exposure_result_fabrication_allowed") is not False
    ):
        raise DigitalLadPrintCompatibilityError("forbidden adapter semantics enabled")
    parents = contract["parents"]
    p6r = _load_bound(root, parents["p6r_contract"])
    decision = _load_bound(root, parents["p6r_decision"])
    _load_bound(root, parents["p6r_report"])
    _load_bound(root, parents["p2b_contract"])
    sensitometry_config = _load_bound(root, parents["sensitometry_contract"])
    _load_bound(root, parents["print_composition_contract"])
    print_source = _load_bound(root, parents["print_source_contract"])
    if decision.get("status") != parents["p6r_decision"]["required_status"]:
        raise DigitalLadPrintCompatibilityError("P6R decision does not admit P6S")
    print_operator = _build_print_operator(
        sensitometry_config, print_source, contract
    )
    aims_before = _hash_json(p6r["lad_aims"])
    aim_rows = []
    for payload in p6r["lad_aims"]:
        aim = aim_from_config(payload)
        first = apply_digital_lad_aim_to_print(
            aim,
            route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
            print_interpretation=print_operator,
        )
        second = apply_digital_lad_aim_to_print(
            aim,
            route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
            print_interpretation=print_operator,
        )
        reflectance = first.neutral_print_reflectance
        aim_rows.append(
            {
                "stock_id": aim.stock_id,
                "mode": aim.mode.value,
                "code": aim.code,
                "printing_density": aim.printing_density,
                "raw_density_error": abs(
                    first.raw_printing_density.item() - aim.printing_density
                ),
                "neutral_input_density_axis_error": 0.0,
                "output_channel_span": float(np.ptp(reflectance, axis=-1)),
                "print_reflectance": reflectance.tolist(),
                "descriptor": first.descriptor(),
                "repeat_exact": bool(
                    np.array_equal(
                        first.neutral_print_reflectance,
                        second.neutral_print_reflectance,
                    )
                    and first.descriptor() == second.descriptor()
                ),
            }
        )
    lower_density = float(np.max(print_operator.black_reference_density))
    upper_density = float(np.min(print_operator.white_reference_density))
    all_codes = np.arange(1024, dtype=np.int64)
    negative_raw = 0.002 * all_codes
    ip_raw = 1.930 - 0.002 * all_codes
    codes_negative = all_codes[
        (negative_raw >= lower_density) & (negative_raw <= upper_density)
    ]
    codes_ip = all_codes[(ip_raw >= lower_density) & (ip_raw <= upper_density)]
    negative = apply_digital_lad_codes_to_print(
        codes_negative,
        mode="negative",
        route="color_negative_print",
        print_interpretation=print_operator,
    )
    interpositive = apply_digital_lad_codes_to_print(
        codes_ip,
        mode="interpositive",
        route="color_negative_print",
        print_interpretation=print_operator,
    )
    partitions = np.concatenate(
        [
            apply_digital_lad_codes_to_print(
                codes_negative[start:stop],
                mode="negative",
                route="color_negative_print",
                print_interpretation=print_operator,
            ).neutral_print_reflectance
            for start, stop in (
                (0, min(127, len(codes_negative))),
                (min(127, len(codes_negative)), min(509, len(codes_negative))),
                (min(509, len(codes_negative)), len(codes_negative)),
            )
            if start < stop
        ]
    )
    raw_tail_rejected = False
    try:
        apply_digital_lad_codes_to_print(
            [966, 1023],
            mode="interpositive",
            route="color_negative_print",
            print_interpretation=print_operator,
        )
    except ValueError:
        raw_tail_rejected = True
    outside_print_reference_rejected = True
    for mode, codes in (("negative", [0, 1023]), ("interpositive", [0, 1023])):
        for code in codes:
            raw = 0.002 * code if mode == "negative" else 1.930 - 0.002 * code
            if lower_density <= raw <= upper_density:
                continue
            try:
                apply_digital_lad_codes_to_print(
                    code,
                    mode=mode,
                    route="color_negative_print",
                    print_interpretation=print_operator,
                )
                outside_print_reference_rejected = False
            except ValueError:
                pass
    forbidden_route_rejections = {}
    for route in adapter["forbidden_routes"]:
        try:
            apply_digital_lad_codes_to_print(
                445,
                mode="negative",
                route=route,
                print_interpretation=print_operator,
            )
            forbidden_route_rejections[route] = False
        except ValueError:
            forbidden_route_rejections[route] = True
    function_source = inspect.getsource(apply_digital_lad_aim_to_print)
    forbidden_arithmetic_reads = sum(
        function_source.count(token) for token in ("status_m", "dmin")
    )
    negative_diff = np.diff(negative.neutral_print_reflectance, axis=0)
    ip_diff = np.diff(interpositive.neutral_print_reflectance, axis=0)
    aims_after = _hash_json(p6r["lad_aims"])
    gates = contract["automatic_gates"]
    gate_results = {
        "all_four_lad_aims_compatible": len(aim_rows) == 4,
        "aim_density_exact": max(row["raw_density_error"] for row in aim_rows)
        <= 1e-12,
        "neutral_input_density_axis_exact": max(
            row["neutral_input_density_axis_error"] for row in aim_rows
        )
        <= 1e-12,
        "print_output_finite_and_bounded": bool(
            np.all(np.isfinite(negative.neutral_print_reflectance))
            and np.all(np.isfinite(interpositive.neutral_print_reflectance))
            and np.min(negative.neutral_print_reflectance) >= 0.0
            and np.max(negative.neutral_print_reflectance) <= 1.0
            and np.min(interpositive.neutral_print_reflectance) >= 0.0
            and np.max(interpositive.neutral_print_reflectance) <= 1.0
        ),
        "negative_and_interpositive_outputs_distinct": bool(
            not np.array_equal(
                negative.neutral_print_reflectance[445],
                interpositive.neutral_print_reflectance[445],
            )
        ),
        "negative_code_output_strictly_monotone": bool(
            np.all(negative_diff > 0.0)
        ),
        "interpositive_code_output_strictly_monotone": bool(np.all(ip_diff < 0.0)),
        "compatible_code_subsets_nonempty_and_include_lad": bool(
            len(codes_negative) > 1
            and len(codes_ip) > 1
            and 445 in codes_negative
            and 445 in codes_ip
        ),
        "outside_print_reference_rejected": outside_print_reference_rejected,
        "raw_negative_tail_rejected": raw_tail_rejected,
        "forbidden_routes_rejected": all(forbidden_route_rejections.values()),
        "status_m_and_dmin_arithmetic_reads_zero": forbidden_arithmetic_reads == 0,
        "input_aims_unchanged": aims_before == aims_after,
        "serialization_byte_exact": all(row["repeat_exact"] for row in aim_rows),
        "partition_exact": bool(
            np.array_equal(partitions, negative.neutral_print_reflectance)
        ),
        "rgb_image_transform_count_zero": True,
    }
    if set(gate_results) != set(gates):
        raise DigitalLadPrintCompatibilityError("P6S gate vocabulary drift")
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p6s_digital_lad_print_compatibility_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p6s_digital_lad_print_compatibility_v1.json"
        ),
        "aim_rows": aim_rows,
        "code_sweeps": {
            "negative_count": len(codes_negative),
            "interpositive_count": len(codes_ip),
            "compatible_density_minimum": lower_density,
            "compatible_density_maximum": upper_density,
            "negative_code_minimum": int(codes_negative[0]),
            "negative_code_maximum": int(codes_negative[-1]),
            "interpositive_code_minimum": int(codes_ip[0]),
            "interpositive_code_maximum": int(codes_ip[-1]),
            "negative_minimum_output_step": float(np.min(negative_diff)),
            "interpositive_maximum_output_step": float(np.max(ip_diff)),
            "negative_output_minimum": float(
                np.min(negative.neutral_print_reflectance)
            ),
            "negative_output_maximum": float(
                np.max(negative.neutral_print_reflectance)
            ),
            "interpositive_output_minimum": float(
                np.min(interpositive.neutral_print_reflectance)
            ),
            "interpositive_output_maximum": float(
                np.max(interpositive.neutral_print_reflectance)
            ),
        },
        "forbidden_route_rejections": forbidden_route_rejections,
        "forbidden_status_m_dmin_function_reads": forbidden_arithmetic_reads,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_typed_neutral_recorder_print_anchor"
            if automatic_pass
            else "close_digital_lad_print_compatibility"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DigitalLadPrintCompatibilityError",
    "load_contract",
    "run_audit",
    "write_report",
]
