"""U6.P6Z measured scanner context-glare transfer evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

CONTRACT_SCHEMA = "neuro_film.u6_p6z_scanner_context_glare_transfer_contract.v1"
SOURCE_SCHEMA = "neuro_film.u6_p6z_scanner_context_glare_table.v1"
REPORT_SCHEMA = "neuro_film.u6_p6z_scanner_context_glare_transfer_report.v1"
SOURCE_TABLE_SEMANTIC_SHA256 = (
    "da3f8e171df82be5aadfb86957f766d9c3b09f032d9cd013baeb85e99be36e03"
)
PASS_DECISION = "RETAIN_GENERIC_SAME_HARDWARE_CONTEXT_GLARE_TRANSFER_HYPOTHESIS"
FAIL_DECISION = "FAIL_CLOSED_CONTEXT_GLARE_FRACTION_NOT_TRANSFERABLE_ACROSS_LABS"


class ScannerContextGlareTransferError(ValueError):
    """Raised when a frozen P6Z input or invariant drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScannerContextGlareTransferError(f"P6Z object required: {path}")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    protocol = payload.get("measurement_protocol", {})
    operator = payload.get("explicit_operator", {})
    gates = payload.get("automatic_gates", {})
    if (
        payload.get("schema") != CONTRACT_SCHEMA
        or payload.get("experiment_id") != "u6.p6z-scanner-context-glare-transfer-v1"
        or payload.get("parent", {}).get("decision_sha256")
        != "e9544a367cb1cd9870a97696d3aad3a29eecc116121a3e65af13a50dac75dbdd"
        or protocol.get("development_scanners") != [1, 4, 8, 11]
        or protocol.get("confirmation_scanners") != [2, 3, 10, 9, 12]
        or protocol.get("same_hardware_groups")
        != {
            "group-123": [1, 2, 3],
            "group-4-10": [4, 10],
            "group-8-9": [8, 9],
            "group-11-12": [11, 12],
        }
        or operator.get("minimum_f") != 0.0
        or operator.get("maximum_f") != 0.25
        or operator.get("no_clipping_or_projection") is not True
        or gates
        != {
            "minimum_valid_confirmation_rows": 6,
            "minimum_hardware_transfer_win_rate_vs_no_context": 0.75,
            "minimum_hardware_transfer_win_rate_vs_global": 0.6,
            "minimum_median_error_improvement_vs_no_context": 0.2,
            "minimum_median_error_improvement_vs_global": 0.05,
            "minimum_median_error_improvement_vs_wrong_group": 0.05,
            "maximum_confirmation_p95_absolute_bc_error": 0.02,
            "maximum_development_fraction": 0.25,
            "minimum_development_fraction": 0.0,
            "zero_nonfinite_outputs": True,
            "two_independent_audits": True,
        }
    ):
        raise ScannerContextGlareTransferError("P6Z frozen contract drift")
    return payload


def load_source_table(root: Path, path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    source = payload.get("source", {})
    extraction = payload.get("extraction", {})
    rows = payload.get("rows", [])
    if (
        hashlib.sha256(canonical_json(payload)).hexdigest()
        != SOURCE_TABLE_SEMANTIC_SHA256
        or payload.get("schema") != SOURCE_SCHEMA
        or payload.get("experiment_id") != "u6.p6z-scanner-context-glare-transfer-v1"
        or source.get("doi") != "10.1007/s11760-025-04569-8"
        or source.get("pdf_bytes") != 2_590_747
        or source.get("pdf_sha256")
        != "7168488e638ad16fe993f0d5dd72a5df6bb76f4be77b448ab6cc7dde295c730b"
        or source.get("pdf_pages") != 9
        or source.get("layout_text_sha256")
        != "bd136d3e150036fb65d2b2fed1f5ca06c2a0978cbf8d4321a0bd958eb9cc10b6"
        or [row.get("polarity") for row in rows]
        != ["positive-scan-negative-image", "negative-scan-positive-image"]
        or any(len(row.get("cg", [])) != 12 for row in rows)
        or any(len(row.get("cg_difference_16bit", [])) != 12 for row in rows)
        or extraction
        != {
            "numeric_table_only": True,
            "figure_digitization": False,
            "scanner_ids_are_one_based_array_positions": True,
            "dash_is_null": True,
            "negative_difference_is_preserved": True,
            "thresholds_splits_operator_or_claim_changed": False,
        }
    ):
        raise ScannerContextGlareTransferError("P6Z source table drift")
    pdf_path = root / Path(str(source["pdf_path"]))
    if (
        not pdf_path.is_file()
        or pdf_path.stat().st_size != source["pdf_bytes"]
        or sha256_file(pdf_path) != source["pdf_sha256"]
        or len(PdfReader(pdf_path).pages) != source["pdf_pages"]
    ):
        raise ScannerContextGlareTransferError("P6Z retained PDF drift")
    return payload


def _derive_row(
    polarity: str, scanner_id: int, cg: Any, difference: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "polarity": polarity,
        "scanner_id": scanner_id,
        "cg": cg,
        "cg_difference_16bit": difference,
    }
    if cg is None or difference is None:
        return {**base, "eligible": False, "exclusion_reason": "undefined-table-value"}
    cg_value = float(cg)
    difference_value = float(difference)
    if difference_value < 0.0:
        return {**base, "eligible": False, "exclusion_reason": "negative-difference"}
    if cg_value <= 1.0:
        return {
            **base,
            "eligible": False,
            "exclusion_reason": "nonpositive-ratio-excess",
        }
    bv = (difference_value / 65_535.0) / (cg_value - 1.0)
    bc = cg_value * bv
    fraction = (bc - bv) / (1.0 - bv)
    values = (bv, bc, fraction)
    if not all(math.isfinite(value) for value in values) or not (0.0 <= bv < 1.0):
        raise ScannerContextGlareTransferError("P6Z non-finite or invalid derived row")
    return {
        **base,
        "eligible": True,
        "exclusion_reason": None,
        "bv_relative": bv,
        "bc_relative": bc,
        "oracle_fraction": fraction,
    }


def _improvement(candidate_error: float, control_error: float) -> float:
    if control_error <= 0.0:
        return 0.0 if candidate_error <= 0.0 else -math.inf
    return (control_error - candidate_error) / control_error


def evaluate_scanner_context_glare_transfer(
    config: dict[str, Any], source_table: dict[str, Any]
) -> dict[str, Any]:
    protocol = config["measurement_protocol"]
    groups = protocol["same_hardware_groups"]
    group_for_scanner = {
        scanner: group for group, scanners in groups.items() for scanner in scanners
    }
    group_order = list(groups)
    wrong_group = {
        group: group_order[(index + 1) % len(group_order)]
        for index, group in enumerate(group_order)
    }
    all_rows = [
        _derive_row(polarity_row["polarity"], scanner_id, cg, difference)
        for polarity_row in source_table["rows"]
        for scanner_id, (cg, difference) in enumerate(
            zip(
                polarity_row["cg"],
                polarity_row["cg_difference_16bit"],
                strict=True,
            ),
            start=1,
        )
    ]
    eligible = {
        (row["polarity"], row["scanner_id"]): row for row in all_rows if row["eligible"]
    }
    development_fractions: dict[str, dict[str, float]] = {}
    global_fractions: dict[str, float] = {}
    for polarity in protocol["polarities"]:
        fractions: dict[str, float] = {}
        for scanner_id in protocol["development_scanners"]:
            row = eligible.get((polarity, scanner_id))
            if row is None:
                raise ScannerContextGlareTransferError(
                    "P6Z development row unexpectedly ineligible"
                )
            fractions[group_for_scanner[scanner_id]] = row["oracle_fraction"]
        development_fractions[polarity] = fractions
        global_fractions[polarity] = statistics.median(fractions.values())

    confirmation_rows = []
    for polarity in protocol["polarities"]:
        for scanner_id in protocol["confirmation_scanners"]:
            row = eligible.get((polarity, scanner_id))
            if row is None:
                continue
            group = group_for_scanner[scanner_id]
            fraction = development_fractions[polarity][group]
            global_fraction = global_fractions[polarity]
            wrong_fraction = development_fractions[polarity][wrong_group[group]]
            bv = row["bv_relative"]
            bc = row["bc_relative"]
            candidate = (1.0 - fraction) * bv + fraction
            global_prediction = (1.0 - global_fraction) * bv + global_fraction
            wrong_prediction = (1.0 - wrong_fraction) * bv + wrong_fraction
            errors = {
                "hardware_transfer": abs(candidate - bc),
                "no_context": abs(bv - bc),
                "global": abs(global_prediction - bc),
                "wrong_group": abs(wrong_prediction - bc),
            }
            confirmation_rows.append(
                {
                    "polarity": polarity,
                    "scanner_id": scanner_id,
                    "hardware_group": group,
                    "development_fraction": fraction,
                    "global_fraction": global_fraction,
                    "wrong_group": wrong_group[group],
                    "wrong_group_fraction": wrong_fraction,
                    "oracle_fraction": row["oracle_fraction"],
                    "bv_relative": bv,
                    "bc_relative": bc,
                    "predicted_bc_relative": candidate,
                    "errors": errors,
                    "improvement_vs_no_context": _improvement(
                        errors["hardware_transfer"], errors["no_context"]
                    ),
                    "improvement_vs_global": _improvement(
                        errors["hardware_transfer"], errors["global"]
                    ),
                    "improvement_vs_wrong_group": _improvement(
                        errors["hardware_transfer"], errors["wrong_group"]
                    ),
                }
            )

    count = len(confirmation_rows)
    candidate_errors = [row["errors"]["hardware_transfer"] for row in confirmation_rows]
    metrics = {
        "valid_confirmation_rows": count,
        "hardware_transfer_win_rate_vs_no_context": statistics.fmean(
            row["errors"]["hardware_transfer"] < row["errors"]["no_context"]
            for row in confirmation_rows
        ),
        "hardware_transfer_win_rate_vs_global": statistics.fmean(
            row["errors"]["hardware_transfer"] < row["errors"]["global"]
            for row in confirmation_rows
        ),
        "median_error_improvement_vs_no_context": statistics.median(
            row["improvement_vs_no_context"] for row in confirmation_rows
        ),
        "median_error_improvement_vs_global": statistics.median(
            row["improvement_vs_global"] for row in confirmation_rows
        ),
        "median_error_improvement_vs_wrong_group": statistics.median(
            row["improvement_vs_wrong_group"] for row in confirmation_rows
        ),
        "confirmation_p95_absolute_bc_error": float(
            np.percentile(candidate_errors, 95.0, method="linear")
        ),
        "maximum_development_fraction": max(
            fraction
            for by_group in development_fractions.values()
            for fraction in by_group.values()
        ),
        "minimum_development_fraction": min(
            fraction
            for by_group in development_fractions.values()
            for fraction in by_group.values()
        ),
    }
    gates = config["automatic_gates"]
    checks = {
        "minimum_valid_confirmation_rows": metrics["valid_confirmation_rows"]
        >= gates["minimum_valid_confirmation_rows"],
        "hardware_transfer_win_rate_vs_no_context": metrics[
            "hardware_transfer_win_rate_vs_no_context"
        ]
        >= gates["minimum_hardware_transfer_win_rate_vs_no_context"],
        "hardware_transfer_win_rate_vs_global": metrics[
            "hardware_transfer_win_rate_vs_global"
        ]
        >= gates["minimum_hardware_transfer_win_rate_vs_global"],
        "median_error_improvement_vs_no_context": metrics[
            "median_error_improvement_vs_no_context"
        ]
        >= gates["minimum_median_error_improvement_vs_no_context"],
        "median_error_improvement_vs_global": metrics[
            "median_error_improvement_vs_global"
        ]
        >= gates["minimum_median_error_improvement_vs_global"],
        "median_error_improvement_vs_wrong_group": metrics[
            "median_error_improvement_vs_wrong_group"
        ]
        >= gates["minimum_median_error_improvement_vs_wrong_group"],
        "confirmation_p95_absolute_bc_error": metrics[
            "confirmation_p95_absolute_bc_error"
        ]
        <= gates["maximum_confirmation_p95_absolute_bc_error"],
        "development_fraction_bounds": metrics["minimum_development_fraction"]
        >= gates["minimum_development_fraction"]
        and metrics["maximum_development_fraction"]
        <= gates["maximum_development_fraction"],
        "finite_outputs": all(
            math.isfinite(value)
            for row in confirmation_rows
            for value in (
                row["predicted_bc_relative"],
                *row["errors"].values(),
                row["improvement_vs_no_context"],
                row["improvement_vs_global"],
                row["improvement_vs_wrong_group"],
            )
        ),
        "two_independent_audits": gates["two_independent_audits"] is True,
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "source_pdf": source_table["source"],
        "derived_table_rows": all_rows,
        "development_fractions": development_fractions,
        "global_fractions": global_fractions,
        "confirmation_rows": confirmation_rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "decision": PASS_DECISION if automatic_pass else FAIL_DECISION,
        "pixel_decode_count": 0,
        "image_render_count": 0,
        "visual_review_count": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SCHEMA",
    "FAIL_DECISION",
    "PASS_DECISION",
    "REPORT_SCHEMA",
    "SOURCE_SCHEMA",
    "SOURCE_TABLE_SEMANTIC_SHA256",
    "ScannerContextGlareTransferError",
    "canonical_json",
    "evaluate_scanner_context_glare_transfer",
    "load_contract",
    "load_source_table",
    "sha256_file",
    "write_report",
]
