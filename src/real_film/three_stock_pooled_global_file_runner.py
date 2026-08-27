"""File-backed SF3.A2B pooled-global three-stock control runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.real_film.three_stock_k1_file_runner import load_aligned_file_rows
from src.real_film.three_stock_paired_sampling import (
    extract_common_paired_samples_streaming,
)
from src.real_film.three_stock_pooled_global_control import (
    evaluate as evaluate_pooled_global,
)
from src.real_film.three_stock_pooled_global_control import (
    load_contract as load_pooled_contract,
)
from src.real_film.three_stock_scan_integrity import (
    decode_integer_rgb,
    decode_scan_integer_rgb,
)
from src.real_film.three_stock_scan_integrity import evaluate as evaluate_integrity

REPORT_SCHEMA = "neuro-film.sf3-a2b-three-stock-pooled-global-file-report.v1"


class ThreeStockPooledGlobalFileRunnerError(ValueError):
    """Raised when the file-backed SF3.A2B handoff is structurally invalid."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ThreeStockPooledGlobalFileRunnerError(f"expected JSON object: {path}")
    return raw, value


def evaluate_files(
    *,
    root: Path,
    integrity_contract_path: Path,
    pooled_contract_path: Path,
    ledger_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run A1 integrity, common-coordinate sampling, then frozen SF3.A2B."""

    pooled_raw, pooled_contract, _, k1_contract = load_pooled_contract(
        pooled_contract_path, root=root
    )
    integrity_report = evaluate_integrity(
        integrity_contract_path, ledger_path, manifest_path, root=root
    )
    ledger_raw, ledger = _read_object(ledger_path)
    manifest_raw, manifest = _read_object(manifest_path)
    common = {
        "schema": REPORT_SCHEMA,
        "experiment_id": pooled_contract["experiment_id"],
        "pooled_contract_sha256": _sha256(pooled_raw),
        "ledger_sha256": _sha256(ledger_raw),
        "manifest_sha256": _sha256(manifest_raw),
        "integrity_stable_evidence_id": integrity_report["stable_evidence_id"],
        "integrity_automatic_pass": bool(integrity_report["automatic_pass"]),
    }
    if not integrity_report["automatic_pass"]:
        core = {
            **common,
            "paired_sampling_executed": False,
            "operator_fits": 0,
            "automatic_pass": False,
            "decision": integrity_report["decision"],
            "claim_ceiling": pooled_contract["claim_ceiling"],
        }
        return {**core, "stable_evidence_id": _sha256(_canonical(core))}

    _, integrity_contract = _read_object(integrity_contract_path)
    aligned, paths = load_aligned_file_rows(
        root=root,
        ledger=ledger,
        manifest=manifest,
        alignment_schema=integrity_contract["record_schemas"]["alignment"],
    )
    decode_contract = integrity_contract["decode"]
    development, confirmation, sampling_facts = extract_common_paired_samples_streaming(
        aligned,
        k1_contract["paired_sampling"],
        load_source=lambda row: decode_integer_rgb(
            paths[row.row_id][0], decode_contract
        ),
        load_scan=lambda row: decode_scan_integer_rgb(
            paths[row.row_id][1], decode_contract
        ),
    )
    result = evaluate_pooled_global(
        pooled_contract_path,
        root=root,
        development=development,
        confirmation=confirmation,
    )
    candidate_count = len(
        k1_contract["operator_selection"]["candidate_order_simplest_first"]
    )
    total_development_rolls = sum(
        len({frame.roll_id for frame in development[stock]})
        for stock in pooled_contract["required_stocks"]
    )
    operator_fits = (
        2 * total_development_rolls * candidate_count
        + len(pooled_contract["required_stocks"])
        + 1
    )
    core = {
        **common,
        "paired_sampling_executed": True,
        "paired_sampling": sampling_facts,
        "pooled_global_result": result,
        "operator_fits": operator_fits,
        "automatic_pass": result["automatic_pass"],
        "decision": result["decision"],
        "claim_ceiling": result["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "REPORT_SCHEMA",
    "ThreeStockPooledGlobalFileRunnerError",
    "evaluate_files",
]
