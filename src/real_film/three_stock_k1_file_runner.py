"""File-backed SF3.A1 to SF3.A2 controlled three-stock baseline runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.real_film.three_stock_k1_baseline import (
    evaluate as evaluate_k1,
)
from src.real_film.three_stock_k1_baseline import (
    load_contract as load_k1_contract,
)
from src.real_film.three_stock_paired_sampling import (
    AlignedScanRow,
    extract_common_paired_samples,
)
from src.real_film.three_stock_scan_integrity import (
    decode_integer_rgb,
)
from src.real_film.three_stock_scan_integrity import (
    evaluate as evaluate_integrity,
)

REPORT_SCHEMA = "neuro-film.sf3-a2-three-stock-k1-file-runner-report.v1"


class ThreeStockK1FileRunnerError(ValueError):
    """Raised when a file-backed A1/A2 handoff is structurally invalid."""


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
        raise ThreeStockK1FileRunnerError(f"expected JSON object: {path}")
    return raw, value


def _data_path(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThreeStockK1FileRunnerError(f"invalid {field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ThreeStockK1FileRunnerError(f"{field} must be repository-relative")
    if not relative.parts or relative.parts[0].casefold() != "data":
        raise ThreeStockK1FileRunnerError(f"{field} must use logical data root")
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise ThreeStockK1FileRunnerError(f"missing {field}: {value}")
    return path


def load_aligned_rows(
    *,
    root: Path,
    ledger: dict[str, Any],
    manifest: dict[str, Any],
    decode_contract: dict[str, Any],
    alignment_schema: str,
) -> list[AlignedScanRow]:
    """Load only identities already verified by the immediately preceding A1 audit."""

    source_rows = ledger.get("rows")
    manifest_rows = manifest.get("rows")
    if (
        not isinstance(source_rows, list)
        or not isinstance(manifest_rows, list)
        or len(source_rows) != len(manifest_rows)
        or not source_rows
    ):
        raise ThreeStockK1FileRunnerError("ledger/manifest row inventory drift")
    rows: list[AlignedScanRow] = []
    for source, row in zip(source_rows, manifest_rows, strict=True):
        if not isinstance(source, dict) or not isinstance(row, dict):
            raise ThreeStockK1FileRunnerError("ledger/manifest row type drift")
        evidence_path = _data_path(
            root, source.get("alignment_evidence_path"), field="alignment_evidence_path"
        )
        _, evidence = _read_object(evidence_path)
        if (
            evidence.get("schema") != alignment_schema
            or evidence.get("row_id") != row.get("row_id")
            or evidence.get("digital_reference_sha256")
            != row.get("digital_reference_sha256")
            or evidence.get("scan_sample_sha256") != row.get("scan_sample_sha256")
        ):
            raise ThreeStockK1FileRunnerError("alignment evidence identity drift")
        homography = np.asarray(
            evidence.get("homography_source_to_scan"), dtype=np.float64
        )
        digital_path = _data_path(
            root, source.get("digital_reference_path"), field="digital_reference_path"
        )
        scan_path = _data_path(
            root, source.get("scan_sample_path"), field="scan_sample_path"
        )
        rows.append(
            AlignedScanRow(
                row_id=str(row["row_id"]),
                stock_id=str(row["stock_id"]),
                role=str(row["role"]),
                scene_id=str(row["scene_id"]),
                film_frame_id=str(row["film_frame_id"]),
                roll_id=str(row["roll_id"]),
                source_rgb=decode_integer_rgb(digital_path, decode_contract),
                scan_rgb=decode_integer_rgb(scan_path, decode_contract),
                homography_source_to_scan=homography,
            )
        )
    return rows


def evaluate_files(
    *,
    root: Path,
    integrity_contract_path: Path,
    k1_contract_path: Path,
    ledger_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run integrity, common-coordinate sampling, then the frozen K=1 experiment."""

    k1_raw, k1_contract = load_k1_contract(k1_contract_path, root=root)
    integrity_report = evaluate_integrity(
        integrity_contract_path, ledger_path, manifest_path, root=root
    )
    ledger_raw, ledger = _read_object(ledger_path)
    manifest_raw, manifest = _read_object(manifest_path)
    if not integrity_report["automatic_pass"]:
        core = {
            "schema": REPORT_SCHEMA,
            "experiment_id": "SF3.A2",
            "k1_contract_sha256": _sha256(k1_raw),
            "ledger_sha256": _sha256(ledger_raw),
            "manifest_sha256": _sha256(manifest_raw),
            "integrity_stable_evidence_id": integrity_report["stable_evidence_id"],
            "integrity_automatic_pass": False,
            "paired_sampling_executed": False,
            "operator_fits": 0,
            "automatic_pass": False,
            "decision": integrity_report["decision"],
            "claim_ceiling": k1_contract["claim_ceiling"],
        }
        return {**core, "stable_evidence_id": _sha256(_canonical(core))}

    _, integrity_contract = _read_object(integrity_contract_path)
    aligned = load_aligned_rows(
        root=root,
        ledger=ledger,
        manifest=manifest,
        decode_contract=integrity_contract["decode"],
        alignment_schema=integrity_contract["record_schemas"]["alignment"],
    )
    development, confirmation, sampling_facts = extract_common_paired_samples(
        aligned, k1_contract["paired_sampling"]
    )
    result = evaluate_k1(
        k1_contract_path,
        root=root,
        development=development,
        confirmation=confirmation,
    )
    candidate_count = len(
        k1_contract["operator_selection"]["candidate_order_simplest_first"]
    )
    operator_fits = sum(
        len({frame.roll_id for frame in development[stock]}) * candidate_count + 1
        for stock in k1_contract["required_stocks"]
    )
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "SF3.A2",
        "k1_contract_sha256": _sha256(k1_raw),
        "ledger_sha256": _sha256(ledger_raw),
        "manifest_sha256": _sha256(manifest_raw),
        "integrity_stable_evidence_id": integrity_report["stable_evidence_id"],
        "integrity_automatic_pass": True,
        "paired_sampling_executed": True,
        "paired_sampling": sampling_facts,
        "k1_result": result,
        "operator_fits": operator_fits,
        "automatic_pass": result["automatic_pass"],
        "decision": result["decision"],
        "claim_ceiling": k1_contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "REPORT_SCHEMA",
    "ThreeStockK1FileRunnerError",
    "evaluate_files",
    "load_aligned_rows",
]
