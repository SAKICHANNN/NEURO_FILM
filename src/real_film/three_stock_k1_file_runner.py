"""File-backed SF3.A1 to SF3.A2 controlled three-stock baseline runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image

from src.real_film.three_stock_k1_baseline import (
    evaluate as evaluate_k1,
)
from src.real_film.three_stock_k1_baseline import (
    evaluate_single_stock as evaluate_single_stock_k1,
)
from src.real_film.three_stock_k1_baseline import (
    load_contract as load_k1_contract,
)
from src.real_film.three_stock_paired_sampling import (
    AlignedScanFileRow,
    AlignedScanRow,
    extract_common_paired_samples_streaming,
)
from src.real_film.three_stock_scan_integrity import (
    decode_integer_rgb,
    decode_scan_integer_rgb,
)
from src.real_film.three_stock_scan_integrity import (
    evaluate as evaluate_integrity,
)
from src.real_film.three_stock_scan_integrity import (
    evaluate_single_stock as evaluate_single_stock_integrity,
)

REPORT_SCHEMA = "neuro-film.sf3-a2-three-stock-k1-file-runner-report.v1"
SINGLE_STOCK_REPORT_SCHEMA = "neuro-film.sf3-a2-single-stock-k1-file-runner-report.v1"


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
                scan_rgb=decode_scan_integer_rgb(scan_path, decode_contract),
                homography_source_to_scan=homography,
            )
        )
    return rows


def _integer_rgb_shape(path: Path) -> tuple[int, int, int]:
    """Read bounded RGB geometry without materializing image samples."""

    if path.suffix.casefold() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as image:
            if len(image.pages) != 1:
                raise ThreeStockK1FileRunnerError("multi-page TIFF is forbidden")
            page = image.pages[0]
            shape = (
                int(page.imagelength),
                int(page.imagewidth),
                int(page.samplesperpixel),
            )
    else:
        with Image.open(path) as image:
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ThreeStockK1FileRunnerError("multi-frame image is forbidden")
            if "A" in image.getbands():
                raise ThreeStockK1FileRunnerError("alpha is forbidden")
            width, height = image.size
            shape = (int(height), int(width), len(image.getbands()))
    if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) < 2:
        raise ThreeStockK1FileRunnerError(f"expected non-empty RGB image: {path}")
    return shape


def load_aligned_file_rows(
    *,
    root: Path,
    ledger: dict[str, Any],
    manifest: dict[str, Any],
    alignment_schema: str,
) -> tuple[list[AlignedScanFileRow], dict[str, tuple[Path, Path]]]:
    """Load A1-verified identities and container geometry, never full pixels."""

    source_rows = ledger.get("rows")
    manifest_rows = manifest.get("rows")
    if (
        not isinstance(source_rows, list)
        or not isinstance(manifest_rows, list)
        or len(source_rows) != len(manifest_rows)
        or not source_rows
    ):
        raise ThreeStockK1FileRunnerError("ledger/manifest row inventory drift")
    rows: list[AlignedScanFileRow] = []
    paths: dict[str, tuple[Path, Path]] = {}
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
        digital_path = _data_path(
            root, source.get("digital_reference_path"), field="digital_reference_path"
        )
        scan_path = _data_path(
            root, source.get("scan_sample_path"), field="scan_sample_path"
        )
        row_id = str(row["row_id"])
        if row_id in paths:
            raise ThreeStockK1FileRunnerError("duplicate aligned row identity")
        paths[row_id] = (digital_path, scan_path)
        rows.append(
            AlignedScanFileRow(
                row_id=row_id,
                stock_id=str(row["stock_id"]),
                role=str(row["role"]),
                scene_id=str(row["scene_id"]),
                film_frame_id=str(row["film_frame_id"]),
                roll_id=str(row["roll_id"]),
                source_identity=str(row["digital_reference_sha256"]),
                source_shape=_integer_rgb_shape(digital_path),
                scan_shape=_integer_rgb_shape(scan_path),
                homography_source_to_scan=np.asarray(
                    evidence.get("homography_source_to_scan"), dtype=np.float64
                ),
            )
        )
    return rows, paths


def _evaluate_files(
    *,
    root: Path,
    integrity_contract_path: Path,
    k1_contract_path: Path,
    ledger_path: Path,
    manifest_path: Path,
    stock: str | None,
) -> dict[str, Any]:
    """Run integrity, common-coordinate sampling, then the frozen K=1 experiment."""

    k1_raw, k1_contract = load_k1_contract(k1_contract_path, root=root)
    integrity_report = (
        evaluate_integrity(
            integrity_contract_path, ledger_path, manifest_path, root=root
        )
        if stock is None
        else evaluate_single_stock_integrity(
            integrity_contract_path,
            ledger_path,
            manifest_path,
            root=root,
            stock=stock,
        )
    )
    ledger_raw, ledger = _read_object(ledger_path)
    manifest_raw, manifest = _read_object(manifest_path)
    if not integrity_report["automatic_pass"]:
        core = {
            "schema": (REPORT_SCHEMA if stock is None else SINGLE_STOCK_REPORT_SCHEMA),
            "experiment_id": "SF3.A2" if stock is None else f"SF3.A2.{stock}",
            "k1_contract_sha256": _sha256(k1_raw),
            "ledger_sha256": _sha256(ledger_raw),
            "manifest_sha256": _sha256(manifest_raw),
            "integrity_stable_evidence_id": integrity_report["stable_evidence_id"],
            "integrity_automatic_pass": False,
            "paired_sampling_executed": False,
            "operator_fits": 0,
            "automatic_pass": False,
            "decision": integrity_report["decision"],
            "claim_ceiling": (
                k1_contract["claim_ceiling"]
                if stock is None
                else integrity_report["claim_ceiling"]
            ),
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
    result = (
        evaluate_k1(
            k1_contract_path,
            root=root,
            development=development,
            confirmation=confirmation,
        )
        if stock is None
        else evaluate_single_stock_k1(
            k1_contract_path,
            root=root,
            stock=stock,
            development=development.get(stock, []),
            confirmation=confirmation.get(stock, []),
        )
    )
    candidate_count = len(
        k1_contract["operator_selection"]["candidate_order_simplest_first"]
    )
    evaluated_stocks = k1_contract["required_stocks"] if stock is None else [stock]
    operator_fits = sum(
        len({frame.roll_id for frame in development[stock_id]}) * candidate_count + 1
        for stock_id in evaluated_stocks
    )
    core = {
        "schema": REPORT_SCHEMA if stock is None else SINGLE_STOCK_REPORT_SCHEMA,
        "experiment_id": "SF3.A2" if stock is None else f"SF3.A2.{stock}",
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
        "claim_ceiling": (
            k1_contract["claim_ceiling"] if stock is None else result["claim_ceiling"]
        ),
    }
    if stock is not None:
        core["stock"] = stock
        core["cross_stock_controls_evaluated"] = False
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


def evaluate_files(
    *,
    root: Path,
    integrity_contract_path: Path,
    k1_contract_path: Path,
    ledger_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run the complete three-stock file-backed A1 to A2 experiment."""

    return _evaluate_files(
        root=root,
        integrity_contract_path=integrity_contract_path,
        k1_contract_path=k1_contract_path,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        stock=None,
    )


def evaluate_single_stock_files(
    *,
    root: Path,
    integrity_contract_path: Path,
    k1_contract_path: Path,
    ledger_path: Path,
    manifest_path: Path,
    stock: str,
) -> dict[str, Any]:
    """Run one complete stock lane without claiming cross-stock evidence."""

    return _evaluate_files(
        root=root,
        integrity_contract_path=integrity_contract_path,
        k1_contract_path=k1_contract_path,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
        stock=stock,
    )


__all__ = [
    "REPORT_SCHEMA",
    "SINGLE_STOCK_REPORT_SCHEMA",
    "ThreeStockK1FileRunnerError",
    "evaluate_files",
    "evaluate_single_stock_files",
    "load_aligned_file_rows",
    "load_aligned_rows",
]
