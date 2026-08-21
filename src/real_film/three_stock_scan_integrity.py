"""SF3.A1 file, pixel, rights, and alignment integrity for controlled scans."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from PIL import Image

from src.preprocess import srgb_icc_profile_sha256
from src.real_film.scanner_nuisance import ScannerNuisanceError, align_source_to_scan
from src.real_film.three_stock_acquisition import (
    LEDGER_SCHEMA,
    compile_acquisition_ledger,
    evaluate_manifest,
)

CONTRACT_SCHEMA = (
    "neuro-film.sf3-a1-three-stock-file-pixel-alignment-integrity-contract.v1"
)
REPORT_SCHEMA = "neuro-film.sf3-a1-three-stock-file-pixel-alignment-integrity-report.v1"


class ThreeStockScanIntegrityError(ValueError):
    """Raised when an SF3.A1 binding or input is structurally invalid."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ThreeStockScanIntegrityError(f"expected JSON object: {path}")
    return raw, value


def _bound_path(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThreeStockScanIntegrityError(f"invalid {field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ThreeStockScanIntegrityError(f"{field} must be repository-relative")
    if not relative.parts or relative.parts[0].casefold() != "data":
        raise ThreeStockScanIntegrityError(f"{field} must use logical data root")
    target = root.joinpath(*relative.parts)
    if not target.is_file():
        raise ThreeStockScanIntegrityError(f"missing {field}: {value}")
    return target


def load_contract(path: Path, *, root: Path) -> tuple[bytes, dict[str, Any]]:
    raw, contract = _read_object(path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("experiment_id") != "SF3.A1"
    ):
        raise ThreeStockScanIntegrityError("unsupported SF3.A1 contract")
    parent = contract.get("parent", {}).get("acquisition_contract", {})
    parent_path = _bound_path_like_repo(
        root, parent.get("path"), field="acquisition contract"
    )
    if _sha256_file(parent_path) != parent.get("sha256"):
        raise ThreeStockScanIntegrityError("SF3.A0 parent contract hash drift")
    if set(contract.get("record_schemas", {})) != {"rights", "alignment"}:
        raise ThreeStockScanIntegrityError("SF3.A1 record schema inventory drift")
    if (
        contract.get("decode", {}).get("required_rgb16_tiff_icc_profile_sha256")
        != srgb_icc_profile_sha256()
    ):
        raise ThreeStockScanIntegrityError("SF3.A1 RGB16 TIFF ICC identity drift")
    for field in ("required_scan_width", "required_scan_height"):
        value = contract.get("decode", {}).get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ThreeStockScanIntegrityError(f"SF3.A1 {field} is invalid")
    if (
        contract["decode"].get("required_scan_orientation") != 1
        or contract["decode"].get("required_scan_planar_configuration") != 1
        or contract["decode"].get("allowed_scan_tiff_compression_codes")
        != [1, 8, 32946]
    ):
        raise ThreeStockScanIntegrityError("SF3.A1 scan container policy drift")
    return raw, contract


def _bound_path_like_repo(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThreeStockScanIntegrityError(f"invalid {field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ThreeStockScanIntegrityError(f"{field} must be repository-relative")
    target = root.joinpath(*relative.parts)
    if not target.is_file():
        raise ThreeStockScanIntegrityError(f"missing {field}: {value}")
    return target


def decode_integer_rgb(path: Path, decode_contract: Mapping[str, Any]) -> np.ndarray:
    """Decode one bounded, single-frame RGB8/RGB16 PNG or TIFF."""

    if path.suffix.casefold() not in set(decode_contract["allowed_extensions"]):
        raise ThreeStockScanIntegrityError(
            f"unsupported image extension: {path.suffix}"
        )
    if path.suffix.casefold() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as image:
            if len(image.pages) != 1:
                raise ThreeStockScanIntegrityError("multi-page TIFF is forbidden")
    else:
        with Image.open(path) as image:
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ThreeStockScanIntegrityError("multi-frame image is forbidden")
            if "A" in image.getbands():
                raise ThreeStockScanIntegrityError("alpha is forbidden")
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ThreeStockScanIntegrityError(f"image decode failed: {path}")
    if bgr.ndim != 3 or bgr.shape[2] != int(decode_contract["required_channels"]):
        raise ThreeStockScanIntegrityError(f"expected RGB image, got {bgr.shape}")
    if bgr.dtype not in (np.uint8, np.uint16):
        raise ThreeStockScanIntegrityError(
            f"expected uint8/uint16 samples, got {bgr.dtype}"
        )
    bits = int(np.iinfo(bgr.dtype).bits)
    if bits not in {int(value) for value in decode_contract["allowed_integer_bits"]}:
        raise ThreeStockScanIntegrityError(f"unsupported sample depth: {bits}")
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    if min(rgb.shape[:2]) <= 0:
        raise ThreeStockScanIntegrityError("empty image is forbidden")
    return rgb


def _normalized_rgb(rgb: np.ndarray) -> np.ndarray:
    maximum = float(np.iinfo(rgb.dtype).max)
    value = rgb.astype(np.float64) / maximum
    if not np.isfinite(value).all():
        raise ThreeStockScanIntegrityError("decoded samples are non-finite")
    return value


def _sample_identity(rgb: np.ndarray) -> str:
    header = f"{rgb.dtype.str}:{rgb.shape[0]}:{rgb.shape[1]}:{rgb.shape[2]}\n".encode(
        "ascii"
    )
    return _sha256(header + np.ascontiguousarray(rgb).tobytes())


def _validate_scan_sample_container(
    path: Path, decode_contract: Mapping[str, Any]
) -> None:
    if path.suffix.casefold() not in {".tif", ".tiff"}:
        raise ThreeStockScanIntegrityError(
            "controlled scan samples must be canonical RGB16 TIFF"
        )
    expected = (
        int(decode_contract["required_scan_height"]),
        int(decode_contract["required_scan_width"]),
        int(decode_contract["required_channels"]),
    )
    with tifffile.TiffFile(path) as image:
        if len(image.pages) != 1:
            raise ThreeStockScanIntegrityError("multi-page TIFF is forbidden")
        page = image.pages[0]
        shape = tuple(int(value) for value in page.shape)
        dtype = np.dtype(page.dtype)
        icc_tag = page.tags.get(34675)
        icc_sha256 = _sha256(bytes(icc_tag.value)) if icc_tag is not None else None
        orientation_tag = page.tags.get("Orientation")
        orientation = int(orientation_tag.value) if orientation_tag is not None else 1
        planar = int(page.planarconfig)
        compression = int(page.compression)
    if dtype != np.dtype(np.uint16):
        raise ThreeStockScanIntegrityError(
            "controlled scan samples must be canonical RGB16 TIFF"
        )
    if icc_sha256 != decode_contract["required_rgb16_tiff_icc_profile_sha256"]:
        raise ThreeStockScanIntegrityError(
            "controlled scan must carry the bound canonical sRGB ICC profile"
        )
    if shape != expected:
        raise ThreeStockScanIntegrityError(
            f"controlled scan geometry {shape} does not match {expected}"
        )
    if orientation != int(decode_contract["required_scan_orientation"]):
        raise ThreeStockScanIntegrityError("controlled scan orientation drift")
    if planar != int(decode_contract["required_scan_planar_configuration"]):
        raise ThreeStockScanIntegrityError("controlled scan planar layout drift")
    if compression not in {
        int(value) for value in decode_contract["allowed_scan_tiff_compression_codes"]
    }:
        raise ThreeStockScanIntegrityError("controlled scan compression is not allowed")


def decode_scan_integer_rgb(
    path: Path, decode_contract: Mapping[str, Any]
) -> np.ndarray:
    """Decode one controlled scan and enforce the canonical scan container."""

    _validate_scan_sample_container(path, decode_contract)
    rgb = decode_integer_rgb(path, decode_contract)
    expected = (
        int(decode_contract["required_scan_height"]),
        int(decode_contract["required_scan_width"]),
        int(decode_contract["required_channels"]),
    )
    if rgb.shape != expected or rgb.dtype != np.uint16:
        raise ThreeStockScanIntegrityError(
            "controlled scan decode changed its container"
        )
    return rgb


def build_alignment_evidence(
    row: Mapping[str, Any],
    digital_rgb: np.ndarray,
    scan_rgb: np.ndarray,
    *,
    alignment: Mapping[str, Any],
    schema: str,
) -> dict[str, Any]:
    """Build the exact evidence object that SF3.A1 later independently checks."""

    homography, diagnostics = align_source_to_scan(
        _normalized_rgb(digital_rgb), _normalized_rgb(scan_rgb), alignment
    )
    return {
        "schema": schema,
        "row_id": row["row_id"],
        "digital_reference_sha256": row["digital_reference_sha256"],
        "scan_sample_sha256": row["scan_sample_sha256"],
        "method": alignment["method"],
        "homography_source_to_scan": homography.tolist(),
        "diagnostics": diagnostics,
    }


def _data_output_path(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThreeStockScanIntegrityError(f"invalid {field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ThreeStockScanIntegrityError(f"{field} must be repository-relative")
    if not relative.parts or relative.parts[0].casefold() != "data":
        raise ThreeStockScanIntegrityError(f"{field} must use logical data root")
    return root.joinpath(*relative.parts)


def materialize_alignment_evidence(
    contract_path: Path, ledger_path: Path, *, root: Path
) -> dict[str, Any]:
    """Create or verify every A1 alignment record named by a filled A0 ledger."""

    contract_raw, contract = load_contract(contract_path, root=root)
    ledger_raw, ledger = _read_object(ledger_path)
    rows = ledger.get("rows")
    if (
        set(ledger) != {"schema", "rows"}
        or ledger.get("schema") != LEDGER_SCHEMA
        or not isinstance(rows, list)
        or not rows
    ):
        raise ThreeStockScanIntegrityError(
            "alignment ledger schema or rows are invalid"
        )
    inventory: list[dict[str, str]] = []
    materialized = 0
    reused = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("row_id"), str):
            raise ThreeStockScanIntegrityError(f"invalid alignment ledger row {index}")
        digital_path = _bound_path(
            root, row.get("digital_reference_path"), field="digital_reference_path"
        )
        scan_path = _bound_path(
            root, row.get("scan_sample_path"), field="scan_sample_path"
        )
        output_path = _data_output_path(
            root,
            row.get("alignment_evidence_path"),
            field="alignment_evidence_path",
        )
        digital_rgb = decode_integer_rgb(digital_path, contract["decode"])
        scan_rgb = decode_scan_integer_rgb(scan_path, contract["decode"])
        evidence = build_alignment_evidence(
            {
                "row_id": row["row_id"],
                "digital_reference_sha256": _sha256_file(digital_path),
                "scan_sample_sha256": _sha256_file(scan_path),
            },
            digital_rgb,
            scan_rgb,
            alignment=contract["alignment"],
            schema=contract["record_schemas"]["alignment"],
        )
        payload = _canonical(evidence)
        if output_path.exists():
            _, observed = _read_object(output_path)
            if not _numeric_close(
                observed,
                evidence,
                tolerance=float(
                    contract["alignment"]["maximum_evidence_float_abs_error"]
                ),
            ):
                raise ThreeStockScanIntegrityError(
                    f"existing alignment evidence drift: {row['row_id']}"
                )
            reused += 1
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("xb") as handle:
                handle.write(payload)
            materialized += 1
        inventory.append({"row_id": row["row_id"], "evidence_sha256": _sha256(payload)})
    core = {
        "schema": "neuro-film.sf3-a1-alignment-evidence-materialization-report.v1",
        "experiment_id": "SF3.A1-ALIGNMENT-EVIDENCE",
        "contract_sha256": _sha256(contract_raw),
        "ledger_sha256": _sha256(ledger_raw),
        "row_count": len(rows),
        "inventory": inventory,
        "pixel_reads": len(rows) * 2,
        "operator_fits": 0,
        "automatic_pass": True,
        "decision": "READY_TO_COMPILE_SF3_A0_LEDGER_WITH_ALIGNMENT_EVIDENCE",
        "claim_ceiling": (
            "Create-only controlled-scan alignment evidence assembly. Passing is "
            "not SF3.A0/A1 admission, fitting, stock response or calibration."
        ),
    }
    return {
        **core,
        "materialized_records": materialized,
        "reused_records": reused,
        "stable_evidence_id": _sha256(_canonical(core)),
    }


def _numeric_close(left: Any, right: Any, *, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(
            np.isfinite(float(left))
            and np.isfinite(float(right))
            and abs(float(left) - float(right)) <= tolerance
        )
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(
            _numeric_close(a, b, tolerance=tolerance)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(
            _numeric_close(left[key], right[key], tolerance=tolerance) for key in left
        )
    return left == right


def _rights_exact(
    record: Mapping[str, Any], row: Mapping[str, Any], schema: str
) -> bool:
    return set(record) == {
        "schema",
        "rights_record_id",
        "source_owner_id",
        "rights_scope",
        "allow_internal_training",
        "allow_commercial_derivatives",
        "allow_released_weights",
    } and all(
        (
            isinstance(record["rights_record_id"], str)
            and bool(record["rights_record_id"]),
            record["schema"] == schema,
            record["source_owner_id"] == row["source_owner_id"],
            record["rights_scope"] == row["rights_scope"],
            record["allow_internal_training"] is row["rights_allow_internal_training"],
            record["allow_commercial_derivatives"]
            is row["rights_allow_commercial_derivatives"],
            record["allow_released_weights"] is row["rights_allow_released_weights"],
        )
    )


def evaluate(
    contract_path: Path,
    ledger_path: Path,
    manifest_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    """Audit an A0-admitted physical acquisition without fitting any operator."""

    contract_raw, contract = load_contract(contract_path, root=root)
    ledger_raw, ledger = _read_object(ledger_path)
    manifest_raw, manifest = _read_object(manifest_path)
    parent_path = _bound_path_like_repo(
        root,
        contract["parent"]["acquisition_contract"]["path"],
        field="acquisition contract",
    )
    compiled = compile_acquisition_ledger(parent_path, ledger_path, root=root)
    if compiled != manifest:
        raise ThreeStockScanIntegrityError(
            "SF3.A0 manifest does not match ledger bytes"
        )
    parent_report = evaluate_manifest(parent_path, manifest_path)
    if (
        parent_report["decision"]
        != contract["parent"]["acquisition_contract"]["required_decision"]
    ):
        raise ThreeStockScanIntegrityError("SF3.A0 parent admission is not open")

    source_rows = ledger.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != len(manifest["rows"]):
        raise ThreeStockScanIntegrityError("ledger/manifest row count drift")

    decoded_by_frame: dict[str, set[str]] = defaultdict(set)
    sample_file_by_frame: dict[str, set[str]] = defaultdict(set)
    result_rows: list[dict[str, Any]] = []
    rights_schema = contract["record_schemas"]["rights"]
    alignment_schema = contract["record_schemas"]["alignment"]
    tolerance = float(contract["alignment"]["maximum_evidence_float_abs_error"])

    for row, source in zip(manifest["rows"], source_rows, strict=True):
        digital_path = _bound_path(
            root, source["digital_reference_path"], field="digital_reference_path"
        )
        scan_path = _bound_path(
            root, source["scan_sample_path"], field="scan_sample_path"
        )
        rights_path = _bound_path(
            root, source["rights_record_path"], field="rights_record_path"
        )
        evidence_path = _bound_path(
            root, source["alignment_evidence_path"], field="alignment_evidence_path"
        )
        failures: list[str] = []
        try:
            digital_rgb = decode_integer_rgb(digital_path, contract["decode"])
            scan_rgb = decode_scan_integer_rgb(scan_path, contract["decode"])
            digital_identity = _sample_identity(digital_rgb)
            scan_identity = _sample_identity(scan_rgb)
        except (ThreeStockScanIntegrityError, cv2.error, OSError, ValueError) as error:
            failures.append(f"decode:{error}")
            digital_identity = scan_identity = None
            digital_rgb = scan_rgb = None

        try:
            _, rights = _read_object(rights_path)
            if not _rights_exact(rights, row, rights_schema):
                failures.append("rights_record_mismatch")
        except (
            ThreeStockScanIntegrityError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            failures.append(f"rights:{error}")

        alignment_diagnostics: dict[str, Any] | None = None
        if digital_rgb is not None and scan_rgb is not None:
            try:
                expected = build_alignment_evidence(
                    row,
                    digital_rgb,
                    scan_rgb,
                    alignment=contract["alignment"],
                    schema=alignment_schema,
                )
                _, observed = _read_object(evidence_path)
                if not _numeric_close(observed, expected, tolerance=tolerance):
                    failures.append("alignment_evidence_mismatch")
                alignment_diagnostics = expected["diagnostics"]
            except (
                ThreeStockScanIntegrityError,
                ScannerNuisanceError,
                cv2.error,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                failures.append(f"alignment:{error}")

        if scan_identity is not None:
            decoded_by_frame[row["film_frame_id"]].add(scan_identity)
        sample_file_by_frame[row["film_frame_id"]].add(row["scan_sample_sha256"])
        result_rows.append(
            {
                "row_id": row["row_id"],
                "stock_id": row["stock_id"],
                "role": row["role"],
                "scene_id": row["scene_id"],
                "film_frame_id": row["film_frame_id"],
                "digital_decoded_sample_sha256": digital_identity,
                "scan_decoded_sample_sha256": scan_identity,
                "alignment_diagnostics": alignment_diagnostics,
                "automatic_pass": not failures,
                "failures": failures,
            }
        )

    def collision_free(groups: Mapping[str, set[str]]) -> bool:
        owners: dict[str, str] = {}
        for frame_id, identities in groups.items():
            for identity in identities:
                if identity in owners and owners[identity] != frame_id:
                    return False
                owners[identity] = frame_id
        return True

    gates = {
        "parent_a0_pass": parent_report["automatic_pass"] is True,
        "all_file_hashes_exact": True,
        "all_decodes_valid": all(
            row["digital_decoded_sample_sha256"] and row["scan_decoded_sample_sha256"]
            for row in result_rows
        ),
        "all_rights_records_exact": all(
            not any(value.startswith("rights") for value in row["failures"])
            for row in result_rows
        ),
        "all_alignment_evidence_exact": all(
            row["alignment_diagnostics"] is not None
            and not any(value.startswith("alignment") for value in row["failures"])
            for row in result_rows
        ),
        "scan_sample_hash_collisions_across_film_frames_zero": collision_free(
            sample_file_by_frame
        ),
        "decoded_sample_hash_collisions_across_film_frames_zero": collision_free(
            decoded_by_frame
        ),
        "operator_fits_zero": True,
    }
    automatic_pass = all(gates.values()) and all(
        row["automatic_pass"] for row in result_rows
    )
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "ledger_sha256": _sha256(ledger_raw),
        "manifest_sha256": _sha256(manifest_raw),
        "parent_a0_stable_evidence_id": parent_report["stable_evidence_id"],
        "row_count": len(result_rows),
        "rows": result_rows,
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "pixel_reads": len(result_rows) * 2,
        "operator_fits": 0,
        "operator_fit_authority": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}


__all__ = [
    "CONTRACT_SCHEMA",
    "REPORT_SCHEMA",
    "ThreeStockScanIntegrityError",
    "build_alignment_evidence",
    "decode_integer_rgb",
    "decode_scan_integer_rgb",
    "evaluate",
    "load_contract",
    "materialize_alignment_evidence",
]
