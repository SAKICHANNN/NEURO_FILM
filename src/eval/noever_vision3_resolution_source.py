"""Exact range acquisition and integrity audit for U5.R2BU11."""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import requests
import tifffile

from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json

SCHEMA = "neuro_film.u5-r2bu11-noever-vision3-resolution-source-contract.v1"
REPORT_SCHEMA = "neuro_film.u5-r2bu11-noever-vision3-resolution-source-report.v1"
EXPERIMENT_ID = "U5.R2BU11"
_LOCAL_HEADER = struct.Struct("<4s5H3L2H")


class NoeverVision3SourceError(RuntimeError):
    """Raised when a frozen BU11 source or archive fact drifts."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise NoeverVision3SourceError("BU11 contract identity drift")
    rows = payload.get("source", {}).get("selected_members", [])
    if (
        len(rows) != 18
        or int(payload["source"].get("selected_member_count", -1)) != len(rows)
        or len({row["name"] for row in rows}) != 18
    ):
        raise NoeverVision3SourceError("BU11 selected member inventory drift")
    if sum(int(row["compressed_size"]) for row in rows) != int(
        payload["source"]["selected_compressed_bytes"]
    ):
        raise NoeverVision3SourceError("BU11 compressed-byte total drift")
    return payload


def _range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    response = session.get(
        url, headers={"Range": f"bytes={start}-{end}"}, timeout=120
    )
    if response.status_code != 206:
        raise NoeverVision3SourceError("BU11 server did not honor byte range")
    expected = f"bytes {start}-{end}/"
    if not response.headers.get("Content-Range", "").startswith(expected):
        raise NoeverVision3SourceError("BU11 content-range drift")
    if len(response.content) != end - start + 1:
        raise NoeverVision3SourceError("BU11 range length drift")
    return response.content


def _member_bytes(
    session: requests.Session, url: str, row: dict[str, Any]
) -> bytes:
    offset = int(row["local_header_offset"])
    header = _range(session, url, offset, offset + _LOCAL_HEADER.size - 1)
    values = _LOCAL_HEADER.unpack(header)
    if values[0] != b"PK\x03\x04" or values[3] != 8:
        raise NoeverVision3SourceError("BU11 local header or compression drift")
    name_length, extra_length = values[-2:]
    prefix_end = offset + _LOCAL_HEADER.size + name_length + extra_length - 1
    prefix = _range(session, url, offset + _LOCAL_HEADER.size, prefix_end)
    name = prefix[:name_length].decode("utf-8")
    if name != row["name"]:
        raise NoeverVision3SourceError("BU11 central/local filename mismatch")
    data_start = prefix_end + 1
    compressed = _range(
        session,
        url,
        data_start,
        data_start + int(row["compressed_size"]) - 1,
    )
    try:
        decoded = zlib.decompress(compressed, -zlib.MAX_WBITS)
    except zlib.error as exc:
        raise NoeverVision3SourceError("BU11 deflate decode failed") from exc
    if len(decoded) != int(row["uncompressed_size"]):
        raise NoeverVision3SourceError("BU11 uncompressed size drift")
    if f"{binascii.crc32(decoded) & 0xFFFFFFFF:08x}" != row["crc32"]:
        raise NoeverVision3SourceError("BU11 member CRC drift")
    return decoded


def _relative_output(name: str) -> Path:
    parts = Path(name).parts
    if len(parts) < 5 or parts[0] != "TIFF":
        raise NoeverVision3SourceError("BU11 member path drift")
    return Path(*parts[1:])


def _inspect_tiff(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise NoeverVision3SourceError("BU11 TIFF page count drift")
        page = tif.pages[0]
        array = page.asarray()
        if array.dtype != np.uint16 or array.ndim != 3 or array.shape[-1] != 3:
            raise NoeverVision3SourceError("BU11 TIFF is not RGB16")
        icc = page.tags.get(34675)
        icc_bytes = bytes(icc.value) if icc is not None else b""
        return {
            "width": int(array.shape[1]),
            "height": int(array.shape[0]),
            "dtype": str(array.dtype),
            "channel_count": int(array.shape[2]),
            "pixel_sha256": _sha256(array.tobytes(order="C")),
            "icc_sha256": _sha256(icc_bytes) if icc_bytes else None,
            "minimum_sample": int(np.min(array)),
            "maximum_sample": int(np.max(array)),
        }


def evaluate(
    contract: dict[str, Any], root: Path, *, acquire: bool
) -> dict[str, Any]:
    source = contract["source"]
    output_root = root / contract["acquisition"]["output_root"]
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    for selected in source["selected_members"]:
        relative = _relative_output(selected["name"])
        path = output_root / relative
        if acquire:
            payload = _member_bytes(session, source["download_url"], selected)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.bu11-stage-{os.getpid()}")
            if temporary.exists():
                temporary.unlink()
            try:
                with temporary.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        if not path.is_file() or path.is_symlink():
            raise NoeverVision3SourceError("BU11 acquired member is missing or unsafe")
        payload = path.read_bytes()
        if len(payload) != int(selected["uncompressed_size"]):
            raise NoeverVision3SourceError("BU11 retained size drift")
        if f"{binascii.crc32(payload) & 0xFFFFFFFF:08x}" != selected["crc32"]:
            raise NoeverVision3SourceError("BU11 retained CRC drift")
        facts = _inspect_tiff(path)
        parts = relative.parts
        rows.append(
            {
                "stock": parts[1],
                "format": parts[0],
                "scan_resolution": parts[2].removeprefix("Scans "),
                "relative_path": path.relative_to(root).as_posix(),
                "file_size": len(payload),
                "file_sha256": _sha256(payload),
                **facts,
            }
        )
    icc_by_resolution: dict[str, list[str | None]] = {}
    for row in rows:
        icc_by_resolution.setdefault(row["scan_resolution"], []).append(
            row["icc_sha256"]
        )
    metrics = {
        "member_count": len(rows),
        "stock_count": len({row["stock"] for row in rows}),
        "format_count": len({row["format"] for row in rows}),
        "scan_resolution_count": len({row["scan_resolution"] for row in rows}),
        "decode_count": len(rows),
        "uint16_count": sum(row["dtype"] == "uint16" for row in rows),
        "rgb_count": sum(row["channel_count"] == 3 for row in rows),
        "icc_consistent_within_scan_resolution": all(
            len(set(values)) == 1 for values in icc_by_resolution.values()
        ),
        "retained_bytes": sum(row["file_size"] for row in rows),
    }
    gates = contract["integrity_gates"]
    checks = {
        "member_count": metrics["member_count"] == gates["required_member_count"],
        "stock_count": metrics["stock_count"] == gates["required_stock_count"],
        "format_count": metrics["format_count"] == gates["required_format_count"],
        "scan_resolution_count": metrics["scan_resolution_count"]
        == gates["required_scan_resolution_count"],
        "decode_count": metrics["decode_count"] == gates["required_decode_count"],
        "uint16_count": metrics["uint16_count"] == gates["required_uint16_count"],
        "rgb_count": metrics["rgb_count"] == gates["required_rgb_count"],
        "icc_consistency": metrics["icc_consistent_within_scan_resolution"]
        is gates["required_icc_consistency_within_scan_resolution"],
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": hashlib.sha256(
            canonical_json(contract)
        ).hexdigest(),
        "acquisition_performed": acquire,
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["branch_rule"][
            "pass" if all(checks.values()) else "fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable = dict(report)
    stable.pop("acquisition_performed")
    report["stable_evidence_id"] = _sha256(canonical_json(stable))
    return report


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)


__all__ = [
    "NoeverVision3SourceError",
    "evaluate",
    "load_contract",
    "write_report",
]
