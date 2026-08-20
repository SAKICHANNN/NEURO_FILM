"""Selective ZIP acquisition for the U6.P6AU Vision3 resolution charts."""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

SCHEMA = "neuro-film.u6-p6au-film-resolution-chart-source-contract.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported P6AU contract")
    selection = contract["selection"]
    members = selection["members"]
    if (
        len(members) != selection["expected_members"]
        or sum(int(row["compressed_size"]) for row in members)
        != selection["expected_compressed_bytes"]
        or sum(int(row["uncompressed_size"]) for row in members)
        != selection["expected_uncompressed_bytes"]
        or len({row["name"] for row in members}) != len(members)
    ):
        raise ValueError("P6AU selected inventory drift")
    cells = {tuple(row["name"].split("/")[1:3]) for row in members}
    if len(cells) != selection["expected_groups"]:
        raise ValueError("P6AU group inventory drift")
    return contract


def _range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError("P6AU range response length drift")
    return data


def _central_rows(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pos = 0
    while pos + 46 <= len(data) and data[pos : pos + 4] == b"PK\x01\x02":
        fields = struct.unpack_from("<6H3I5H2I", data, pos + 4)
        name_len, extra_len, comment_len = fields[9:12]
        name = data[pos + 46 : pos + 46 + name_len].decode("utf-8")
        rows.append(
            {
                "name": name,
                "flags": fields[2],
                "method": fields[3],
                "crc32": f"{fields[6]:08x}",
                "compressed_size": fields[7],
                "uncompressed_size": fields[8],
                "local_header_offset": fields[15],
            }
        )
        pos += 46 + name_len + extra_len + comment_len
    if pos != len(data):
        raise ValueError("P6AU central directory parse drift")
    return rows


def _validate_tiff(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as image:
        if len(image.pages) != 1:
            raise ValueError("P6AU TIFF must have exactly one page")
        page = image.pages[0]
        values = page.asarray()
        bits = tuple(int(value) for value in np.atleast_1d(page.bitspersample))
        if values.dtype != np.uint16 or values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("P6AU TIFF is not RGB16")
        if bits not in {(16,), (16, 16, 16)}:
            raise ValueError("P6AU TIFF bit depth drift")
        return {
            "shape": [int(value) for value in values.shape],
            "dtype": str(values.dtype),
            "minimum_code": int(values.min()),
            "maximum_code": int(values.max()),
            "photometric": str(page.photometric.name),
        }


def _fetch_member(url: str, row: dict[str, Any], destination: Path) -> dict[str, Any]:
    target = destination / row["name"]
    if target.exists():
        payload = target.read_bytes()
    else:
        start = int(row["local_header_offset"])
        blob = _range(url, start, start + int(row["compressed_size"]) + 4095)
        if blob[:4] != b"PK\x03\x04":
            raise ValueError("P6AU local header signature drift")
        fields = struct.unpack_from("<5H3I2H", blob, 4)
        flags, method = fields[1], fields[2]
        local_crc, local_compressed, local_uncompressed = fields[5:8]
        name_len, extra_len = fields[8:10]
        name = blob[30 : 30 + name_len].decode("utf-8")
        expected_local = (
            int(row["crc32"], 16),
            int(row["compressed_size"]),
            int(row["uncompressed_size"]),
        )
        observed_local = (local_crc, local_compressed, local_uncompressed)
        sizes_match = observed_local == expected_local
        descriptor_placeholders_match = bool(flags & 0x08) and all(
            observed in {0, expected}
            for observed, expected in zip(observed_local, expected_local, strict=True)
        )
        if (
            name != row["name"]
            or flags != int(row["flags"])
            or method != int(row["method"])
            or not (descriptor_placeholders_match or sizes_match)
        ):
            raise ValueError("P6AU central/local member drift")
        data_start = 30 + name_len + extra_len
        packed = blob[data_start : data_start + int(row["compressed_size"])]
        payload = zlib.decompress(packed, -15) if method == 8 else packed
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
    if (
        len(payload) != int(row["uncompressed_size"])
        or f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}" != row["crc32"]
    ):
        raise ValueError("P6AU member payload drift")
    return {
        "name": row["name"],
        "bytes": len(payload),
        "sha256": _sha(payload),
        "tiff": _validate_tiff(target),
    }


def acquire(contract: dict[str, Any], destination: Path) -> dict[str, Any]:
    source = contract["source"]
    central = _range(
        source["archive_url"],
        int(source["central_directory_offset"]),
        int(source["central_directory_offset"])
        + int(source["central_directory_size"])
        - 1,
    )
    if _sha(central) != source["central_directory_sha256"]:
        raise ValueError("P6AU central directory identity drift")
    central_rows = _central_rows(central)
    if len(central_rows) != int(source["expected_central_entries"]):
        raise ValueError("P6AU central entry count drift")
    real_tiffs = [
        row
        for row in central_rows
        if row["name"].startswith("TIFF/") and row["name"].endswith(".tif")
    ]
    if len(real_tiffs) != int(source["expected_real_tiff_members"]):
        raise ValueError("P6AU TIFF inventory drift")
    by_name = {row["name"]: row for row in central_rows}
    members = contract["selection"]["members"]
    for expected in members:
        if by_name.get(expected["name"]) != expected:
            raise ValueError("P6AU selected central identity drift")
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(
            executor.map(
                lambda row: _fetch_member(source["archive_url"], row, destination),
                members,
            )
        )
    return {
        "schema": "neuro-film.u6-p6au-film-resolution-chart-source-report.v1",
        "experiment_id": contract["experiment_id"],
        "central_directory_sha256": _sha(central),
        "outputs": outputs,
        "aggregate": {
            "members": len(outputs),
            "groups": int(contract["selection"]["expected_groups"]),
            "bytes": sum(row["bytes"] for row in outputs),
            "all_rgb16": all(row["tiff"]["dtype"] == "uint16" for row in outputs),
        },
        "automatic_pass": True,
        "decision": contract["decision_if_pass"],
        "claim_ceiling": contract["claim_ceiling"],
    }
