#!/usr/bin/env python3
"""Range-only structural audit for the official SPCP preference dataset."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import re
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

ARCHIVE_URL = (
    "https://huggingface.co/datasets/zwx8981/SPCP_dataset/resolve/"
    "068af97eed82969f15278db3af4bd450176cf6f3/SPCP_dataset.zip"
)
CENTRAL_OFFSET = 9_049_028_973
CENTRAL_SIZE = 1_433_756
CENTRAL_SHA256 = "f3892690a8434fd42a412c4cf952e050d542740d2dea04b4cad976ad2ea1415d"
ARCHIVE_SIZE = 9_050_462_827
EXPECTED_XLSX = {
    "SPCP_dataset/order_trans.xlsx": {
        "sha256": "8c42140ae0f90f37f32706911ab86cca9f377077bbd18ac301262d952bf5f58c",
        "rows": 45_001,
        "columns": 21,
    },
    "SPCP_dataset/score_trans2.xlsx": {
        "sha256": "ee5f0fc830ebd40cab3e25b379aa0e01ec5cc4793a55315504c2d06f6af7900d",
        "rows": 12_001,
        "columns": 21,
    },
}
CANONICAL_PNG = re.compile(r"(?:^|/)(I\d{4}_\d{2}_\d{2})\.png$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def range_get(url: str, start: int, end: int) -> bytes:
    if start < 0 or end < start or end >= ARCHIVE_SIZE:
        raise ValueError("invalid bounded archive range")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "NeuroFilm-U5-R2SPCP0/1.0 (range-only research audit)",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        content_range = response.headers.get("Content-Range")
        expected_range = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
        if status != 206 or content_range != expected_range:
            raise RuntimeError(
                f"server did not honor exact range: status={status}, "
                f"content_range={content_range!r}"
            )
        expected_length = end - start + 1
        content_length = response.headers.get("Content-Length")
        if content_length is None or int(content_length) != expected_length:
            raise RuntimeError("range Content-Length differs")
        payload = response.read(expected_length + 1)
    if len(payload) != expected_length:
        raise RuntimeError("range payload length differs")
    return payload


@dataclass(frozen=True)
class ZipMember:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def _zip64_values(extra: bytes) -> list[int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        value = extra[cursor : cursor + size]
        cursor += size
        if kind == 0x0001:
            if len(value) % 8:
                raise ValueError("invalid ZIP64 extra field")
            return list(struct.unpack(f"<{len(value) // 8}Q", value))
    return []


def parse_central_directory(payload: bytes) -> list[ZipMember]:
    members: list[ZipMember] = []
    cursor = 0
    while cursor < len(payload):
        if payload[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError(f"invalid central-directory signature at {cursor}")
        if cursor + 46 > len(payload):
            raise ValueError("truncated central-directory entry")
        values = struct.unpack_from("<4s6H3I5H2I", payload, cursor)
        (
            _,
            _,
            _,
            flags,
            method,
            _,
            _,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            _,
            _,
            _,
            local_offset,
        ) = values
        body = cursor + 46
        name_bytes = payload[body : body + name_length]
        extra = payload[body + name_length : body + name_length + extra_length]
        end = body + name_length + extra_length + comment_length
        if end > len(payload):
            raise ValueError("truncated central-directory body")
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        zip64 = iter(_zip64_values(extra))
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = next(zip64)
        if compressed_size == 0xFFFFFFFF:
            compressed_size = next(zip64)
        if local_offset == 0xFFFFFFFF:
            local_offset = next(zip64)
        members.append(
            ZipMember(
                name=name,
                flags=flags,
                method=method,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_offset=local_offset,
            )
        )
        cursor = end
    return members


def extract_member(url: str, member: ZipMember) -> bytes:
    header = range_get(url, member.local_offset, member.local_offset + 29)
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04":
        raise ValueError(f"invalid local header: {member.name}")
    flags, method = values[2], values[3]
    name_length, extra_length = values[-2], values[-1]
    if flags != member.flags or method != member.method:
        raise ValueError(f"local/central metadata mismatch: {member.name}")
    variable = range_get(
        url,
        member.local_offset + 30,
        member.local_offset + 30 + name_length + extra_length - 1,
    )
    name = variable[:name_length].decode("utf-8" if flags & 0x800 else "cp437")
    if name != member.name:
        raise ValueError(f"local/central filename mismatch: {member.name}")
    data_start = member.local_offset + 30 + name_length + extra_length
    compressed = range_get(
        url,
        data_start,
        data_start + member.compressed_size - 1,
    )
    if method == 0:
        raw = compressed
    elif method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported ZIP method {method}: {member.name}")
    if len(raw) != member.uncompressed_size:
        raise ValueError(f"uncompressed size mismatch: {member.name}")
    if binascii.crc32(raw) & 0xFFFFFFFF != member.crc32:
        raise ValueError(f"CRC mismatch: {member.name}")
    return raw


def workbook_facts(path: Path) -> dict[str, Any]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    if len(workbook.sheetnames) != 1:
        raise ValueError(f"workbook sheet count differs: {path.name}")
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"empty workbook: {path.name}")
    return {
        "sheet_name": sheet.title,
        "row_count": len(rows),
        "column_count": max(len(row) for row in rows),
        "header": list(rows[0]),
        "first_data_rows": [list(row) for row in rows[1:4]],
        "all_cells_nonempty": all(value is not None for row in rows for value in row),
    }


def run(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    central = range_get(
        ARCHIVE_URL,
        CENTRAL_OFFSET,
        CENTRAL_OFFSET + CENTRAL_SIZE - 1,
    )
    if sha256_bytes(central) != CENTRAL_SHA256:
        raise ValueError("central-directory hash mismatch")
    members = parse_central_directory(central)
    by_name = {member.name: member for member in members}
    if len(by_name) != len(members):
        raise ValueError("duplicate ZIP member names")

    annotation_facts: dict[str, Any] = {}
    for filename, expected in EXPECTED_XLSX.items():
        member = by_name.get(filename)
        if member is None:
            raise ValueError(f"missing annotation member: {filename}")
        raw = extract_member(ARCHIVE_URL, member)
        if sha256_bytes(raw) != expected["sha256"]:
            raise ValueError(f"annotation SHA mismatch: {filename}")
        path = output_root / Path(filename).name
        path.write_bytes(raw)
        facts = workbook_facts(path)
        if facts["row_count"] != expected["rows"]:
            raise ValueError(f"annotation row count mismatch: {filename}")
        if facts["column_count"] != expected["columns"]:
            raise ValueError(f"annotation column count mismatch: {filename}")
        annotation_facts[filename] = {
            "member": member.__dict__,
            "sha256": sha256_bytes(raw),
            **facts,
        }

    png_names = [member.name for member in members if member.name.lower().endswith(".png")]
    canonical_ids = []
    noncanonical_png = []
    for name in png_names:
        match = CANONICAL_PNG.search(name)
        if match:
            canonical_ids.append(match.group(1))
        else:
            noncanonical_png.append(name)
    report: dict[str, Any] = {
        "schema": "neuro-film.u5-r2spcp0-range-audit-inspection.v1",
        "status": "ANNOTATIONS_EXTRACTED_STRUCTURE_PENDING_FINAL_INTERPRETATION",
        "archive_url": ARCHIVE_URL,
        "archive_size_bytes": ARCHIVE_SIZE,
        "central_directory": {
            "offset": CENTRAL_OFFSET,
            "size": CENTRAL_SIZE,
            "sha256": sha256_bytes(central),
            "member_count": len(members),
        },
        "inventory": {
            "png_member_count": len(png_names),
            "canonical_png_identity_count": len(set(canonical_ids)),
            "canonical_png_member_count": len(canonical_ids),
            "noncanonical_png_members": sorted(noncanonical_png),
        },
        "annotations": annotation_facts,
        "reads": {
            "image_member_payload_bytes": 0,
            "operator_fit_count": 0,
            "render_count": 0,
            "scientific_score_count": 0,
        },
    }
    report["report_id"] = canonical_sha256(report)
    (output_root / "inspection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_root.resolve())
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "member_count": report["central_directory"]["member_count"],
                "inventory": report["inventory"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
