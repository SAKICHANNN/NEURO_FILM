#!/usr/bin/env python3
"""Range-acquire only the frozen SPCP2 fit/calibration winner-loser pairs."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_u5_r2spcp0_pairwise_preference_source_lock import (
    ROOT,
    _canonical_bytes,
    _fetch,
    _sha256,
    _stable_id,
)

DEFAULT_CONTRACT = ROOT / "configs/u5_r2spcp2_global_logit_affine_preference_d0_v1.json"
DEFAULT_MANIFEST = ROOT / "manifests/u5_r2spcp2_global_logit_affine_roles_v1.json"
DEFAULT_REPORT = ROOT / "outputs/eval/u5_r2spcp2_global_logit_affine_v1/acquisition.json"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _validate_payload(data: bytes, member: dict[str, Any]) -> None:
    if len(data) != int(member["uncompressed_size"]):
        raise ValueError(f"payload size drift: {member['name']}")
    if f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != member["crc32_hex"]:
        raise ValueError(f"payload CRC drift: {member['name']}")
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"payload is not PNG: {member['name']}")


def _fetch_member(zip_url: str, member: dict[str, Any]) -> tuple[bytes, int]:
    local_offset = int(member["local_offset"])
    header, _ = _fetch(zip_url, (local_offset, local_offset + 29))
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"local ZIP signature drift: {member['name']}")
    values = struct.unpack_from("<IHHHHHIIIHH", header)
    method, name_len, extra_len = values[3], values[9], values[10]
    if method != 8 or int(member["method"]) != 8:
        raise ValueError(f"compression method drift: {member['name']}")
    total = 30 + name_len + extra_len + int(member["compressed_size"])
    local_record, _ = _fetch(zip_url, (local_offset, local_offset + total - 1))
    encoded_name = local_record[30 : 30 + name_len].decode("utf-8")
    if encoded_name != member["name"]:
        raise ValueError(f"local member name drift: {member['name']}")
    start = 30 + name_len + extra_len
    compressed = local_record[start : start + int(member["compressed_size"])]
    data = zlib.decompress(compressed, -15)
    _validate_payload(data, member)
    return data, len(header) + len(local_record)


def _fetch_member_prefix(zip_url: str, member: dict[str, Any]) -> bytes:
    local_offset = int(member["local_offset"])
    header, _ = _fetch(zip_url, (local_offset, local_offset + 29))
    values = struct.unpack_from("<IHHHHHIIIHH", header)
    method, name_len, extra_len = values[3], values[9], values[10]
    if header[:4] != b"PK\x03\x04" or method != 8:
        raise ValueError(f"local ZIP header drift: {member['name']}")
    compressed_start = local_offset + 30 + name_len + extra_len
    prefix_compressed, _ = _fetch(zip_url, (compressed_start, compressed_start + 65535))
    output = zlib.decompressobj(-15).decompress(prefix_compressed, 32)
    if len(output) < 8:
        raise ValueError(f"could not decode payload signature: {member['name']}")
    return output


def _destination(root: Path, row: dict[str, Any], side: str) -> Path:
    return root / row["role"] / row["scene_id"] / f"{side}.png"


def _acquire_one(
    *,
    zip_url: str,
    root: Path,
    row: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    member = row[f"{side}_member"]
    destination = _destination(root, row, side)
    network_bytes = 0
    if destination.exists():
        data = destination.read_bytes()
        _validate_payload(data, member)
    else:
        data, network_bytes = _fetch_member(zip_url, member)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".png.part")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    return {
        "role": row["role"],
        "scene_id": row["scene_id"],
        "side": side,
        "member_name": member["name"],
        "member_crc32_hex": member["crc32_hex"],
        "relative_path": destination.relative_to(ROOT).as_posix(),
        "bytes": len(data),
        "sha256": _sha256(data),
        "network_bytes_this_execution": network_bytes,
    }


def run(
    contract_path: Path,
    manifest_path: Path,
    report_path: Path,
    workers: int,
) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    contract = json.loads(contract_bytes)
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "PASS_METADATA_ROLE_LOCK" or manifest["failed_gates"]:
        raise ValueError("SPCP2 role manifest is not admitted")
    if manifest["contract_sha256"] != _sha256(contract_bytes):
        raise ValueError("SPCP2 contract/manifest binding drift")
    expected_roles = contract["acquisition"]["initial_roles"]
    rows = [row for row in manifest["selected_rows"] if row["role"] in expected_roles]
    if len(rows) * 2 != contract["acquisition"]["initial_member_count_exact"]:
        raise ValueError("SPCP2 initial role count drift")
    if any(row["role"] == "sealed" for row in rows):
        raise ValueError("sealed role entered acquisition")

    output_root = ROOT / contract["acquisition"]["output_root"]
    tasks = sorted(
        ((row, side) for row in rows for side in ("loser", "winner")),
        key=lambda item: (item[0]["role"], item[0]["scene_id"], item[1]),
    )
    # The archive's filename extension is not trusted. Perform a deterministic
    # signature preflight before scheduling any additional payload writes.
    for row, side in tasks:
        member = row[f"{side}_member"]
        destination = _destination(output_root, row, side)
        prefix = (
            destination.read_bytes()[:32]
            if destination.exists()
            else _fetch_member_prefix(contract["source"]["zip_url"], member)
        )
        if not prefix.startswith(PNG_SIGNATURE):
            report = {
                "schema": "neuro-film.u5-r2spcp2-selected-pair-acquisition-report.v1",
                "experiment_id": contract["experiment_id"],
                "contract_sha256": _sha256(contract_bytes),
                "roles_manifest_sha256": _sha256(manifest_bytes),
                "roles_stable_manifest_id": manifest["stable_manifest_id"],
                "failed_member": {
                    "role": row["role"],
                    "scene_id": row["scene_id"],
                    "side": side,
                    "member_name": member["name"],
                    "declared_extension": ".png",
                    "observed_prefix_hex": prefix[:32].hex(),
                },
                "sealed_payload_count": 0,
                "operator_fit_count": 0,
                "calibration_score_count": 0,
                "failed_gates": ["required_payload_format_png"],
                "status": "FAIL_CLOSED_BEFORE_FIT_PAYLOAD_FORMAT_MISMATCH",
                "decision": "close_exact_spcp2_decode_and_operator_family",
                "claim_ceiling": contract["claim_ceiling"],
            }
            report["stable_acquisition_id"] = _stable_id(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_bytes(_canonical_bytes(report))
            return report
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _acquire_one,
                zip_url=contract["source"]["zip_url"],
                root=output_root,
                row=row,
                side=side,
            ): (row["scene_id"], side)
            for row, side in tasks
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: (item["role"], item["scene_id"], item["side"]))
    scientific_records = [
        {key: value for key, value in record.items() if key != "network_bytes_this_execution"}
        for record in records
    ]
    role_counts: dict[str, int] = {}
    for role in expected_roles:
        role_counts[role] = sum(record["role"] == role for record in scientific_records)
    sealed_root = output_root / "sealed"
    sealed_payload_count = (
        len([path for path in sealed_root.rglob("*.png") if path.is_file()])
        if sealed_root.exists()
        else 0
    )
    report = {
        "schema": "neuro-film.u5-r2spcp2-selected-pair-acquisition-report.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_bytes),
        "roles_manifest_sha256": _sha256(manifest_bytes),
        "roles_stable_manifest_id": manifest["stable_manifest_id"],
        "records": scientific_records,
        "member_count": len(scientific_records),
        "role_member_counts": role_counts,
        "payload_bytes": sum(record["bytes"] for record in scientific_records),
        "sealed_payload_count": sealed_payload_count,
        "all_png": all(record["relative_path"].endswith(".png") for record in scientific_records),
        "all_unique_paths": len({record["relative_path"] for record in scientific_records})
        == len(scientific_records),
        "status": "PASS_FIT_CAL_SELECTED_PAIR_ACQUISITION",
        "decision": "open_prefit_decode_and_operator_execution",
        "claim_ceiling": contract["claim_ceiling"],
    }
    if report["member_count"] != contract["acquisition"]["initial_member_count_exact"]:
        raise ValueError("acquired member count drift")
    if sealed_payload_count != 0:
        raise ValueError("sealed payload appeared before development pass")
    report["stable_acquisition_id"] = _stable_id(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(_canonical_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be in [1,16]")
    report = run(
        args.contract.resolve(),
        args.manifest.resolve(),
        args.report.resolve(),
        args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
