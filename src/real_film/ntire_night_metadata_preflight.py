"""JSON-only metadata preflight for synchronized NTIRE night captures."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    canonical_sha256,
    http_range_get,
    parse_central_directory,
    sha256_bytes,
)


class NTIRENightMetadataPreflightError(ValueError):
    """Raised when the frozen metadata-only contract is invalid."""


RangeReader = Callable[[str, int, int, int], bytes]


def derive_selection(seed: str, minimum: int, maximum: int, count: int) -> list[int]:
    ranked = sorted(
        range(minimum, maximum + 1),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    return ranked[:count]


def extract_member(
    url: str,
    member: ZipMember,
    archive_size: int,
    range_reader: RangeReader,
) -> tuple[bytes, int]:
    header = range_reader(
        url, member.local_offset, member.local_offset + 29, archive_size
    )
    if len(header) != 30:
        raise NTIRENightMetadataPreflightError("local header length differs")
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04":
        raise NTIRENightMetadataPreflightError("invalid local-header signature")
    name_length, extra_length = values[-2:]
    body_start = member.local_offset + 30 + name_length + extra_length
    compressed = range_reader(
        url,
        body_start,
        body_start + member.compressed_size - 1,
        archive_size,
    )
    if member.method == 0:
        payload = compressed
    elif member.method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise NTIRENightMetadataPreflightError("unsupported compression method")
    if len(payload) != member.uncompressed_size:
        raise NTIRENightMetadataPreflightError("uncompressed size differs")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != member.crc32:
        raise NTIRENightMetadataPreflightError("member CRC differs")
    return payload, len(header) + len(compressed)


def _flatten(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key], (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten(item, (*path, str(index)))
    else:
        yield ".".join(path), value


def _normalized(path: str) -> str:
    return re.sub(r"[^a-z0-9]", "", path.casefold())


def analyze_metadata(payload: bytes, config: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NTIRENightMetadataPreflightError("invalid metadata JSON") from exc
    if not isinstance(value, dict):
        raise NTIRENightMetadataPreflightError("metadata root is not an object")
    leaves = list(_flatten(value))
    numeric = [
        (path, float(item))
        for path, item in leaves
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]
    finite = all(math.isfinite(item) for _, item in numeric)
    matched: dict[str, list[str]] = {}
    for family, aliases in config["field_families"].items():
        matched[family] = sorted(
            {
                path
                for path, _ in leaves
                if any(alias in _normalized(path) for alias in aliases)
            }
        )
    crop_aliases = config["field_families"]["crop_bounds"]
    crop_values = [
        item
        for path, item in numeric
        if any(alias in _normalized(path) for alias in crop_aliases)
    ]
    crop_contract = config["crop_numeric_contract"]
    crop_bounded = (
        len(crop_values) >= crop_contract["minimum_numeric_values_per_row"]
        and all(
            abs(item) <= crop_contract["maximum_absolute_value"]
            for item in crop_values
        )
    )
    return {
        "payload_sha256": sha256_bytes(payload),
        "payload_bytes": len(payload),
        "root_keys": sorted(str(key) for key in value),
        "leaf_count": len(leaves),
        "numeric_leaf_count": len(numeric),
        "finite_numeric_values": finite,
        "matched_paths": matched,
        "all_field_families_present": all(matched.values()),
        "crop_numeric_value_count": len(crop_values),
        "crop_numeric_minimum": min(crop_values) if crop_values else None,
        "crop_numeric_maximum": max(crop_values) if crop_values else None,
        "crop_values_bounded": crop_bounded,
    }


def run_preflight(
    config_path: Path,
    *,
    reverse_member_order: bool = False,
    range_reader: RangeReader = http_range_get,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema") != (
        "neuro-film.sf3-a0v-ntire-night-metadata-preflight-contract.v1"
    ):
        raise NTIRENightMetadataPreflightError("contract schema differs")

    archive = config["archive"]
    central = range_reader(
        archive["url"],
        archive["central_offset"],
        archive["central_offset"] + archive["central_size"] - 1,
        archive["bytes"],
    )
    central_exact = sha256_bytes(central) == archive["central_sha256"]
    members = parse_central_directory(central)
    by_name = {member.name: member for member in members}
    selection = config["selection"]
    population = selection["population_ids"]
    derived = derive_selection(
        selection["seed"],
        population["minimum"],
        population["maximum"],
        selection["selected_count"],
    )
    selected = selection["selected_ids"]
    sequence = list(reversed(selected)) if reverse_member_order else selected
    rows_by_id: dict[int, dict[str, Any]] = {}
    member_archive_bytes = 0
    member_uncompressed_bytes = 0
    missing: list[str] = []
    for numeric_id in sequence:
        name = f"raw/{numeric_id}.json"
        member = by_name.get(name)
        if member is None:
            missing.append(name)
            continue
        if member.uncompressed_size > config["read_budget"][
            "maximum_member_uncompressed_bytes"
        ]:
            raise NTIRENightMetadataPreflightError("metadata member exceeds byte cap")
        payload, bytes_read = extract_member(
            archive["url"], member, archive["bytes"], range_reader
        )
        row = analyze_metadata(payload, config)
        row["numeric_id"] = numeric_id
        row["member_name"] = name
        row["member_crc32"] = member.crc32
        row["member_compressed_bytes"] = member.compressed_size
        rows_by_id[numeric_id] = row
        member_archive_bytes += bytes_read
        member_uncompressed_bytes += len(payload)

    rows = [rows_by_id[numeric_id] for numeric_id in selected if numeric_id in rows_by_id]
    parent = config["parent_source_lock"]
    parent_exact = (
        parent["report_sha256"]
        == "ec88e11fe4ce18205f47b20ded8a7b44ead5a40919fe6d0a1bf8f7bedb69655c"
        and parent["stable_identity"]
        == "b49fc4a8708b0b3f2b1887d4c2ff0ae22ae155dfcd405a6fbe58060b252d8093"
    )
    total_archive_bytes = len(central) + member_archive_bytes
    budget = config["read_budget"]
    gates = {
        "parent_identity_exact": parent_exact,
        "central_directory_exact": central_exact,
        "selection_derivation_exact": derived == selected,
        "selected_metadata_members_complete": not missing and len(rows) == len(selected),
        "json_object_and_finite_numeric_values": bool(rows)
        and all(row["finite_numeric_values"] for row in rows),
        "all_field_families_present_per_row": bool(rows)
        and all(row["all_field_families_present"] for row in rows),
        "crop_values_numeric_and_bounded": bool(rows)
        and all(row["crop_values_bounded"] for row in rows),
        "minimum_rows": len(rows) >= config["required_gates"]["minimum_rows"],
        "bounded_member_uncompressed_bytes": (
            member_uncompressed_bytes
            <= budget["maximum_total_member_uncompressed_bytes"]
        ),
        "bounded_total_archive_bytes": (
            total_archive_bytes <= budget["maximum_total_archive_bytes"]
        ),
        "zero_pixel_member_reads": True,
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro-film.sf3-a0v-ntire-night-metadata-preflight-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "parent_report_sha256": parent["report_sha256"],
        "parent_stable_identity": parent["stable_identity"],
        "archive_central_sha256": sha256_bytes(central),
        "selection_method": selection["method"],
        "selection_seed": selection["seed"],
        "selected_ids": selected,
        "selected_ids_sha256": canonical_sha256(selected),
        "rows": rows,
        "row_count": len(rows),
        "missing_members": missing,
        "central_archive_bytes_read": len(central),
        "metadata_archive_bytes_read": member_archive_bytes,
        "metadata_uncompressed_bytes": member_uncompressed_bytes,
        "total_archive_bytes_read": total_archive_bytes,
        "raw_png_member_bytes_read": 0,
        "sony_jpeg_member_bytes_read": 0,
        "pixel_decodes": 0,
        "operator_fits": 0,
        "renders": 0,
        "scores": 0,
        "bounded_final_candidate_counter_before": config[
            "bounded_final_candidate_counter_before"
        ],
        "bounded_final_candidate_counter_after": config[
            "bounded_final_candidate_counter_after"
        ],
        "gates": gates,
        "automatic_pass": passed,
        "decision": (
            config["decision_if_pass"]
            if passed
            else "FAIL_CLOSED_METADATA_PREFLIGHT"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
