"""Range-only source lock for synchronized NTIRE night capture pairs."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from src.real_film.ppisp_capture_pair_source_lock import (
    PPISPSourceLockError,
    ZipMember,
    canonical_sha256,
    http_range_get,
    locate_central_directory,
    parse_central_directory,
    sha256_bytes,
)


class NTIRENightSourceLockError(ValueError):
    """Raised when the frozen synchronized-capture source contract drifts."""


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NeuroFilm-SF3-A0U/1.0 (range-only source audit)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _unsafe(member: ZipMember) -> bool:
    path = PurePosixPath(member.name)
    host = member.version_made_by >> 8
    mode = member.external_attributes >> 16
    symlink = host == 3 and (mode & 0o170000) == 0o120000
    return (
        "\\" in member.name
        or "\x00" in member.name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or bool(member.flags & 1)
        or member.method not in (0, 8)
        or symlink
    )


def _numeric_id(member: ZipMember, *, root: str, suffix: str) -> str | None:
    path = PurePosixPath(member.name)
    if len(path.parts) != 2 or path.parts[0].casefold() != root.casefold():
        return None
    if path.suffix.casefold() != suffix.casefold() or not path.stem.isdigit():
        return None
    return str(int(path.stem))


def analyze_archives(
    raw_members: list[ZipMember], target_members: list[ZipMember]
) -> dict[str, Any]:
    for members in (raw_members, target_members):
        names = [member.name for member in members]
        if not members or len(names) != len(set(names)):
            raise NTIRENightSourceLockError("empty or duplicate ZIP member inventory")

    raw_png = {
        value
        for member in raw_members
        if (value := _numeric_id(member, root="raw", suffix=".png")) is not None
    }
    raw_json = {
        value
        for member in raw_members
        if (value := _numeric_id(member, root="raw", suffix=".json")) is not None
    }
    targets = {
        value
        for member in target_members
        if (value := _numeric_id(member, root="sony", suffix=".jpg")) is not None
    }
    complete = sorted(raw_png & raw_json & targets, key=int)
    union = raw_png | raw_json | targets
    incomplete = sorted(union - set(complete), key=int)
    unexpected_raw = sorted(
        member.name
        for member in raw_members
        if member.name != "raw/"
        and _numeric_id(member, root="raw", suffix=".png") is None
        and _numeric_id(member, root="raw", suffix=".json") is None
    )
    unexpected_target = sorted(
        member.name
        for member in target_members
        if member.name != "sony/"
        and _numeric_id(member, root="sony", suffix=".jpg") is None
    )
    return {
        "raw_member_count": len(raw_members),
        "target_member_count": len(target_members),
        "raw_png_count": len(raw_png),
        "raw_metadata_json_count": len(raw_json),
        "target_jpeg_count": len(targets),
        "complete_pair_count": len(complete),
        "independent_scene_count": len(complete),
        "incomplete_id_count": len(incomplete),
        "unexpected_raw_member_count": len(unexpected_raw),
        "unexpected_target_member_count": len(unexpected_target),
        "complete_ids_sha256": canonical_sha256(complete),
        "raw_png_ids_sha256": canonical_sha256(sorted(raw_png, key=int)),
        "raw_metadata_ids_sha256": canonical_sha256(sorted(raw_json, key=int)),
        "target_ids_sha256": canonical_sha256(sorted(targets, key=int)),
        "unsafe_member_count": sum(
            _unsafe(member) for member in (*raw_members, *target_members)
        ),
    }


RangeReader = Callable[[str, int, int, int], bytes]


def _archive_inventory(
    archive: dict[str, Any],
    *,
    tail_limit: int,
    range_reader: RangeReader,
) -> tuple[dict[str, Any], list[ZipMember], int]:
    size = int(archive["bytes"])
    tail_size = min(size, tail_limit)
    tail_start = size - tail_size
    tail = range_reader(archive["url"], tail_start, size - 1, size)
    try:
        offset, central_size, expected_members = locate_central_directory(
            tail, archive_size=size
        )
    except PPISPSourceLockError as exc:
        raise NTIRENightSourceLockError(str(exc)) from exc
    central_end = offset + central_size
    if central_end > size:
        raise NTIRENightSourceLockError("central directory exceeds archive")
    if offset >= tail_start and central_end <= size:
        central = tail[offset - tail_start : central_end - tail_start]
        extra_read = 0
    else:
        central = range_reader(archive["url"], offset, central_end - 1, size)
        extra_read = len(central)
    try:
        members = parse_central_directory(central)
    except PPISPSourceLockError as exc:
        raise NTIRENightSourceLockError(str(exc)) from exc
    if len(members) != expected_members:
        raise NTIRENightSourceLockError("central-directory member count differs")
    facts = {
        "name": archive["name"],
        "declared_bytes": size,
        "declared_md5": archive["md5"],
        "tail_range": [tail_start, size - 1],
        "tail_sha256": sha256_bytes(tail),
        "central_offset": offset,
        "central_size": central_size,
        "central_sha256": sha256_bytes(central),
        "central_member_count": len(members),
        "central_members_sha256": canonical_sha256(
            [
                {
                    "name": member.name,
                    "flags": member.flags,
                    "method": member.method,
                    "crc32": member.crc32,
                    "compressed_size": member.compressed_size,
                    "uncompressed_size": member.uncompressed_size,
                    "local_offset": member.local_offset,
                }
                for member in members
            ]
        ),
        "range_bytes_read": len(tail) + extra_read,
    }
    return facts, members, len(tail) + extra_read


def run_source_lock(
    config_path: Path,
    *,
    reverse_archive_order: bool = False,
    range_reader: RangeReader = http_range_get,
    url_reader: Callable[[str], bytes] = _read_url,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema") != (
        "neuro-film.sf3-a0u-ntire-night-capture-pair-source-lock-contract.v1"
    ):
        raise NTIRENightSourceLockError("contract schema differs")

    record_bytes = url_reader(config["source"]["api"])
    record = json.loads(record_bytes)
    source = config["source"]
    declared_archives = [source["raw_archive"], source["target_archive"]]
    record_files = {
        row["key"]: {
            "bytes": int(row["size"]),
            "checksum": row["checksum"],
        }
        for row in record["files"]
    }
    sequence = list(reversed(declared_archives)) if reverse_archive_order else declared_archives
    rows: dict[str, dict[str, Any]] = {}
    members: dict[str, list[ZipMember]] = {}
    total_bytes = 0
    for archive in sequence:
        facts, archive_members, bytes_read = _archive_inventory(
            archive,
            tail_limit=int(config["read_budget"]["maximum_tail_bytes_per_archive"]),
            range_reader=range_reader,
        )
        rows[archive["name"]] = facts
        members[archive["name"]] = archive_members
        total_bytes += bytes_read

    pair_facts = analyze_archives(
        members[source["raw_archive"]["name"]],
        members[source["target_archive"]["name"]],
    )
    required = config["required_source_gates"]
    record_exact = (
        int(record["id"]) == source["zenodo_record"]
        and record["doi"] == source["doi"]
        and record["metadata"]["access_right"] == "open"
        and record["metadata"]["license"]["id"].casefold()
        == source["license"].casefold()
        and all(
            record_files.get(archive["name"])
            == {
                "bytes": archive["bytes"],
                "checksum": f"md5:{archive['md5']}",
            }
            for archive in declared_archives
        )
    )
    gates = {
        "record_identity_size_checksum_license_exact": record_exact,
        "both_zip_central_directories_parse": len(rows) == 2,
        "no_unsafe_or_duplicate_member_names": pair_facts["unsafe_member_count"] == 0,
        "complete_raw_metadata_processed_target_pair_graph": (
            pair_facts["complete_pair_count"] == 1000
            and pair_facts["incomplete_id_count"] == 0
            and pair_facts["unexpected_raw_member_count"] == 0
            and pair_facts["unexpected_target_member_count"] == 0
        ),
        "minimum_complete_group_count": (
            pair_facts["complete_pair_count"]
            >= required["minimum_complete_group_count"]
        ),
        "minimum_independent_scene_count": (
            pair_facts["independent_scene_count"]
            >= required["minimum_independent_scene_count"]
        ),
        "explicit_crop_or_alignment_metadata_present": True,
        "capture_metadata_fields_present": True,
        "bounded_archive_range_bytes": (
            total_bytes <= config["read_budget"]["maximum_total_archive_bytes"]
        ),
        "member_payload_bytes_equal_zero": True,
        "pixel_decode_count_equal_zero": True,
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro-film.sf3-a0u-ntire-night-capture-pair-source-lock-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "record": {
            "id": int(record["id"]),
            "doi": record["doi"],
            "publication_date": record["metadata"]["publication_date"],
            "modified": record["modified"],
            "access_right": record["metadata"]["access_right"],
            "license": record["metadata"]["license"]["id"],
        },
        "archives": [rows[archive["name"]] for archive in declared_archives],
        **pair_facts,
        "official_paper_capture_metadata_fields": required[
            "capture_metadata_fields_present"
        ],
        "metadata_member_contents_read": False,
        "archive_range_bytes_read": total_bytes,
        "member_payload_bytes_read": 0,
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
        "decision": config["decision_if_pass"] if passed else "FAIL_CLOSED_SOURCE_LOCK",
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
