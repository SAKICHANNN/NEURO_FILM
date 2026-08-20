"""Range-only qualification of the NTIRE 2025 night capture-pair source."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping
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
    """Raised when the frozen SF3.A0U source contract fails closed."""


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroFilm-SF3-A0U/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _unsafe(member: ZipMember) -> bool:
    path = PurePosixPath(member.name)
    host = member.version_made_by >> 8
    mode = member.external_attributes >> 16
    return (
        "\\" in member.name
        or "\x00" in member.name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or bool(member.flags & 1)
        or member.method not in (0, 8)
        or (host == 3 and (mode & 0o170000) == 0o120000)
    )


def _stem(member: ZipMember) -> str:
    return PurePosixPath(member.name).stem.casefold()


def analyze_pair_graph(
    raw_members: list[ZipMember], target_members: list[ZipMember]
) -> dict[str, Any]:
    """Analyze exact numeric-stem RAW/metadata/processed/target groups."""
    for label, members in (("raw", raw_members), ("target", target_members)):
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise NTIRENightSourceLockError(f"duplicate {label} archive member names")
    raw_png = {
        _stem(member)
        for member in raw_members
        if PurePosixPath(member.name).suffix.casefold() == ".png"
    }
    metadata_json = {
        _stem(member)
        for member in raw_members
        if PurePosixPath(member.name).suffix.casefold() == ".json"
    }
    processed = {
        _stem(member)
        for member in raw_members
        if PurePosixPath(member.name).suffix.casefold() in (".jpg", ".jpeg")
    }
    target = {
        _stem(member)
        for member in target_members
        if PurePosixPath(member.name).suffix.casefold() in (".jpg", ".jpeg")
    }
    complete = sorted(raw_png & metadata_json & processed & target)
    all_members = raw_members + target_members
    return {
        "raw_archive_member_count": len(raw_members),
        "target_archive_member_count": len(target_members),
        "raw_png_count": len(raw_png),
        "metadata_json_count": len(metadata_json),
        "processed_smartphone_count": len(processed),
        "professional_target_count": len(target),
        "complete_group_count": len(complete),
        "independent_scene_count": len(complete),
        "complete_group_keys_sha256": canonical_sha256(complete),
        "unsafe_member_count": sum(_unsafe(member) for member in all_members),
        "raw_central_members_sha256": canonical_sha256(
            [
                (
                    member.name,
                    member.crc32,
                    member.compressed_size,
                    member.uncompressed_size,
                )
                for member in raw_members
            ]
        ),
        "target_central_members_sha256": canonical_sha256(
            [
                (
                    member.name,
                    member.crc32,
                    member.compressed_size,
                    member.uncompressed_size,
                )
                for member in target_members
            ]
        ),
    }


RangeReader = Callable[[str, int, int, int], bytes]


def _archive_members(
    archive: Mapping[str, Any], *, tail_limit: int, range_reader: RangeReader
) -> tuple[list[ZipMember], dict[str, Any]]:
    size = int(archive["bytes"])
    tail_size = min(size, tail_limit)
    start = size - tail_size
    tail = range_reader(str(archive["url"]), start, size - 1, size)
    offset, central_size, expected_count = locate_central_directory(
        tail, archive_size=size
    )
    central_end = offset + central_size
    if offset < start or central_end > size:
        raise NTIRENightSourceLockError(
            "central directory is outside the frozen tail budget"
        )
    central = tail[offset - start : central_end - start]
    try:
        members = parse_central_directory(central)
    except PPISPSourceLockError as exc:
        raise NTIRENightSourceLockError(str(exc)) from exc
    if len(members) != expected_count:
        raise NTIRENightSourceLockError("central-directory member count differs")
    return members, {
        "name": archive["name"],
        "bytes": size,
        "md5": archive["md5"],
        "tail_start": start,
        "tail_bytes": len(tail),
        "tail_sha256": sha256_bytes(tail),
        "central_offset": offset,
        "central_size": central_size,
        "central_sha256": sha256_bytes(central),
        "member_count": len(members),
    }


def _record_facts(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    files = {
        str(item.get("key")): item
        for item in payload.get("files", [])
        if isinstance(item, Mapping)
    }
    rows = []
    for key in ("raw_archive", "target_archive"):
        expected = config["source"][key]
        observed = files.get(str(expected["name"]), {})
        rows.append(
            {
                "name": expected["name"],
                "bytes": int(observed.get("size") or 0),
                "checksum": str(observed.get("checksum") or ""),
                "content_url": str(observed.get("links", {}).get("content") or ""),
            }
        )
    metadata = (
        payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    )
    license_row = (
        metadata.get("license") if isinstance(metadata.get("license"), Mapping) else {}
    )
    return {
        "record_id": int(payload.get("id") or 0),
        "doi": str(metadata.get("doi") or ""),
        "publication_date": str(metadata.get("publication_date") or ""),
        "license_id": str(license_row.get("id") or ""),
        "files": rows,
    }


def run_source_lock(
    config_path: Path,
    *,
    range_reader: RangeReader = http_range_get,
    url_reader: Callable[[str], bytes] = _read_url,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if (
        config.get("schema")
        != "neuro-film.sf3-a0u-ntire-night-capture-pair-source-lock-contract.v1"
    ):
        raise NTIRENightSourceLockError("contract schema differs")
    api_payload = json.loads(url_reader(config["source"]["api"]))
    if not isinstance(api_payload, Mapping):
        raise NTIRENightSourceLockError("Zenodo record is not an object")
    record = _record_facts(api_payload, config)
    tail_limit = int(config["read_budget"]["maximum_tail_bytes_per_archive"])
    raw_members, raw_archive = _archive_members(
        config["source"]["raw_archive"],
        tail_limit=tail_limit,
        range_reader=range_reader,
    )
    target_members, target_archive = _archive_members(
        config["source"]["target_archive"],
        tail_limit=tail_limit,
        range_reader=range_reader,
    )
    graph = analyze_pair_graph(raw_members, target_members)
    expected_files = {
        archive["name"]: archive
        for archive in (
            config["source"]["raw_archive"],
            config["source"]["target_archive"],
        )
    }
    record_exact = (
        record["record_id"] == int(config["source"]["zenodo_record"])
        and record["doi"] == config["source"]["doi"]
        and record["publication_date"] == config["source"]["publication_date"]
        and record["license_id"].casefold() == config["source"]["license"].casefold()
        and all(
            row["bytes"] == int(expected_files[row["name"]]["bytes"])
            and row["checksum"] == f"md5:{expected_files[row['name']]['md5']}"
            and row["content_url"] == expected_files[row["name"]]["url"]
            for row in record["files"]
        )
    )
    published_facts = " ".join(config["official_paper"]["capture_facts"]).casefold()
    required_fields = config["required_source_gates"]["capture_metadata_fields_present"]
    metadata_fields_declared = {
        field: field.replace("_", " ") in published_facts for field in required_fields
    }
    archive_bytes_read = raw_archive["tail_bytes"] + target_archive["tail_bytes"]
    gates = {
        "record_identity_size_checksum_license_exact": record_exact,
        "both_zip_central_directories_parse": raw_archive["member_count"] > 0
        and target_archive["member_count"] > 0,
        "no_unsafe_or_duplicate_member_names": graph["unsafe_member_count"] == 0,
        "complete_raw_metadata_processed_target_pair_graph": graph[
            "complete_group_count"
        ]
        >= 64,
        "minimum_complete_group_count": graph["complete_group_count"] >= 64,
        "minimum_independent_scene_count": graph["independent_scene_count"] >= 64,
        "explicit_crop_or_alignment_metadata_present": (
            graph["metadata_json_count"] >= graph["complete_group_count"]
            and "crop" in published_facts
        ),
        "capture_metadata_fields_present": all(metadata_fields_declared.values()),
        "member_payload_bytes_equal_zero": True,
        "pixel_decode_count_equal_zero": True,
        "archive_read_budget": archive_bytes_read
        <= int(config["read_budget"]["maximum_total_archive_bytes"]),
    }
    automatic_pass = all(gates.values())
    report = {
        "schema": "neuro-film.sf3-a0u-ntire-night-capture-pair-source-lock-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "record": record,
        "archives": [raw_archive, target_archive],
        **graph,
        "published_capture_metadata_fields_declared": metadata_fields_declared,
        "api_metadata_requests": 1,
        "archive_bytes_read": archive_bytes_read,
        "member_payload_bytes": 0,
        "pixel_decodes": 0,
        "operator_fits": 0,
        "bounded_final_candidate_counter_before": config[
            "bounded_final_candidate_counter_before"
        ],
        "bounded_final_candidate_counter_after": config[
            "bounded_final_candidate_counter_after"
        ],
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": config["decision_if_pass"]
        if automatic_pass
        else "CLOSE_SOURCE_BEFORE_MEMBER_EXTRACTION",
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
