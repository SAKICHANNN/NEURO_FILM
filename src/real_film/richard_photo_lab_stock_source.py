"""Range-only source lock for Richard Photo Lab stock exposure ladders."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    http_range_get,
    locate_central_directory,
    parse_central_directory,
)


class RichardPhotoLabSourceError(ValueError):
    """Raised when the frozen source does not satisfy its exact contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NeuroFilm-SF3-A3G/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _unsafe(member: ZipMember) -> bool:
    path = PurePosixPath(member.name)
    host = member.version_made_by >> 8
    mode = member.external_attributes >> 16
    symlink = host == 3 and (mode & 0o170000) == 0o120000
    unsafe_name = (
        "\\" in member.name
        or "\x00" in member.name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    )
    return (
        unsafe_name or symlink or bool(member.flags & 1) or member.method not in (0, 8)
    )


def _inventory(members: list[ZipMember]) -> list[dict[str, Any]]:
    return [
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


def _stock_row(members: list[ZipMember], spec: dict[str, Any]) -> dict[str, Any]:
    root = (
        "Richards-Film-Stock-And-Exposure-Comparisons/"
        + spec["archive_directory"]
        + "/"
    )
    images = [
        member
        for member in members
        if member.name.startswith(root)
        and PurePosixPath(member.name).suffix.casefold() in (".jpg", ".jpeg")
    ]
    relative = [member.name[len(root) :] for member in images]
    full_resolution = [name for name in relative if "/" not in name]
    negative_sequence = [name for name in relative if "/" in name]
    required = [
        spec["required_full_resolution_normal_name"],
        spec["required_negative_normal_name"],
    ]
    return {
        "film_stock_id": spec["film_stock_id"],
        "archive_directory": spec["archive_directory"],
        "full_resolution_scan_count": len(full_resolution),
        "negative_sequence_count": len(negative_sequence),
        "required_normal_members_present": {
            name: name in relative for name in required
        },
        "member_names_sha256": _canonical_sha256(sorted(relative)),
        "ladder_pass": (
            len(full_resolution) >= int(spec["minimum_full_resolution_scan_count"])
            and len(negative_sequence) >= int(spec["minimum_negative_sequence_count"])
            and all(name in relative for name in required)
        ),
    }


RangeReader = Callable[[str, int, int, int], bytes]
URLReader = Callable[[str], bytes]


def run_source_lock(
    config_path: Path,
    *,
    range_reader: RangeReader = http_range_get,
    url_reader: URLReader = _read_url,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    expected_schema = (
        "neuro-film.sf3-a3g-richard-photo-lab-stock-exposure-source-lock-contract.v1"
    )
    if config.get("schema") != expected_schema:
        raise RichardPhotoLabSourceError("contract schema differs")

    source = config["source"]
    page = url_reader(source["page_url"])
    page_text = page.decode("utf-8")
    phrase_presence = {
        phrase: phrase.casefold() in page_text.casefold()
        for phrase in source["required_page_phrases"]
    }

    archive_size = int(source["archive_size"])
    tail_size = min(int(config["network_limits"]["tail_probe_bytes"]), archive_size)
    tail_start = archive_size - tail_size
    tail = range_reader(
        source["archive_url"], tail_start, archive_size - 1, archive_size
    )
    central_offset, central_size, expected_members = locate_central_directory(
        tail, archive_size=archive_size
    )
    central_end = central_offset + central_size
    if central_end > archive_size:
        raise RichardPhotoLabSourceError("central directory exceeds archive")
    if central_offset >= tail_start and central_end <= archive_size:
        central = tail[central_offset - tail_start : central_end - tail_start]
        central_extra_bytes = 0
    else:
        central = range_reader(
            source["archive_url"], central_offset, central_end - 1, archive_size
        )
        central_extra_bytes = len(central)
    members = parse_central_directory(central)
    if len(members) != expected_members:
        raise RichardPhotoLabSourceError("central-directory member count differs")
    inventory = _inventory(members)
    stock_rows = [_stock_row(members, spec) for spec in config["required_stocks"]]
    jpeg_count = sum(
        PurePosixPath(member.name).suffix.casefold() in (".jpg", ".jpeg")
        for member in members
    )
    network_bytes = len(page) + len(tail) + central_extra_bytes
    unsafe_count = sum(_unsafe(member) for member in members)

    gates = {
        "page_identity_exact": (
            len(page) == int(source["page_size"])
            and _sha256(page) == source["page_sha256"]
        ),
        "all_page_phrases_present": all(phrase_presence.values()),
        "central_identity_exact": (
            central_offset == int(source["central_offset"])
            and central_size == int(source["central_size"])
            and len(members) == int(source["central_member_count"])
            and _sha256(tail) == source["tail_sha256"]
            and _sha256(central) == source["central_sha256"]
            and _canonical_sha256(inventory) == source["central_inventory_sha256"]
            and jpeg_count == int(source["jpeg_member_count"])
        ),
        "safe_unencrypted_members": unsafe_count == 0,
        "all_stock_ladders_present": all(row["ladder_pass"] for row in stock_rows),
        "network_budget": network_bytes
        <= int(config["network_limits"]["maximum_total_bytes"]),
        "zero_member_payload_reads": int(
            config["network_limits"]["member_payload_reads"]
        )
        == 0,
        "zero_image_member_reads": int(config["network_limits"]["image_member_reads"])
        == 0,
        "zero_pixel_decodes": int(config["network_limits"]["pixel_decodes"]) == 0,
    }
    passed = all(gates.values())
    scientific = {
        "experiment_id": config["experiment_id"],
        "source_page_size": len(page),
        "source_page_sha256": _sha256(page),
        "required_page_phrases_present": phrase_presence,
        "archive_size": archive_size,
        "tail_range": [tail_start, archive_size - 1],
        "tail_sha256": _sha256(tail),
        "central_offset": central_offset,
        "central_size": central_size,
        "central_sha256": _sha256(central),
        "central_member_count": len(members),
        "central_inventory_sha256": _canonical_sha256(inventory),
        "jpeg_member_count": jpeg_count,
        "unsafe_member_count": unsafe_count,
        "stock_rows": stock_rows,
        "network_bytes_read": network_bytes,
        "member_payload_reads": 0,
        "image_member_reads": 0,
        "pixel_decodes": 0,
        "gates": gates,
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        "schema": "neuro-film.sf3-a3g-richard-photo-lab-stock-exposure-source-lock-result.v1",
        **scientific,
        "contract_sha256": _sha256(config_bytes),
        "stable_evidence_id": _canonical_sha256(scientific),
    }
