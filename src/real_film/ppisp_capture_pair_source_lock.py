"""Range-only structural qualification for the official PPISP capture pairs."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class PPISPSourceLockError(ValueError):
    """Raised when the frozen PPISP source contract is structurally invalid."""


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


@dataclass(frozen=True)
class ZipMember:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int
    external_attributes: int
    version_made_by: int


def _zip64_values(extra: bytes) -> list[int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        value = extra[cursor : cursor + size]
        cursor += size
        if kind == 0x0001:
            if len(value) % 8:
                raise PPISPSourceLockError("invalid ZIP64 extra field")
            return list(struct.unpack(f"<{len(value) // 8}Q", value))
    return []


def parse_central_directory(payload: bytes) -> list[ZipMember]:
    members: list[ZipMember] = []
    cursor = 0
    while cursor < len(payload):
        if payload[cursor : cursor + 4] != b"PK\x01\x02":
            raise PPISPSourceLockError(
                f"invalid central-directory signature at {cursor}"
            )
        if cursor + 46 > len(payload):
            raise PPISPSourceLockError("truncated central-directory entry")
        values = struct.unpack_from("<4s6H3I5H2I", payload, cursor)
        (
            _,
            version_made_by,
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
            external_attributes,
            local_offset,
        ) = values
        body = cursor + 46
        name_bytes = payload[body : body + name_length]
        extra = payload[body + name_length : body + name_length + extra_length]
        end = body + name_length + extra_length + comment_length
        if end > len(payload):
            raise PPISPSourceLockError("truncated central-directory body")
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        zip64 = iter(_zip64_values(extra))
        try:
            if uncompressed_size == 0xFFFFFFFF:
                uncompressed_size = next(zip64)
            if compressed_size == 0xFFFFFFFF:
                compressed_size = next(zip64)
            if local_offset == 0xFFFFFFFF:
                local_offset = next(zip64)
        except StopIteration as exc:
            raise PPISPSourceLockError("incomplete ZIP64 extra field") from exc
        members.append(
            ZipMember(
                name=name,
                flags=flags,
                method=method,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_offset=local_offset,
                external_attributes=external_attributes,
                version_made_by=version_made_by,
            )
        )
        cursor = end
    return members


def locate_central_directory(tail: bytes, *, archive_size: int) -> tuple[int, int, int]:
    tail_start = archive_size - len(tail)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0 or eocd_offset + 22 > len(tail):
        raise PPISPSourceLockError("ZIP EOCD not found in frozen tail probe")
    values = struct.unpack_from("<4s4H2IH", tail, eocd_offset)
    _, disk, central_disk, disk_entries, total_entries, central_size, central_offset, _ = values
    if disk != 0 or central_disk != 0 or disk_entries != total_entries:
        raise PPISPSourceLockError("multi-disk ZIP is forbidden")
    if central_size != 0xFFFFFFFF and central_offset != 0xFFFFFFFF:
        return central_offset, central_size, total_entries

    locator_offset = tail.rfind(b"PK\x06\x07", 0, eocd_offset)
    if locator_offset < 0 or locator_offset + 20 > len(tail):
        raise PPISPSourceLockError("ZIP64 locator missing")
    _, zip64_disk, zip64_offset, zip64_disks = struct.unpack_from(
        "<4sIQI", tail, locator_offset
    )
    if zip64_disk != 0 or zip64_disks != 1:
        raise PPISPSourceLockError("multi-disk ZIP64 is forbidden")
    relative = zip64_offset - tail_start
    if relative < 0 or relative + 56 > len(tail):
        raise PPISPSourceLockError("ZIP64 EOCD outside frozen tail probe")
    values64 = struct.unpack_from("<4sQ2H2I4Q", tail, relative)
    if values64[0] != b"PK\x06\x06" or values64[4] != 0 or values64[5] != 0:
        raise PPISPSourceLockError("invalid ZIP64 EOCD")
    if values64[6] != values64[7]:
        raise PPISPSourceLockError("ZIP64 entry counts differ")
    return int(values64[9]), int(values64[8]), int(values64[7])


def _is_unsafe_name(name: str) -> bool:
    if "\\" in name or "\x00" in name:
        return True
    path = PurePosixPath(name)
    return path.is_absolute() or any(part in ("", ".", "..") for part in path.parts)


def _is_symlink(member: ZipMember) -> bool:
    host = member.version_made_by >> 8
    unix_mode = member.external_attributes >> 16
    return host == 3 and (unix_mode & 0o170000) == 0o120000


def _variant(name: str) -> str:
    lowered = name.lower()
    return "auto" if re.search(r"(?:^|[/_.-])auto(?:[/_.-]|$)", lowered) else "standard"


def _normalized_pair_key(name: str) -> str:
    parts = []
    for part in PurePosixPath(name).parts:
        cleaned = re.sub(
            r"(?i)(?:^|[_-])(?:auto|standard)(?:$|[_-])", "_variant_", part
        )
        cleaned = re.sub(r"__+", "_", cleaned).strip("_")
        parts.append(cleaned.lower())
    return "/".join(parts)


def analyze_members(members: list[ZipMember]) -> dict[str, Any]:
    if not members:
        raise PPISPSourceLockError("empty central directory")
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise PPISPSourceLockError("duplicate ZIP member names")
    unsafe = [member.name for member in members if _is_unsafe_name(member.name)]
    encrypted = [member.name for member in members if member.flags & 1]
    symlinks = [member.name for member in members if _is_symlink(member)]
    unsupported = [member.name for member in members if member.method not in (0, 8)]
    images = [
        member
        for member in members
        if PurePosixPath(member.name).suffix.lower() in (".jpg", ".jpeg")
    ]
    standard: dict[str, list[str]] = {}
    auto: dict[str, list[str]] = {}
    for member in images:
        key = _normalized_pair_key(member.name)
        target = auto if _variant(member.name) == "auto" else standard
        target.setdefault(key, []).append(member.name)
    pair_keys = sorted(set(standard) & set(auto))
    ambiguous_pairs = [
        key for key in pair_keys if len(standard[key]) != 1 or len(auto[key]) != 1
    ]
    exact_pairs = [key for key in pair_keys if key not in ambiguous_pairs]
    parent_groups = {str(PurePosixPath(key).parent) for key in exact_pairs}
    lower_names = [name.lower() for name in names]
    metadata = {
        "cameras": sorted(
            name
            for name in names
            if PurePosixPath(name).name.lower() in ("cameras.bin", "cameras.txt")
        ),
        "images": sorted(
            name
            for name in names
            if PurePosixPath(name).name.lower() in ("images.bin", "images.txt")
        ),
    }
    return {
        "member_count": len(members),
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
        "jpeg_member_count": len(images),
        "standard_jpeg_count": sum(len(value) for value in standard.values()),
        "auto_jpeg_count": sum(len(value) for value in auto.values()),
        "exact_pair_count": len(exact_pairs),
        "ambiguous_pair_count": len(ambiguous_pairs),
        "pair_parent_group_count": len(parent_groups),
        "pair_keys_sha256": canonical_sha256(exact_pairs),
        "standard_namespace_present": any(
            _variant(name) == "standard" for name in lower_names
        ),
        "auto_namespace_present": any(_variant(name) == "auto" for name in lower_names),
        "camera_metadata_member_count": len(metadata["cameras"]),
        "image_metadata_member_count": len(metadata["images"]),
        "metadata_member_names": metadata,
        "unsafe_member_count": len(unsafe),
        "encrypted_member_count": len(encrypted),
        "symlink_member_count": len(symlinks),
        "unsupported_method_member_count": len(unsupported),
        "member_payload_reads": 0,
        "image_member_reads": 0,
        "pixel_decodes": 0,
    }


RangeReader = Callable[[str, int, int, int], bytes]


def http_range_get(url: str, start: int, end: int, archive_size: int) -> bytes:
    if start < 0 or end < start or end >= archive_size:
        raise PPISPSourceLockError("invalid bounded archive range")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "NeuroFilm-SF3-A0P/1.0 (range-only source audit)",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        status = getattr(response, "status", None)
        expected_range = f"bytes {start}-{end}/{archive_size}"
        content_range = response.headers.get("Content-Range")
        expected_length = end - start + 1
        if status != 206 or content_range != expected_range:
            raise PPISPSourceLockError(
                f"server did not honor exact range: {status=} {content_range=}"
            )
        payload = response.read(expected_length + 1)
    if len(payload) != expected_length:
        raise PPISPSourceLockError("range payload length differs")
    return payload


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroFilm-SF3-A0P/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def run_source_lock(
    config_path: Path,
    *,
    range_reader: RangeReader = http_range_get,
    url_reader: Callable[[str], bytes] = _read_url,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema") != "neuro-film.sf3-a0p-ppisp-capture-pair-source-lock-contract.v1":
        raise PPISPSourceLockError("contract schema differs")
    source = config["source"]
    readme = url_reader(source["readme_url"])
    readme_text = readme.decode("utf-8")
    readme_facts = {
        "size": len(readme),
        "sha256": sha256_bytes(readme),
        "required_phrases_present": {
            phrase: phrase in readme_text for phrase in source["required_readme_phrases"]
        },
    }
    tail_limit = int(config["network_limits"]["tail_probe_bytes_per_archive"])
    revision = source["revision"]
    archive_rows = []
    total_archive_bytes_read = 0
    for archive in config["archives"]:
        size = int(archive["size"])
        url = (
            "https://huggingface.co/datasets/"
            f"{source['dataset_id']}/resolve/{revision}/{archive['path']}"
        )
        tail_size = min(tail_limit, size)
        tail_start = size - tail_size
        tail = range_reader(url, tail_start, size - 1, size)
        total_archive_bytes_read += len(tail)
        central_offset, central_size, expected_members = locate_central_directory(
            tail, archive_size=size
        )
        central_end = central_offset + central_size
        if central_end > size:
            raise PPISPSourceLockError("central directory exceeds archive")
        if central_offset >= tail_start and central_end <= size:
            central = tail[central_offset - tail_start : central_end - tail_start]
            central_extra_read = 0
        else:
            central = range_reader(
                url, central_offset, central_end - 1, size
            )
            central_extra_read = len(central)
            total_archive_bytes_read += central_extra_read
        members = parse_central_directory(central)
        if len(members) != expected_members:
            raise PPISPSourceLockError("central-directory member count differs")
        archive_rows.append(
            {
                "scene_id": archive["scene_id"],
                "path": archive["path"],
                "declared_archive_size": size,
                "declared_archive_sha256": archive["sha256"],
                "tail_range": [tail_start, size - 1],
                "tail_sha256": sha256_bytes(tail),
                "central_offset": central_offset,
                "central_size": central_size,
                "central_sha256": sha256_bytes(central),
                "central_extra_bytes_read": central_extra_read,
                **analyze_members(members),
            }
        )

    gate_config = config["gates"]
    gates = {
        "readme_identity_exact": (
            readme_facts["size"] == source["readme_size"]
            and readme_facts["sha256"] == source["readme_sha256"]
        ),
        "explicit_cc_by_4_commercial_scope": all(
            readme_facts["required_phrases_present"].values()
        ),
        "scene_count_exact": len(archive_rows) == gate_config["required_scene_count"],
        "standard_and_auto_namespaces_each_scene": all(
            row["standard_namespace_present"] and row["auto_namespace_present"]
            for row in archive_rows
        ),
        "paired_jpeg_support_each_scene": all(
            row["exact_pair_count"] >= gate_config["minimum_paired_jpegs_per_scene"]
            for row in archive_rows
        ),
        "pair_parent_group_support_each_scene": all(
            row["pair_parent_group_count"]
            >= gate_config["minimum_distinct_pair_parent_groups_per_scene"]
            for row in archive_rows
        ),
        "colmap_camera_and_image_metadata_each_scene": all(
            row["camera_metadata_member_count"] > 0
            and row["image_metadata_member_count"] > 0
            for row in archive_rows
        ),
        "safe_unencrypted_supported_members": all(
            row["unsafe_member_count"] == 0
            and row["encrypted_member_count"] == 0
            and row["symlink_member_count"] == 0
            and row["unsupported_method_member_count"] == 0
            for row in archive_rows
        ),
        "bounded_archive_metadata_bytes": (
            total_archive_bytes_read
            <= config["network_limits"]["maximum_total_archive_bytes"]
        ),
        "zero_member_payload_and_pixel_reads": all(
            row["member_payload_reads"] == row["image_member_reads"] == row["pixel_decodes"] == 0
            for row in archive_rows
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": "neuro-film.sf3-a0p-ppisp-capture-pair-source-lock-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "source": {
            "dataset_id": source["dataset_id"],
            "revision": revision,
            "readme": readme_facts,
        },
        "archives": archive_rows,
        "total_archive_bytes_read": total_archive_bytes_read,
        "image_member_reads": 0,
        "pixel_decodes": 0,
        "operator_fits": 0,
        "renders": 0,
        "scores": 0,
        "bounded_final_candidate_counter_before": 0,
        "bounded_final_candidate_counter_after": 0,
        "gates": gates,
        "automatic_pass": passed,
        "decision": (
            config["decision_if_pass"] if passed else config["decision_if_fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report
