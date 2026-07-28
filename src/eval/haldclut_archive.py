"""Bounded acquisition and integrity audit for the U5.R2AJ0B HaldCLUT archive."""

from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import stat
import struct
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from PIL import Image


class HaldArchiveError(RuntimeError):
    """Raised when the frozen archive contract cannot be executed safely."""


class _QuarantinePartial(RuntimeError):
    """Internal signal that a task-local partial must be isolated."""


EOCD_SIGNATURE = b"PK\x05\x06"
EOCD_STRUCT = struct.Struct("<4s4H2LH")
CONTENT_RANGE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
CONTRACT_EVIDENCE_SCHEMAS = {
    "u5-r2aj0b-haldclut-acquisition-v1": {
        "manifest": "u5-r2aj0b-haldclut-manifest-v1",
        "report": "u5-r2aj0b-haldclut-report-v1",
        "repeat_decision": "u5-r2aj0b-haldclut-repeat-decision-v1",
    },
    "u5-r2aj0b2-haldclut-acquisition-v2": {
        "manifest": "u5-r2aj0b2-haldclut-manifest-v2",
        "report": "u5-r2aj0b2-haldclut-report-v2",
        "repeat_decision": "u5-r2aj0b2-haldclut-repeat-decision-v2",
    },
}


def hash_file(path: Path) -> dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def sha256_file(path: Path) -> str:
    return hash_file(path)["sha256"]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise HaldArchiveError("evidence destination is not a regular file")
        if path.read_bytes() == payload:
            return
        raise HaldArchiveError(
            f"refusing to overwrite different evidence bytes: {path}"
        )
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise HaldArchiveError(f"temporary evidence path already exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config_snapshot(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    """Read and parse exactly one immutable config byte snapshot."""

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise HaldArchiveError("config byte snapshot SHA-256 mismatch")
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise HaldArchiveError("config snapshot is not valid JSON") from exc
    if not isinstance(value, dict):
        raise HaldArchiveError("config snapshot must be a JSON object")
    validate_contract(value)
    return value, digest


def evidence_schemas(config: dict[str, Any]) -> dict[str, str]:
    schema = str(config.get("schema_version"))
    try:
        expected = CONTRACT_EVIDENCE_SCHEMAS[schema]
    except KeyError as exc:
        raise HaldArchiveError("unsupported acquisition schema") from exc
    if schema.endswith("-v2") and config.get("evidence_schemas") != expected:
        raise HaldArchiveError("v2 evidence schemas do not match frozen identities")
    return dict(expected)


def _validate_decode_profiles(decode: dict[str, Any]) -> None:
    profiles = decode.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise HaldArchiveError("v2 decode profiles must be a non-empty list")
    if sum(int(profile["expected_files"]) for profile in profiles) != int(
        decode["expected_image_files"]
    ):
        raise HaldArchiveError("v2 decode profile count mismatch")
    profile_ids = [str(profile["profile_id"]) for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise HaldArchiveError("v2 decode profile IDs must be unique")
    allowed_modes = {str(value) for value in decode["allowed_modes"]}
    allowed_extensions = {
        str(value).casefold() for value in decode["allowed_extensions"]
    }
    exact_paths: set[str] = set()
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        include_prefix = profile.get("include_prefix")
        paths = profile.get("exact_paths")
        excludes = profile.get("exclude_paths")
        if not isinstance(paths, list) or not isinstance(excludes, list):
            raise HaldArchiveError(
                f"v2 profile paths must be lists: {profile_id}"
            )
        if (include_prefix is None) == (not paths):
            raise HaldArchiveError(
                f"v2 profile needs exactly one selector form: {profile_id}"
            )
        if include_prefix is not None:
            if not isinstance(include_prefix, str) or not include_prefix:
                raise HaldArchiveError(
                    f"invalid v2 profile prefix: {profile_id}"
                )
            if paths:
                raise HaldArchiveError(
                    f"prefix profile may not have exact paths: {profile_id}"
                )
        elif excludes:
            raise HaldArchiveError(
                f"exact-path profile may not have exclusions: {profile_id}"
            )
        for path in (*paths, *excludes):
            _safe_zip_path(str(path))
        overlap = exact_paths.intersection(str(path) for path in paths)
        if overlap:
            raise HaldArchiveError("v2 exact profile paths overlap")
        exact_paths.update(str(path) for path in paths)
        if int(profile["expected_files"]) <= 0:
            raise HaldArchiveError(
                f"v2 profile must retain at least one file: {profile_id}"
            )
        if str(profile["mode"]) not in allowed_modes:
            raise HaldArchiveError(
                f"v2 profile mode leaves allowed set: {profile_id}"
            )
        if str(profile["suffix"]).casefold() not in allowed_extensions:
            raise HaldArchiveError(
                f"v2 profile suffix leaves allowed set: {profile_id}"
            )
        if not isinstance(profile.get("bands"), list) or not isinstance(
            profile.get("bits_per_sample"), list
        ):
            raise HaldArchiveError(
                f"v2 profile bands/bits must be lists: {profile_id}"
            )
        if len(profile["bands"]) != len(profile["bits_per_sample"]):
            raise HaldArchiveError(
                f"v2 profile band/bit count mismatch: {profile_id}"
            )
        if profile.get("icc_profile_present") is not False:
            raise HaldArchiveError(
                f"v2 profile must require absent ICC: {profile_id}"
            )
        if int(profile["cube_side"]) != int(profile["hald_level"]) ** 2:
            raise HaldArchiveError(
                f"v2 profile Hald geometry mismatch: {profile_id}"
            )
    required_v2_flags = (
        "require_single_frame",
        "require_no_embedded_icc",
        "require_exactly_one_profile_per_image",
    )
    if not all(decode.get(key) is True for key in required_v2_flags):
        raise HaldArchiveError("v2 decode profile requirements were weakened")
    if decode.get("pillow_pixel_samples_authoritative_for_16bit_rgb") is not False:
        raise HaldArchiveError("v2 must reject Pillow as 16-bit pixel authority")
    for key in ("expected_image_records_sha256",):
        if len(str(decode[key])) != 64:
            raise HaldArchiveError(f"invalid v2 decode hash: {key}")


def validate_contract(config: dict[str, Any]) -> None:
    schema = str(config.get("schema_version"))
    evidence_schemas(config)
    archive = config["archive"]
    expected = config["expected_inventory"]
    primary = config["primary_universe"]
    decode = config["decode"]
    execution = config["execution"]
    if int(archive["expected_bytes"]) != int(archive["maximum_bytes"]):
        raise HaldArchiveError("archive byte ceiling must equal exact size")
    if len(str(archive["expected_md5"])) != 32:
        raise HaldArchiveError("invalid published archive MD5")
    if schema.endswith("-v2") and len(str(archive["expected_sha256"])) != 64:
        raise HaldArchiveError("invalid frozen archive SHA-256")
    if not str(archive["url"]).startswith(
        "https://rawtherapee.com/shared/"
    ):
        raise HaldArchiveError("archive URL leaves frozen host/path")
    destination = _safe_relative_filesystem_path(
        str(archive["destination"]), label="archive destination"
    )
    partial = _safe_relative_filesystem_path(
        str(archive["partial_destination"]), label="partial destination"
    )
    if destination == partial:
        raise HaldArchiveError("archive and partial destinations must differ")
    if int(expected["files"]) + int(expected["directories"]) != int(
        expected["entries"]
    ):
        raise HaldArchiveError("expected ZIP entry arithmetic mismatch")
    if (
        int(expected["png_files"])
        + int(expected["tiff_files"])
        + int(expected["text_files"])
        != int(expected["files"])
    ):
        raise HaldArchiveError("expected extension arithmetic mismatch")
    if int(primary["expected_files"]) != int(
        expected["primary_noncreative_color_files"]
    ):
        raise HaldArchiveError("primary universe count mismatch")
    if schema.endswith("-v2") and len(
        str(primary["expected_paths_sha256"])
    ) != 64:
        raise HaldArchiveError("invalid frozen primary-path SHA-256")
    if int(decode["expected_image_files"]) != (
        int(expected["png_files"]) + int(expected["tiff_files"])
    ):
        raise HaldArchiveError("image decode count mismatch")
    if int(execution["complete_runs"]) != 2:
        raise HaldArchiveError("AJ0B requires exactly two complete runs")
    required_execution = (
        "new_process_per_run",
        "rehash_archive_per_run",
        "redecode_all_images_per_run",
        "manifest_and_report_must_be_byte_identical",
    )
    if not all(bool(execution[key]) for key in required_execution):
        raise HaldArchiveError("AJ0B complete-run requirements were weakened")
    if archive["mirror_fallback_allowed"] or not archive[
        "retain_exact_archive_only"
    ]:
        raise HaldArchiveError("AJ0B archive retention policy was weakened")
    if not decode["decode_from_archive_memory_only"]:
        raise HaldArchiveError("AJ0B must decode in memory without extraction")
    if schema.endswith("-v2"):
        _validate_decode_profiles(decode)


def _validate_existing_archive(path: Path, archive: dict[str, Any]) -> dict[str, Any]:
    size = path.stat().st_size
    if size != int(archive["expected_bytes"]):
        raise HaldArchiveError(
            f"existing archive size mismatch: expected "
            f"{archive['expected_bytes']}, got {size}"
        )
    hashes = hash_file(path)
    if hashes["md5"] != str(archive["expected_md5"]):
        raise HaldArchiveError("existing archive MD5 mismatch")
    expected_sha256 = archive.get("expected_sha256")
    if expected_sha256 is not None and hashes["sha256"] != str(
        expected_sha256
    ):
        raise HaldArchiveError("existing archive SHA-256 mismatch")
    return {"bytes": size, **hashes}


def _safe_relative_filesystem_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or path.drive
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise HaldArchiveError(f"{label} must be a canonical relative path")
    return path


def _confined_path(root: Path, value: str, *, label: str) -> Path:
    relative = _safe_relative_filesystem_path(value, label=label)
    root_resolved = root.resolve()
    current = root_resolved
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise HaldArchiveError(f"{label} parent may not be a symlink")
    candidate = root_resolved / relative
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise HaldArchiveError(f"{label} escapes repository root") from exc
    if candidate.exists() and candidate.is_symlink():
        raise HaldArchiveError(f"{label} may not be a symlink")
    return candidate


def _confined_output_directory(root: Path, path: Path) -> Path:
    """Keep evidence writes below the caller-owned root without symlink hops."""

    root_absolute = Path(os.path.abspath(root))
    candidate = path if path.is_absolute() else root_absolute / path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise HaldArchiveError("output directory escapes execution root") from exc
    if not relative.parts:
        raise HaldArchiveError("output directory may not equal execution root")
    current = root_absolute
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise HaldArchiveError("output directory may not traverse a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_absolute.resolve())
    except ValueError as exc:
        raise HaldArchiveError("resolved output directory escapes execution root") from exc
    return candidate


def _quarantine_partial(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise HaldArchiveError("cannot quarantine a non-file partial")
    safe_reason = re.sub(r"[^a-z0-9_-]+", "-", reason.lower()).strip("-")
    base = path.with_name(
        f"{path.name}.quarantine-{safe_reason}-{path.stat().st_size}"
    )
    candidate = base
    index = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{index}")
        index += 1
    os.replace(path, candidate)
    return candidate


def acquire_archive(
    *,
    root: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Acquire the one exact archive with bounded, fail-closed resume."""

    validate_contract(config)
    archive = config["archive"]
    destination = _confined_path(
        root, str(archive["destination"]), label="archive destination"
    )
    partial = _confined_path(
        root, str(archive["partial_destination"]), label="partial destination"
    )
    if destination.is_file():
        return destination, _validate_existing_archive(destination, archive)
    if destination.exists():
        raise HaldArchiveError("archive destination exists but is not a file")

    partial.parent.mkdir(parents=True, exist_ok=True)
    maximum = int(archive["maximum_bytes"])
    start = partial.stat().st_size if partial.is_file() else 0
    if partial.exists() and not partial.is_file():
        raise HaldArchiveError("partial destination exists but is not a file")
    if start > maximum:
        quarantined = _quarantine_partial(partial, "size-overflow")
        raise HaldArchiveError(
            "partial archive exceeds byte ceiling and was quarantined as "
            f"{quarantined.name if quarantined else 'missing'}"
        )
    if start == int(archive["expected_bytes"]):
        observed = _validate_existing_archive(partial, archive)
        os.replace(partial, destination)
        return destination, observed

    headers = {
        "User-Agent": str(config["execution"]["network_user_agent"]),
    }
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(str(archive["url"]), headers=headers)
    timeout = int(config["execution"]["request_timeout_seconds"])
    chunk_bytes = int(config["execution"]["stream_chunk_bytes"])
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            expected_url = urlsplit(str(archive["url"]))
            observed_url = urlsplit(str(response.geturl()))
            expected_identity = (
                expected_url.scheme.casefold(),
                expected_url.hostname.casefold() if expected_url.hostname else None,
                expected_url.port,
                expected_url.path,
                expected_url.query,
            )
            observed_identity = (
                observed_url.scheme.casefold(),
                observed_url.hostname.casefold() if observed_url.hostname else None,
                observed_url.port,
                observed_url.path,
                observed_url.query,
            )
            if observed_identity != expected_identity:
                raise _QuarantinePartial("redirect-mismatch")
            status = getattr(response, "status", None)
            if start:
                content_range = response.headers.get("Content-Range", "")
                match = CONTENT_RANGE.fullmatch(content_range)
                valid_range = False
                if match is not None:
                    range_start, range_end, range_total = (
                        int(value) for value in match.groups()
                    )
                    valid_range = (
                        range_start == start
                        and range_start <= range_end
                        and range_end < range_total
                        and range_total == int(archive["expected_bytes"])
                    )
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            parsed_length = int(content_length)
                        except (TypeError, ValueError):
                            raise _QuarantinePartial(
                                "malformed-content-length"
                            ) from None
                        valid_range = valid_range and parsed_length == (
                            range_end - range_start + 1
                        )
                if status != 206 or not valid_range:
                    raise _QuarantinePartial("range-mismatch")
                mode = "ab"
                total = start
            else:
                if status not in (None, 200):
                    raise HaldArchiveError(
                        f"unexpected download status: {status}"
                    )
                mode = "wb"
                total = 0
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(chunk_bytes)
                    if not chunk:
                        break
                    if total + len(chunk) > maximum:
                        raise _QuarantinePartial("byte-overflow")
                    handle.write(chunk)
                    total += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except _QuarantinePartial as exc:
        quarantined = _quarantine_partial(partial, str(exc))
        suffix = "no partial existed" if quarantined is None else quarantined.name
        raise HaldArchiveError(
            f"invalid range/overflow; task partial quarantined as {suffix}"
        ) from exc

    if total != int(archive["expected_bytes"]):
        raise HaldArchiveError(
            f"incomplete archive: expected {archive['expected_bytes']}, got {total}"
        )
    observed = _validate_existing_archive(partial, archive)
    os.replace(partial, destination)
    return destination, observed


def _integer_cube_root(value: int) -> int:
    if value <= 0:
        raise HaldArchiveError("Hald image side must be positive")
    approximate = int(round(value ** (1.0 / 3.0)))
    for candidate in range(max(1, approximate - 2), approximate + 3):
        if candidate**3 == value:
            return candidate
    raise HaldArchiveError(f"image side {value} is not an integer cube")


def hald_geometry(width: int, height: int) -> dict[str, int]:
    if width != height:
        raise HaldArchiveError("Hald image must be square")
    level = _integer_cube_root(width)
    if level <= 1:
        raise HaldArchiveError("Hald level must exceed one")
    return {
        "width": width,
        "height": height,
        "hald_level": level,
        "cube_side": level * level,
    }


def _central_directory(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    tail_size = min(size, 65557)
    with path.open("rb") as handle:
        handle.seek(size - tail_size)
        tail = handle.read(tail_size)
        position = tail.rfind(EOCD_SIGNATURE)
        if position < 0 or position + EOCD_STRUCT.size > len(tail):
            raise HaldArchiveError("ZIP EOCD not found")
        (
            signature,
            disk,
            directory_disk,
            disk_entries,
            total_entries,
            directory_bytes,
            directory_offset,
            comment_bytes,
        ) = EOCD_STRUCT.unpack(tail[position : position + EOCD_STRUCT.size])
        if signature != EOCD_SIGNATURE:
            raise HaldArchiveError("invalid ZIP EOCD signature")
        if disk or directory_disk or disk_entries != total_entries:
            raise HaldArchiveError("multi-disk ZIP is forbidden")
        if position + EOCD_STRUCT.size + comment_bytes != len(tail):
            raise HaldArchiveError("ZIP EOCD comment/length mismatch")
        if directory_offset + directory_bytes > size - EOCD_STRUCT.size:
            raise HaldArchiveError("ZIP central directory leaves archive bounds")
        handle.seek(directory_offset)
        directory = handle.read(directory_bytes)
    if len(directory) != directory_bytes:
        raise HaldArchiveError("short ZIP central-directory read")
    return {
        "offset": int(directory_offset),
        "bytes": int(directory_bytes),
        "entries": int(total_entries),
        "sha256": hashlib.sha256(directory).hexdigest(),
    }


def _safe_zip_path(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise HaldArchiveError(f"unsafe ZIP path spelling: {name!r}")
    raw = name[:-1] if name.endswith("/") else name
    if not raw or raw.startswith("/"):
        raise HaldArchiveError(f"absolute/empty ZIP path: {name!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise HaldArchiveError(f"traversal ZIP path: {name!r}")
    if ":" in path.parts[0]:
        raise HaldArchiveError(f"drive-like ZIP path: {name!r}")
    normalized = posixpath.normpath(raw)
    if normalized != raw:
        raise HaldArchiveError(f"non-canonical ZIP path: {name!r}")
    return normalized.casefold()


def _bits_per_sample(raw: bytes, image: Image.Image, suffix: str) -> list[int]:
    if suffix == ".png":
        if len(raw) < 25 or raw[:8] != b"\x89PNG\r\n\x1a\n":
            raise HaldArchiveError("invalid PNG signature/IHDR")
        bit_depth = int(raw[24])
        return [bit_depth] * len(image.getbands())
    tag = getattr(image, "tag_v2", None)
    bits = tag.get(258) if tag is not None else None
    if bits is None:
        raise HaldArchiveError("TIFF BitsPerSample metadata is missing")
    if isinstance(bits, int):
        return [int(bits)] * len(image.getbands())
    values = [int(value) for value in bits]
    if len(values) != len(image.getbands()) or any(
        value <= 0 or value > 32 for value in values
    ):
        raise HaldArchiveError("invalid TIFF BitsPerSample metadata")
    return values


def _image_record(
    info: zipfile.ZipInfo,
    raw: bytes,
    *,
    allowed_modes: tuple[str, ...],
) -> dict[str, Any]:
    suffix = PurePosixPath(info.filename).suffix.lower()
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            expected_format = "PNG" if suffix == ".png" else "TIFF"
            if opened.format != expected_format:
                raise HaldArchiveError(
                    f"image format/suffix mismatch: {info.filename}"
                )
            if int(getattr(opened, "n_frames", 1)) != 1:
                raise HaldArchiveError(
                    f"multi-frame Hald image is forbidden: {info.filename}"
                )
            if opened.mode not in allowed_modes:
                raise HaldArchiveError(
                    f"unsupported image mode {opened.mode}: {info.filename}"
                )
            geometry = hald_geometry(opened.width, opened.height)
            bits = _bits_per_sample(raw, opened, suffix)
            return {
                "path": info.filename,
                "bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "format": str(opened.format),
                "mode": opened.mode,
                "bands": list(opened.getbands()),
                "bits_per_sample": bits,
                "icc_profile_present": bool(opened.info.get("icc_profile")),
                **geometry,
            }
    except HaldArchiveError:
        raise
    except Exception as exc:
        raise HaldArchiveError(
            f"image decode failed for {info.filename}: {exc!r}"
        ) from exc


def _profile_matches_path(profile: dict[str, Any], path: str) -> bool:
    exact_paths = {str(value) for value in profile["exact_paths"]}
    if exact_paths:
        return path in exact_paths
    include_prefix = str(profile["include_prefix"])
    excludes = {str(value) for value in profile["exclude_paths"]}
    return path.startswith(include_prefix) and path not in excludes


def _decode_profile_audit(
    image_records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any] | None:
    if not str(config["schema_version"]).endswith("-v2"):
        return None
    profiles = config["decode"]["profiles"]
    counts: Counter[str] = Counter()
    assignments: list[dict[str, str]] = []
    profile_by_id = {
        str(profile["profile_id"]): profile for profile in profiles
    }
    for record in image_records:
        matches = [
            profile
            for profile in profiles
            if _profile_matches_path(profile, str(record["path"]))
        ]
        if len(matches) != 1:
            raise HaldArchiveError(
                "v2 image must match exactly one decode profile: "
                f"{record['path']}"
            )
        profile = matches[0]
        profile_id = str(profile["profile_id"])
        observed = {
            "format": record["format"],
            "suffix": PurePosixPath(str(record["path"])).suffix.casefold(),
            "mode": record["mode"],
            "bands": record["bands"],
            "bits_per_sample": record["bits_per_sample"],
            "hald_level": record["hald_level"],
            "cube_side": record["cube_side"],
            "icc_profile_present": record["icc_profile_present"],
        }
        expected = {
            "format": str(profile["format"]),
            "suffix": str(profile["suffix"]).casefold(),
            "mode": str(profile["mode"]),
            "bands": [str(value) for value in profile["bands"]],
            "bits_per_sample": [
                int(value) for value in profile["bits_per_sample"]
            ],
            "hald_level": int(profile["hald_level"]),
            "cube_side": int(profile["cube_side"]),
            "icc_profile_present": bool(profile["icc_profile_present"]),
        }
        if observed != expected:
            raise HaldArchiveError(
                f"v2 decode profile mismatch for {record['path']}: "
                f"{profile_id}"
            )
        counts[profile_id] += 1
        assignments.append(
            {"path": str(record["path"]), "profile_id": profile_id}
        )
    expected_counts = {
        profile_id: int(profile["expected_files"])
        for profile_id, profile in profile_by_id.items()
    }
    if dict(counts) != expected_counts:
        raise HaldArchiveError("v2 decode profile counts mismatch")
    assignments = sorted(assignments, key=lambda row: row["path"])
    return {
        "counts": expected_counts,
        "assignments": assignments,
        "assignments_sha256": canonical_sha256(assignments),
        "profiles_exact": True,
    }


def _inventory(infos: list[zipfile.ZipInfo]) -> dict[str, int]:
    files = [info for info in infos if not info.is_dir()]
    extensions = Counter(
        PurePosixPath(info.filename).suffix.lower() for info in files
    )
    return {
        "entries": len(infos),
        "files": len(files),
        "directories": len(infos) - len(files),
        "png_files": extensions[".png"],
        "tiff_files": extensions[".tif"] + extensions[".tiff"],
        "text_files": extensions[".txt"],
        "color_files": sum(
            info.filename.startswith("HaldCLUT/Color/") for info in files
        ),
        "black_and_white_files": sum(
            info.filename.startswith("HaldCLUT/Black-and-White/")
            for info in files
        ),
        "creative_pack_color_files": sum(
            info.filename.startswith("HaldCLUT/Color/CreativePack-1/")
            for info in files
        ),
    }


def _primary_paths(
    infos: list[zipfile.ZipInfo], primary: dict[str, Any]
) -> list[str]:
    include = str(primary["include_prefix"])
    excludes = tuple(str(value) for value in primary["exclude_prefixes"])
    return sorted(
        info.filename
        for info in infos
        if not info.is_dir()
        and info.filename.startswith(include)
        and not info.filename.startswith(excludes)
    )


def audit_archive(
    *,
    archive_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Rehash, inventory, CRC-read and decode every frozen archive member."""

    validate_contract(config)
    observed_archive = _validate_existing_archive(
        archive_path, config["archive"]
    )
    expected = config["expected_inventory"]
    central = _central_directory(archive_path)
    expected_central = {
        "offset": int(expected["central_directory_offset"]),
        "bytes": int(expected["central_directory_bytes"]),
        "entries": int(expected["entries"]),
        "sha256": str(expected["central_directory_sha256"]),
    }
    if central != expected_central:
        raise HaldArchiveError("ZIP central-directory identity mismatch")
    path_keys: set[str] = set()
    image_records: list[dict[str, Any]] = []
    readme: dict[str, Any] | None = None
    encrypted_entries = 0
    symlink_like_entries = 0

    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        inventory = _inventory(infos)
        primary_paths = _primary_paths(infos, config["primary_universe"])
        inventory["primary_noncreative_color_files"] = len(primary_paths)
        expected_inventory = {
            key: int(expected[key])
            for key in (
                "entries",
                "files",
                "directories",
                "png_files",
                "tiff_files",
                "text_files",
                "color_files",
                "black_and_white_files",
                "creative_pack_color_files",
                "primary_noncreative_color_files",
            )
        }
        if inventory != expected_inventory:
            raise HaldArchiveError("ZIP inventory identity mismatch")
        for info in infos:
            path_key = _safe_zip_path(info.filename)
            if path_key in path_keys:
                raise HaldArchiveError(
                    f"duplicate normalized ZIP path: {info.filename}"
                )
            path_keys.add(path_key)
            if info.flag_bits & 0x1:
                encrypted_entries += 1
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                symlink_like_entries += 1
        if encrypted_entries:
            raise HaldArchiveError("encrypted ZIP entries are forbidden")
        if symlink_like_entries:
            raise HaldArchiveError("symlink-like ZIP entries are forbidden")
        allowed_modes = tuple(str(value) for value in config["decode"]["allowed_modes"])
        for info in infos:
            if info.is_dir():
                continue
            raw = archive.read(info)
            if len(raw) != info.file_size:
                raise HaldArchiveError(
                    f"short decoded ZIP member: {info.filename}"
                )
            suffix = PurePosixPath(info.filename).suffix.lower()
            if suffix in tuple(config["decode"]["allowed_extensions"]):
                image_records.append(
                    _image_record(info, raw, allowed_modes=allowed_modes)
                )
            elif info.filename == str(expected["readme_path"]):
                readme = {
                    "path": info.filename,
                    "bytes": len(raw),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "license_marker_present": b"CC BY-SA 4.0" in raw,
                    "version_marker_present": (
                        b"version 2015-09-20" in raw
                    ),
                }
            else:
                raise HaldArchiveError(
                    f"unexpected non-image ZIP member: {info.filename}"
                )

    image_records = sorted(image_records, key=lambda row: row["path"])
    image_records_sha256 = canonical_sha256(image_records)
    primary_paths_sha256 = canonical_sha256(primary_paths)
    decode_profile_audit = _decode_profile_audit(image_records, config)
    inventory_exact = inventory == expected_inventory
    central_exact = central == expected_central
    readme_exact = readme == {
        "path": str(expected["readme_path"]),
        "bytes": int(expected["readme_bytes"]),
        "crc32": str(expected["readme_crc32"]),
        "sha256": str(expected["readme_sha256"]),
        "license_marker_present": True,
        "version_marker_present": True,
    }
    gates = {
        "archive_size_exact": observed_archive["bytes"]
        == int(config["archive"]["expected_bytes"]),
        "archive_md5_exact": observed_archive["md5"]
        == str(config["archive"]["expected_md5"]),
        "central_directory_exact": central_exact,
        "safe_paths_only": True,
        "duplicate_normalized_paths": 0,
        "encrypted_entries": encrypted_entries,
        "symlink_like_entries": symlink_like_entries,
        "inventory_exact": inventory_exact,
        "readme_exact": readme_exact,
        "all_images_decode": len(image_records)
        == int(config["decode"]["expected_image_files"]),
        "primary_membership_exact": len(primary_paths)
        == int(config["primary_universe"]["expected_files"]),
        "all_member_crc_reads_complete": True,
    }
    if decode_profile_audit is not None:
        gates.update(
            {
                "archive_sha256_exact": observed_archive["sha256"]
                == str(config["archive"]["expected_sha256"]),
                "decode_profiles_exact": decode_profile_audit[
                    "profiles_exact"
                ],
                "primary_paths_sha256_exact": primary_paths_sha256
                == str(
                    config["primary_universe"]["expected_paths_sha256"]
                ),
                "image_records_sha256_exact": image_records_sha256
                == str(
                    config["decode"]["expected_image_records_sha256"]
                ),
            }
        )
    automatic_pass = (
        gates["archive_size_exact"]
        and gates["archive_md5_exact"]
        and gates["central_directory_exact"]
        and gates["safe_paths_only"]
        and gates["duplicate_normalized_paths"]
        == int(config["automatic_gates"]["duplicate_normalized_paths"])
        and gates["encrypted_entries"]
        == int(config["automatic_gates"]["encrypted_entries"])
        and gates["symlink_like_entries"]
        == int(config["automatic_gates"]["symlink_like_entries"])
        and gates["inventory_exact"]
        and gates["readme_exact"]
        and gates["all_images_decode"]
        and gates["primary_membership_exact"]
        and gates["all_member_crc_reads_complete"]
    )
    if decode_profile_audit is not None:
        automatic_pass = automatic_pass and all(
            gates[key]
            for key in (
                "archive_sha256_exact",
                "decode_profiles_exact",
                "primary_paths_sha256_exact",
                "image_records_sha256_exact",
            )
        )
    result = {
        "archive": observed_archive,
        "central_directory": central,
        "inventory": inventory,
        "readme": readme,
        "primary_paths": primary_paths,
        "primary_paths_sha256": primary_paths_sha256,
        "image_records": image_records,
        "image_records_sha256": image_records_sha256,
        "gates": gates,
        "automatic_pass": bool(automatic_pass),
        "operator_applied": False,
        "photograph_rendered": False,
    }
    if decode_profile_audit is not None:
        result["decode_profile_audit"] = decode_profile_audit
    return result


def run_acquisition(
    *,
    root: Path,
    config: dict[str, Any],
    config_sha256: str,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validate_contract(config)
    schemas = evidence_schemas(config)
    output_dir = _confined_output_directory(root, output_dir)
    archive_path, acquired = acquire_archive(root=root, config=config)
    audited = audit_archive(archive_path=archive_path, config=config)
    if acquired != audited["archive"]:
        raise HaldArchiveError("acquisition and audit archive identity mismatch")

    manifest = {
        "schema_version": schemas["manifest"],
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "archive_relative_path": archive_path.relative_to(root).as_posix(),
        "archive": audited["archive"],
        "central_directory": audited["central_directory"],
        "readme": audited["readme"],
        "inventory": audited["inventory"],
        "primary_paths": audited["primary_paths"],
        "primary_paths_sha256": audited["primary_paths_sha256"],
        "image_records": audited["image_records"],
        "image_records_sha256": audited["image_records_sha256"],
        "operator_applied": False,
        "photograph_rendered": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {
        "schema_version": schemas["report"],
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "archive": audited["archive"],
        "inventory": audited["inventory"],
        "primary_paths_sha256": audited["primary_paths_sha256"],
        "image_records_sha256": audited["image_records_sha256"],
        "gates": audited["gates"],
        "single_run_pass": audited["automatic_pass"],
        "automatic_pass": False,
        "repeat_confirmation_required": True,
        "structural_audit_ready": False,
        "visual_review_allowed": False,
        "operator_applied": False,
        "photograph_rendered": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    if "decode_profile_audit" in audited:
        manifest["decode_profile_audit"] = audited["decode_profile_audit"]
        report["decode_profile_audit"] = audited["decode_profile_audit"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "automatic_report.json"
    write_canonical_json(manifest_path, manifest)
    write_canonical_json(report_path, report)
    return {
        "manifest": manifest,
        "report": report,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "report_path": report_path,
        "report_sha256": sha256_file(report_path),
    }


BASE_GATE_KEYS = frozenset(
    {
        "archive_size_exact",
        "archive_md5_exact",
        "central_directory_exact",
        "safe_paths_only",
        "duplicate_normalized_paths",
        "encrypted_entries",
        "symlink_like_entries",
        "inventory_exact",
        "readme_exact",
        "all_images_decode",
        "primary_membership_exact",
        "all_member_crc_reads_complete",
    }
)
V2_GATE_KEYS = frozenset(
    {
        "archive_sha256_exact",
        "decode_profiles_exact",
        "primary_paths_sha256_exact",
        "image_records_sha256_exact",
    }
)
INVENTORY_KEYS = frozenset(
    {
        "entries",
        "files",
        "directories",
        "png_files",
        "tiff_files",
        "text_files",
        "color_files",
        "black_and_white_files",
        "creative_pack_color_files",
        "primary_noncreative_color_files",
    }
)
IMAGE_RECORD_KEYS = frozenset(
    {
        "path",
        "bytes",
        "compressed_bytes",
        "crc32",
        "sha256",
        "format",
        "mode",
        "bands",
        "bits_per_sample",
        "icc_profile_present",
        "width",
        "height",
        "hald_level",
        "cube_side",
    }
)


def _is_strict_int(value: Any) -> bool:
    return type(value) is int


def _strict_archive_shape(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"bytes", "md5", "sha256"}
        and _is_strict_int(value["bytes"])
        and isinstance(value["md5"], str)
        and len(value["md5"]) == 32
        and isinstance(value["sha256"], str)
        and len(value["sha256"]) == 64
    )


def _strict_inventory_shape(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == INVENTORY_KEYS
        and all(_is_strict_int(item) for item in value.values())
    )


def _strict_image_records_shape(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    integer_keys = {
        "bytes",
        "compressed_bytes",
        "width",
        "height",
        "hald_level",
        "cube_side",
    }
    string_keys = {"path", "crc32", "sha256", "format", "mode"}
    for row in value:
        if not isinstance(row, dict) or set(row) != IMAGE_RECORD_KEYS:
            return False
        if not all(_is_strict_int(row[key]) for key in integer_keys):
            return False
        if not all(isinstance(row[key], str) for key in string_keys):
            return False
        if not isinstance(row["bands"], list) or not all(
            isinstance(item, str) for item in row["bands"]
        ):
            return False
        if not isinstance(row["bits_per_sample"], list) or not all(
            _is_strict_int(item) for item in row["bits_per_sample"]
        ):
            return False
        if type(row["icc_profile_present"]) is not bool:
            return False
    return True


def _strict_profile_audit_shape(
    value: Any, config: dict[str, Any]
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "counts",
        "assignments",
        "assignments_sha256",
        "profiles_exact",
    }:
        return False
    profile_ids = {
        str(profile["profile_id"]) for profile in config["decode"]["profiles"]
    }
    counts = value["counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != profile_ids
        or not all(_is_strict_int(item) for item in counts.values())
    ):
        return False
    assignments = value["assignments"]
    if not isinstance(assignments, list):
        return False
    for row in assignments:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "profile_id"}
            or not isinstance(row["path"], str)
            or not isinstance(row["profile_id"], str)
            or row["profile_id"] not in profile_ids
        ):
            return False
    return (
        isinstance(value["assignments_sha256"], str)
        and len(value["assignments_sha256"]) == 64
        and value["profiles_exact"] is True
    )


def _strict_evidence_shape(
    value: Any, *, kind: str, config: dict[str, Any]
) -> bool:
    if not isinstance(value, dict):
        return False
    is_v2 = str(config["schema_version"]).endswith("-v2")
    common = {
        "schema_version",
        "experiment_id",
        "software_commit",
        "config_sha256",
        "archive",
        "inventory",
        "primary_paths_sha256",
        "image_records_sha256",
        "operator_applied",
        "photograph_rendered",
        "claim_ceiling",
    }
    if kind == "manifest":
        expected_keys = common | {
            "archive_relative_path",
            "central_directory",
            "readme",
            "primary_paths",
            "image_records",
        }
    elif kind == "report":
        expected_keys = common | {
            "gates",
            "single_run_pass",
            "automatic_pass",
            "repeat_confirmation_required",
            "structural_audit_ready",
            "visual_review_allowed",
        }
    else:
        raise HaldArchiveError("unsupported evidence kind")
    if is_v2:
        expected_keys.add("decode_profile_audit")
    if set(value) != expected_keys:
        return False
    string_keys = {
        "schema_version",
        "experiment_id",
        "software_commit",
        "config_sha256",
        "primary_paths_sha256",
        "image_records_sha256",
        "claim_ceiling",
    }
    if not all(isinstance(value[key], str) for key in string_keys):
        return False
    if not _strict_archive_shape(value["archive"]):
        return False
    if not _strict_inventory_shape(value["inventory"]):
        return False
    if (
        type(value["operator_applied"]) is not bool
        or type(value["photograph_rendered"]) is not bool
    ):
        return False
    if is_v2 and not _strict_profile_audit_shape(
        value["decode_profile_audit"], config
    ):
        return False
    if kind == "manifest":
        if (
            not isinstance(value["archive_relative_path"], str)
            or not isinstance(value["central_directory"], dict)
            or set(value["central_directory"])
            != {"offset", "bytes", "entries", "sha256"}
            or not all(
                _is_strict_int(value["central_directory"][key])
                for key in ("offset", "bytes", "entries")
            )
            or not isinstance(value["central_directory"]["sha256"], str)
            or not isinstance(value["readme"], dict)
            or set(value["readme"])
            != {
                "path",
                "bytes",
                "crc32",
                "sha256",
                "license_marker_present",
                "version_marker_present",
            }
            or not isinstance(value["readme"]["path"], str)
            or not _is_strict_int(value["readme"]["bytes"])
            or not isinstance(value["readme"]["crc32"], str)
            or not isinstance(value["readme"]["sha256"], str)
            or type(value["readme"]["license_marker_present"]) is not bool
            or type(value["readme"]["version_marker_present"]) is not bool
            or not isinstance(value["primary_paths"], list)
            or not all(
                isinstance(path, str) for path in value["primary_paths"]
            )
            or not _strict_image_records_shape(value["image_records"])
        ):
            return False
    else:
        gates = value["gates"]
        expected_gate_keys = BASE_GATE_KEYS | (V2_GATE_KEYS if is_v2 else set())
        if not isinstance(gates, dict) or set(gates) != expected_gate_keys:
            return False
        zero_keys = {
            "duplicate_normalized_paths",
            "encrypted_entries",
            "symlink_like_entries",
        }
        if not all(_is_strict_int(gates[key]) for key in zero_keys):
            return False
        if not all(
            type(gates[key]) is bool for key in set(gates) - zero_keys
        ):
            return False
        bool_keys = {
            "single_run_pass",
            "automatic_pass",
            "repeat_confirmation_required",
            "structural_audit_ready",
            "visual_review_allowed",
        }
        if not all(type(value[key]) is bool for key in bool_keys):
            return False
    return True


def finalize_repeat_evidence(
    *,
    first_dir: Path,
    second_dir: Path,
    config: dict[str, Any],
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    """Promote two separate per-run audits only when their bytes are exact."""

    validate_contract(config)
    schemas = evidence_schemas(config)
    first_manifest = (first_dir / "manifest.json").read_bytes()
    second_manifest = (second_dir / "manifest.json").read_bytes()
    first_report = (first_dir / "automatic_report.json").read_bytes()
    second_report = (second_dir / "automatic_report.json").read_bytes()
    manifest_exact = first_manifest == second_manifest
    report_exact = first_report == second_report
    try:
        report_a = json.loads(first_report)
        report_b = json.loads(second_report)
        manifest_a = json.loads(first_manifest)
        manifest_b = json.loads(second_manifest)
    except Exception as exc:
        raise HaldArchiveError("per-run evidence is not valid JSON") from exc
    if not all(
        isinstance(value, dict)
        for value in (report_a, report_b, manifest_a, manifest_b)
    ):
        raise HaldArchiveError("per-run evidence roots must be JSON objects")
    evidence_shapes_exact = all(
        _strict_evidence_shape(value, kind=kind, config=config)
        for value, kind in (
            (report_a, "report"),
            (report_b, "report"),
            (manifest_a, "manifest"),
            (manifest_b, "manifest"),
        )
    )

    canonical_bytes_exact = all(
        raw == _canonical_bytes(value)
        for raw, value in (
            (first_report, report_a),
            (second_report, report_b),
            (first_manifest, manifest_a),
            (second_manifest, manifest_b),
        )
    )
    identities_exact = all(
        value.get("config_sha256") == config_sha256
        and value.get("software_commit") == software_commit
        and value.get("experiment_id") == config["experiment_id"]
        and value.get("claim_ceiling") == config["claim_ceiling"]
        for value in (report_a, report_b, manifest_a, manifest_b)
    )
    schemas_exact = evidence_shapes_exact and (
        report_a.get("schema_version") == schemas["report"]
        and report_b.get("schema_version") == schemas["report"]
        and manifest_a.get("schema_version") == schemas["manifest"]
        and manifest_b.get("schema_version") == schemas["manifest"]
    )
    single_runs_pass = all(
        report.get("single_run_pass") is True
        and report.get("automatic_pass") is False
        and report.get("structural_audit_ready") is False
        and report.get("repeat_confirmation_required") is True
        and report.get("visual_review_allowed") is False
        and report.get("operator_applied") is False
        and report.get("photograph_rendered") is False
        for report in (report_a, report_b)
    )
    required_true_gates = {
        "archive_size_exact",
        "archive_md5_exact",
        "central_directory_exact",
        "safe_paths_only",
        "inventory_exact",
        "readme_exact",
        "all_images_decode",
        "primary_membership_exact",
        "all_member_crc_reads_complete",
    }
    if str(config["schema_version"]).endswith("-v2"):
        required_true_gates.update(
            {
                "archive_sha256_exact",
                "decode_profiles_exact",
                "primary_paths_sha256_exact",
                "image_records_sha256_exact",
            }
        )
    required_zero_gates = {
        "duplicate_normalized_paths",
        "encrypted_entries",
        "symlink_like_entries",
    }
    gate_facts_exact = evidence_shapes_exact and all(
        isinstance(report.get("gates"), dict)
        and all(report["gates"].get(key) is True for key in required_true_gates)
        and all(
            type(report["gates"].get(key)) is int
            and report["gates"].get(key) == 0
            for key in required_zero_gates
        )
        for report in (report_a, report_b)
    )
    archive_identity_exact = (
        report_a.get("archive") == report_b.get("archive")
        == manifest_a.get("archive")
        == manifest_b.get("archive")
    )
    report_manifest_bindings_exact = all(
        report.get("experiment_id") == manifest.get("experiment_id")
        and report.get("inventory") == manifest.get("inventory")
        and report.get("primary_paths_sha256")
        == manifest.get("primary_paths_sha256")
        and report.get("image_records_sha256")
        == manifest.get("image_records_sha256")
        and report.get("claim_ceiling") == manifest.get("claim_ceiling")
        and report.get("decode_profile_audit")
        == manifest.get("decode_profile_audit")
        and manifest.get("operator_applied") is False
        and manifest.get("photograph_rendered") is False
        for report, manifest in (
            (report_a, manifest_a),
            (report_b, manifest_b),
        )
    )
    manifest_facts_exact = evidence_shapes_exact and all(
        manifest.get("archive", {}).get("bytes")
        == int(config["archive"]["expected_bytes"])
        and manifest.get("archive", {}).get("md5")
        == str(config["archive"]["expected_md5"])
        and manifest.get("archive_relative_path")
        == str(config["archive"]["destination"]).replace("\\", "/")
        and manifest.get("inventory")
        == {
            key: int(config["expected_inventory"][key])
            for key in (
                "entries",
                "files",
                "directories",
                "png_files",
                "tiff_files",
                "text_files",
                "color_files",
                "black_and_white_files",
                "creative_pack_color_files",
                "primary_noncreative_color_files",
            )
        }
        and manifest.get("central_directory")
        == {
            "offset": int(
                config["expected_inventory"]["central_directory_offset"]
            ),
            "bytes": int(
                config["expected_inventory"]["central_directory_bytes"]
            ),
            "entries": int(config["expected_inventory"]["entries"]),
            "sha256": str(
                config["expected_inventory"]["central_directory_sha256"]
            ),
        }
        and manifest.get("readme")
        == {
            "path": str(config["expected_inventory"]["readme_path"]),
            "bytes": int(config["expected_inventory"]["readme_bytes"]),
            "crc32": str(config["expected_inventory"]["readme_crc32"]),
            "sha256": str(config["expected_inventory"]["readme_sha256"]),
            "license_marker_present": True,
            "version_marker_present": True,
        }
        and isinstance(manifest.get("primary_paths"), list)
        and len(manifest["primary_paths"])
        == int(config["primary_universe"]["expected_files"])
        and manifest["primary_paths"] == sorted(manifest["primary_paths"])
        and len({path.casefold() for path in manifest["primary_paths"]})
        == len(manifest["primary_paths"])
        and all(
            isinstance(path, str)
            and path.startswith(config["primary_universe"]["include_prefix"])
            and not any(
                path.startswith(prefix)
                for prefix in config["primary_universe"]["exclude_prefixes"]
            )
            for path in manifest["primary_paths"]
        )
        and canonical_sha256(manifest["primary_paths"])
        == manifest.get("primary_paths_sha256")
        and isinstance(manifest.get("image_records"), list)
        and len(manifest["image_records"])
        == int(config["decode"]["expected_image_files"])
        and manifest["image_records"]
        == sorted(manifest["image_records"], key=lambda row: row["path"])
        and canonical_sha256(manifest["image_records"])
        == manifest.get("image_records_sha256")
        and (
            (
                str(config["schema_version"]).endswith("-v2")
                and manifest.get("archive", {}).get("sha256")
                == str(config["archive"]["expected_sha256"])
                and manifest.get("primary_paths_sha256")
                == str(
                    config["primary_universe"]["expected_paths_sha256"]
                )
                and manifest.get("image_records_sha256")
                == str(
                    config["decode"]["expected_image_records_sha256"]
                )
                and manifest.get("decode_profile_audit")
                == _decode_profile_audit(manifest["image_records"], config)
            )
            or (
                not str(config["schema_version"]).endswith("-v2")
                and "decode_profile_audit" not in manifest
            )
        )
        for manifest in (manifest_a, manifest_b)
    )
    checks = {
        "manifest_bytes_exact": manifest_exact,
        "report_bytes_exact": report_exact,
        "canonical_evidence_bytes_exact": canonical_bytes_exact,
        "schema_identities_exact": schemas_exact,
        "config_and_software_identities_exact": identities_exact,
        "both_single_runs_pass": single_runs_pass,
        "single_run_gate_facts_exact": gate_facts_exact,
        "archive_identity_exact": archive_identity_exact,
        "report_manifest_bindings_exact": report_manifest_bindings_exact,
        "manifest_facts_exact": manifest_facts_exact,
    }
    automatic_pass = all(checks.values())
    archive = report_a.get("archive") if archive_identity_exact else None
    return {
        "schema_version": schemas["repeat_decision"],
        "experiment_id": report_a.get("experiment_id"),
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "archive": archive,
        "run_a": {
            "manifest_sha256": hashlib.sha256(first_manifest).hexdigest(),
            "report_sha256": hashlib.sha256(first_report).hexdigest(),
        },
        "run_b": {
            "manifest_sha256": hashlib.sha256(second_manifest).hexdigest(),
            "report_sha256": hashlib.sha256(second_report).hexdigest(),
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "structural_audit_ready": automatic_pass,
        "visual_review_allowed": False,
        "operator_applied": False,
        "photograph_rendered": False,
        "claim_ceiling": report_a.get("claim_ceiling"),
    }


__all__ = [
    "HaldArchiveError",
    "acquire_archive",
    "audit_archive",
    "canonical_sha256",
    "evidence_schemas",
    "finalize_repeat_evidence",
    "hald_geometry",
    "hash_file",
    "load_config_snapshot",
    "run_acquisition",
    "sha256_file",
    "validate_contract",
    "write_canonical_json",
]
