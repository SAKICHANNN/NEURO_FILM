"""Bounded acquisition and archive audit for the Apollo AS07-03 step chart."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import requests


SCHEMA = "neuro_film.u6_p2k_apollo_step_chart_source.v1"
MANIFEST_SCHEMA = "neuro_film.u6_p2k_apollo_step_chart_acquisition.v1"


class ApolloStepChartSourceError(RuntimeError):
    """Raised when the bounded source contract fails closed."""


def sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ApolloStepChartSourceError("unsupported source schema")
    source = config.get("source")
    acquisition = config.get("acquisition")
    evidence = config.get("magazine_evidence")
    if not isinstance(source, Mapping) or not isinstance(acquisition, Mapping):
        raise ApolloStepChartSourceError("source/acquisition contract is missing")
    if not isinstance(evidence, Mapping):
        raise ApolloStepChartSourceError("magazine evidence is missing")
    url = str(source.get("download_url", ""))
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != source.get("allowed_download_host")
        or parsed.path != "/download_step"
        or parsed.query != "step_name=AS07-03-StepChart"
    ):
        raise ApolloStepChartSourceError("download URL escaped the frozen source")
    if source.get("download_filename") != "AS07-03-StepChart.zip":
        raise ApolloStepChartSourceError("download filename drifted")
    if acquisition.get("resume_allowed") is not False:
        raise ApolloStepChartSourceError("range resume must remain disabled")
    for key in (
        "maximum_compressed_bytes",
        "maximum_uncompressed_bytes",
        "maximum_member_count",
        "request_timeout_seconds",
        "chunk_bytes",
    ):
        value = acquisition.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ApolloStepChartSourceError(f"{key} must be a positive integer")
    if (
        evidence.get("photo_roll") != "3"
        or evidence.get("film_code") != "SO368"
        or evidence.get("film_stock_id") != "kodak_ektachrome_so_368"
    ):
        raise ApolloStepChartSourceError("magazine association drifted")
    if config.get("training_allowed") is not False:
        raise ApolloStepChartSourceError("training must remain forbidden")
    if config.get("operator_fitting_allowed") is not False:
        raise ApolloStepChartSourceError("operator fitting must remain forbidden")


def validate_parent_evidence(root: Path, config: Mapping[str, Any]) -> None:
    """Bind the acquisition to the already-audited magazine metadata."""

    validate_config(config)
    evidence = config["magazine_evidence"]
    parent = root / str(evidence["parent_path"])
    if not parent.is_file():
        raise ApolloStepChartSourceError("magazine evidence file is missing")
    if sha256_file(parent) != str(evidence["parent_sha256"]):
        raise ApolloStepChartSourceError("magazine evidence hash drifted")


def safe_zip_members(
    archive: zipfile.ZipFile,
    *,
    maximum_member_count: int,
    maximum_uncompressed_bytes: int,
) -> list[dict[str, Any]]:
    infos = archive.infolist()
    if not infos or len(infos) > maximum_member_count:
        raise ApolloStepChartSourceError("ZIP member count is outside the bound")
    output: list[dict[str, Any]] = []
    total = 0
    for info in infos:
        posix = PurePosixPath(info.filename.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts or not posix.parts:
            raise ApolloStepChartSourceError("ZIP contains an unsafe member path")
        if info.flag_bits & 0x1:
            raise ApolloStepChartSourceError("encrypted ZIP members are forbidden")
        total += int(info.file_size)
        if total > maximum_uncompressed_bytes:
            raise ApolloStepChartSourceError("ZIP exceeds uncompressed byte bound")
        output.append(
            {
                "path": posix.as_posix(),
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
                "crc32": f"{int(info.CRC):08x}",
                "is_directory": info.is_dir(),
            }
        )
    bad_member = archive.testzip()
    if bad_member is not None:
        raise ApolloStepChartSourceError(f"ZIP CRC failed: {bad_member}")
    return output


def inspect_archive(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    acquisition = config["acquisition"]
    size = path.stat().st_size
    if size <= 0 or size > int(acquisition["maximum_compressed_bytes"]):
        raise ApolloStepChartSourceError("archive size is outside the bound")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = safe_zip_members(
                archive,
                maximum_member_count=int(acquisition["maximum_member_count"]),
                maximum_uncompressed_bytes=int(
                    acquisition["maximum_uncompressed_bytes"]
                ),
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ApolloStepChartSourceError("archive is not a valid ZIP") from exc
    return {
        "schema": MANIFEST_SCHEMA,
        "source_url": str(config["source"]["download_url"]),
        "source_filename": str(config["source"]["download_filename"]),
        "path": str(config["acquisition"]["destination"]),
        "compressed_bytes": size,
        "sha256": sha256_file(path),
        "members": members,
        "member_count": len(members),
        "total_uncompressed_bytes": sum(
            int(row["uncompressed_bytes"]) for row in members
        ),
        "zip_crc_pass": True,
        "raw_scan_identity_status": "pending_member_audit",
        "analysis_allowed": False,
        "claim_ceiling": str(config["claim_ceiling"]),
    }


def write_stream_bounded(
    blocks: Iterable[bytes],
    destination: Path,
    *,
    maximum_bytes: int,
) -> int:
    written = 0
    with destination.open("xb") as handle:
        for block in blocks:
            if not block:
                continue
            written += len(block)
            if written > maximum_bytes:
                raise ApolloStepChartSourceError(
                    "download exceeded compressed byte bound"
                )
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if written <= 0:
        raise ApolloStepChartSourceError("download was empty")
    return written


def acquire_archive(
    root: Path,
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download once to an owned partial, validate, then publish atomically."""

    validate_config(config)
    validate_parent_evidence(root, config)
    acquisition = config["acquisition"]
    destination = root / str(acquisition["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return inspect_archive(destination, config)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    client = session or requests.Session()
    response = None
    try:
        response = client.get(
            str(config["source"]["download_url"]),
            stream=True,
            timeout=int(acquisition["request_timeout_seconds"]),
            allow_redirects=True,
            headers={"User-Agent": "neuro-film-u6-p2k/1.0"},
        )
        response.raise_for_status()
        final = urlparse(str(response.url))
        if (
            final.scheme != "https"
            or final.hostname != config["source"]["allowed_download_host"]
        ):
            raise ApolloStepChartSourceError("download redirected off frozen host")
        content_type = str(response.headers.get("Content-Type", "")).split(";")[0]
        disposition = str(response.headers.get("Content-Disposition", ""))
        if content_type != "application/zip":
            raise ApolloStepChartSourceError("download content type is not ZIP")
        if str(config["source"]["download_filename"]) not in disposition:
            raise ApolloStepChartSourceError("download disposition drifted")
        write_stream_bounded(
            response.iter_content(chunk_size=int(acquisition["chunk_bytes"])),
            partial,
            maximum_bytes=int(acquisition["maximum_compressed_bytes"]),
        )
        manifest = inspect_archive(partial, config)
        partial.replace(destination)
        manifest["path"] = str(acquisition["destination"])
        return manifest
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    finally:
        if response is not None:
            response.close()
        if session is None:
            client.close()


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


__all__ = [
    "ApolloStepChartSourceError",
    "MANIFEST_SCHEMA",
    "SCHEMA",
    "acquire_archive",
    "inspect_archive",
    "safe_zip_members",
    "sha256_file",
    "validate_config",
    "validate_parent_evidence",
    "write_stream_bounded",
    "write_manifest",
]
