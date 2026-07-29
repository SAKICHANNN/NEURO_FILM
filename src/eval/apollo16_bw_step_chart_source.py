"""Bounded acquisition audit for the Apollo 16 magazine 111/J B&W wedge."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests

from src.eval.apollo_step_chart_source import (
    safe_zip_members,
    sha256_file,
    write_manifest,
    write_stream_bounded,
)


SCHEMA = "neuro_film.u6_p2l_apollo16_bw_step_chart_source.v1"
MANIFEST_SCHEMA = "neuro_film.u6_p2l_apollo16_bw_step_chart_acquisition.v1"


class Apollo16BWStepChartSourceError(RuntimeError):
    """Raised when the frozen P2L source contract fails closed."""


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise Apollo16BWStepChartSourceError("unsupported P2L source schema")
    source = config.get("source")
    acquisition = config.get("acquisition")
    evidence = config.get("magazine_evidence")
    if not all(isinstance(row, Mapping) for row in (source, acquisition, evidence)):
        raise Apollo16BWStepChartSourceError("source contract is incomplete")
    parsed = urlparse(str(source["download_url"]))
    if (
        parsed.scheme != "https"
        or parsed.hostname != source["allowed_download_host"]
        or parsed.path != "/download_step"
        or parsed.query != "step_name=AS16-111-stepchart05"
        or source["download_filename"] != "AS16-111-stepchart05.zip"
    ):
        raise Apollo16BWStepChartSourceError("download identity drifted")
    if (
        evidence.get("mission") != "AS16"
        or evidence.get("photo_roll") != "111"
        or evidence.get("physical_magazine") != "J"
        or evidence.get("film_code") != "3401"
        or evidence.get("film_stock_id") != "kodak_3401_plus_x"
    ):
        raise Apollo16BWStepChartSourceError("magazine evidence drifted")
    if acquisition.get("resume_allowed") is not False:
        raise Apollo16BWStepChartSourceError("range resume must remain disabled")
    for key in (
        "maximum_compressed_bytes",
        "maximum_uncompressed_bytes",
        "maximum_member_count",
        "request_timeout_seconds",
        "chunk_bytes",
    ):
        if not isinstance(acquisition.get(key), int) or acquisition[key] <= 0:
            raise Apollo16BWStepChartSourceError(f"{key} must be positive")
    if any(
        config.get(key) is not False
        for key in (
            "training_allowed",
            "operator_fitting_allowed",
            "stock_calibration_allowed",
        )
    ):
        raise Apollo16BWStepChartSourceError("learning/calibration must be closed")


def inspect_archive(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    acquisition = config["acquisition"]
    size = path.stat().st_size
    if size <= 0 or size > int(acquisition["maximum_compressed_bytes"]):
        raise Apollo16BWStepChartSourceError("archive size is outside the bound")
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
        raise Apollo16BWStepChartSourceError("archive is not a valid ZIP") from exc
    return {
        "schema": MANIFEST_SCHEMA,
        "source_url": str(config["source"]["download_url"]),
        "source_filename": str(config["source"]["download_filename"]),
        "path": str(acquisition["destination"]),
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


def acquire_archive(
    root: Path,
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download once to an owned partial, audit, then publish atomically."""

    validate_config(config)
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
            headers={"User-Agent": "neuro-film-u6-p2l/1.0"},
        )
        response.raise_for_status()
        final = urlparse(str(response.url))
        if (
            final.scheme != "https"
            or final.hostname != config["source"]["allowed_download_host"]
        ):
            raise Apollo16BWStepChartSourceError("download redirected off source")
        content_type = str(response.headers.get("Content-Type", "")).split(";")[0]
        disposition = str(response.headers.get("Content-Disposition", ""))
        if content_type != "application/zip":
            raise Apollo16BWStepChartSourceError("content type is not ZIP")
        if str(config["source"]["download_filename"]) not in disposition:
            raise Apollo16BWStepChartSourceError("content disposition drifted")
        write_stream_bounded(
            response.iter_content(chunk_size=int(acquisition["chunk_bytes"])),
            partial,
            maximum_bytes=int(acquisition["maximum_compressed_bytes"]),
        )
        manifest = inspect_archive(partial, config)
        partial.replace(destination)
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


__all__ = [
    "Apollo16BWStepChartSourceError",
    "acquire_archive",
    "inspect_archive",
    "validate_config",
    "write_manifest",
]
