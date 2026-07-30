"""Bounded offline audit for the Color Precision chart-comparison archive."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "neuro-film.u5-r2bd0-color-precision-chart-source-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2bd0-color-precision-chart-source-report.v1"


class ColorPrecisionChartSourceError(RuntimeError):
    """Raised when the frozen source or archive contract fails closed."""


_STOCK_IDS = {
    "portra 160": "kodak_portra_160",
    "portra 400": "kodak_portra_400",
    "portra 800": "kodak_portra_800",
    "gold": "kodak_gold",
    "ektar": "kodak_ektar",
    "ektachrome": "kodak_ektachrome",
    "ultramax": "kodak_ultramax",
    "colorplus": "kodak_colorplus",
    "pro image": "kodak_pro_image",
}


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ColorPrecisionChartSourceError("unsupported source schema")
    if config.get("status") != "contract_frozen_before_payload_access":
        raise ColorPrecisionChartSourceError("source contract is not frozen")
    source = config.get("source")
    acquisition = config.get("acquisition")
    if not isinstance(source, Mapping) or not isinstance(acquisition, Mapping):
        raise ColorPrecisionChartSourceError("source/acquisition contract missing")
    if (
        source.get("landing_page")
        != "https://www.colorprecision.com/pages/film-comparison-tool"
        or source.get("terms_page")
        != "https://www.colorprecision.com/pages/terms"
        or source.get("chart_download_google_drive_id")
        != "12U8xfTge_k5yEzovnU_GELBAc7KpVrD9"
        or source.get("rights_status")
        != "unknown_all_rights_reserved_no_training_or_redistribution_grant"
    ):
        raise ColorPrecisionChartSourceError("source or rights boundary drift")
    for key in (
        "maximum_download_bytes",
        "maximum_files_after_extraction",
        "maximum_extracted_bytes",
    ):
        value = acquisition.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ColorPrecisionChartSourceError(f"{key} must be positive")
    if (
        acquisition.get("purchase_allowed") is not False
        or acquisition.get("authentication_allowed") is not False
        or acquisition.get("other_download_categories_allowed") is not False
        or acquisition.get("emulation_products_or_profiles_allowed") is not False
    ):
        raise ColorPrecisionChartSourceError("acquisition boundary drift")
    forbidden = set(config.get("forbidden_actions", ()))
    if not {"operator_fitting", "training", "redistribution"} <= forbidden:
        raise ColorPrecisionChartSourceError("forbidden-action boundary drift")


def _safe_members(
    archive: zipfile.ZipFile,
    *,
    maximum_member_count: int,
    maximum_uncompressed_bytes: int,
) -> list[dict[str, Any]]:
    infos = archive.infolist()
    if not infos or len(infos) > maximum_member_count:
        raise ColorPrecisionChartSourceError(
            "ZIP member count is outside the frozen bound"
        )
    seen: set[str] = set()
    total = 0
    rows: list[dict[str, Any]] = []
    for info in infos:
        posix = PurePosixPath(info.filename.replace("\\", "/"))
        normalized = posix.as_posix()
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or not posix.parts
            or normalized in seen
        ):
            raise ColorPrecisionChartSourceError(
                "ZIP contains an unsafe or duplicate member path"
            )
        seen.add(normalized)
        if info.flag_bits & 0x1:
            raise ColorPrecisionChartSourceError(
                "encrypted ZIP members are forbidden"
            )
        total += int(info.file_size)
        if total > maximum_uncompressed_bytes:
            raise ColorPrecisionChartSourceError(
                "ZIP exceeds the frozen uncompressed-byte bound"
            )
        rows.append(
            {
                "path": normalized,
                "extension": posix.suffix.lower(),
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
                "crc32": f"{int(info.CRC):08x}",
                "is_directory": bool(info.is_dir()),
            }
        )
    return rows


def inspect_archive(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash and inventory the archive without extracting or rendering pixels."""

    validate_contract(config)
    acquisition = config["acquisition"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ColorPrecisionChartSourceError("archive is unavailable") from exc
    if size <= 0 or size > int(acquisition["maximum_download_bytes"]):
        raise ColorPrecisionChartSourceError(
            "archive size is outside the frozen download bound"
        )
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = _safe_members(
                archive,
                maximum_member_count=int(
                    acquisition["maximum_files_after_extraction"]
                ),
                maximum_uncompressed_bytes=int(
                    acquisition["maximum_extracted_bytes"]
                ),
            )
            bad_member = archive.testzip()
    except (OSError, zipfile.BadZipFile) as exc:
        raise ColorPrecisionChartSourceError(
            "archive is not a valid readable ZIP"
        ) from exc
    if bad_member is not None:
        raise ColorPrecisionChartSourceError(
            f"ZIP CRC failed for {bad_member}"
        )
    extension_counts = Counter(
        row["extension"] or "<none>"
        for row in members
        if not row["is_directory"]
    )
    stable = {
        "archive_bytes": int(size),
        "archive_sha256": sha256_file(path),
        "member_count": len(members),
        "file_count": sum(not row["is_directory"] for row in members),
        "total_uncompressed_bytes": sum(
            int(row["uncompressed_bytes"]) for row in members
        ),
        "extension_counts": dict(sorted(extension_counts.items())),
        "members": members,
        "zip_crc_pass": True,
    }
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        **stable,
        "stable_evidence_id": canonical_sha256(stable),
        "pixel_rendering_performed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "connectivity_decision": "pending_member_semantic_audit",
        "claim_ceiling": config["claim_ceiling"],
    }


def parse_pdf_text_pages(text: str, *, scanner_id: str) -> list[dict[str, Any]]:
    """Parse page headings emitted by pdftotext without inspecting page pixels."""

    if scanner_id not in {"frontier", "noritsu"}:
        raise ColorPrecisionChartSourceError("unsupported scanner identity")
    rows: list[dict[str, Any]] = []
    for page_index, raw_page in enumerate(text.split("\f"), start=1):
        lines = [" ".join(line.split()) for line in raw_page.splitlines()]
        lines = [line for line in lines if line and line != "Color Precision"]
        if not lines or lines[0].lower() == "lighting diagram":
            continue
        exposure_index = next(
            (
                index
                for index, line in enumerate(lines)
                if re.match(r"(?i)^ev\s", line)
            ),
            None,
        )
        if exposure_index is None or exposure_index == 0:
            raise ColorPrecisionChartSourceError(
                f"page {page_index} lacks a stock/exposure heading"
            )
        title = " ".join(lines[:exposure_index]).lower()
        exposure_line = lines[exposure_index]
        normalized_title = title
        process_variant = "normal"
        if normalized_title.endswith(" pushed +1"):
            process_variant = "pushed_plus_1"
            normalized_title = normalized_title.removesuffix(" pushed +1")
        elif normalized_title.endswith(" cross processed"):
            process_variant = "cross_processed"
            normalized_title = normalized_title.removesuffix(
                " cross processed"
            )
        stock_id = _STOCK_IDS.get(normalized_title)
        if stock_id is None:
            raise ColorPrecisionChartSourceError(
                f"unknown stock heading: {normalized_title}"
            )
        exposures = tuple(
            int(value)
            for value in re.findall(r"(?<![A-Za-z])([+-]?\d+)", exposure_line)
        )
        if len(exposures) != 7 or len(exposures) != len(set(exposures)):
            raise ColorPrecisionChartSourceError(
                f"page {page_index} exposure sequence drift"
            )
        remaining = " ".join(lines[exposure_index:]).lower()
        if scanner_id not in remaining:
            raise ColorPrecisionChartSourceError(
                f"page {page_index} scanner heading drift"
            )
        rows.append(
            {
                "page_index": page_index,
                "stock_id": stock_id,
                "display_label": title,
                "process_variant": process_variant,
                "scanner_id": scanner_id,
                "exposure_offsets_ev": list(exposures),
                "illuminant_text_present": "tungsten" in remaining,
                "chart_group": "color_precision_charts_fixed_lighting_setup",
            }
        )
    if not rows:
        raise ColorPrecisionChartSourceError("PDF contains no chart pages")
    return rows


def parse_pdfimages_list(text: str, *, scanner_id: str) -> list[dict[str, Any]]:
    """Parse Poppler's object listing and retain only full chart photographs."""

    if scanner_id not in {"frontier", "noritsu"}:
        raise ColorPrecisionChartSourceError("unsupported scanner identity")
    by_page: dict[int, list[tuple[int, int]]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 16 or not fields[0].isdigit():
            continue
        page = int(fields[0])
        object_type = fields[2]
        width = int(fields[3])
        height = int(fields[4])
        object_id = int(fields[10])
        generation = int(fields[11])
        if object_type != "image" or width < 2500 or height < 1800:
            continue
        by_page.setdefault(page, []).append((object_id, generation))
    if not by_page:
        raise ColorPrecisionChartSourceError(
            "pdfimages listing contains no chart photographs"
        )
    return [
        {
            "scanner_id": scanner_id,
            "page_index": page,
            "photo_placement_count": len(objects),
            "unique_photo_object_count": len(set(objects)),
            "duplicate_photo_placement_count": len(objects)
            - len(set(objects)),
        }
        for page, objects in sorted(by_page.items())
    ]


def summarize_connectivity(
    rows: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate filename/text connectivity while preserving the rights stop."""

    validate_contract(config)
    if not rows:
        raise ColorPrecisionChartSourceError("connectivity rows are empty")
    identities = [
        (row["stock_id"], row["process_variant"], row["scanner_id"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ColorPrecisionChartSourceError("duplicate chart-page identity")
    scanners = sorted({str(row["scanner_id"]) for row in rows})
    stocks = sorted({str(row["stock_id"]) for row in rows})
    exposures = sorted(
        {
            int(value)
            for row in rows
            for value in row["exposure_offsets_ev"]
        }
    )
    variants = sorted(
        {
            (str(row["stock_id"]), str(row["process_variant"]))
            for row in rows
        }
    )
    variant_scanner_support = {
        f"{stock_id}/{process_variant}": sorted(
            {
                str(row["scanner_id"])
                for row in rows
                if row["stock_id"] == stock_id
                and row["process_variant"] == process_variant
            }
        )
        for stock_id, process_variant in variants
    }
    fully_scanner_connected_variants = sum(
        len(value) == len(scanners)
        for value in variant_scanner_support.values()
    )
    largest_component_rows = len(rows)
    gates = config["data_readiness_gates"]
    checks = {
        "named_stocks": len(stocks)
        >= int(gates["minimum_distinct_named_stock_labels"]),
        "scanners": len(scanners)
        >= int(gates["minimum_distinct_scanner_labels"]),
        "exposures": len(exposures)
        >= int(gates["minimum_distinct_exposure_labels"]),
        "same_chart_component": largest_component_rows
        >= int(gates["minimum_rows_in_largest_connected_component"]),
        "all_variants_cross_scanner": fully_scanner_connected_variants
        == len(variants),
    }
    metrics = {
        "chart_page_count": len(rows),
        "exposure_cell_count": sum(
            len(row["exposure_offsets_ev"]) for row in rows
        ),
        "distinct_named_stock_count": len(stocks),
        "distinct_scanner_count": len(scanners),
        "distinct_exposure_offset_count": len(exposures),
        "stock_process_variant_count": len(variants),
        "cross_scanner_variant_count": fully_scanner_connected_variants,
        "largest_same_chart_component_rows": largest_component_rows,
        "named_stock_ids": stocks,
        "scanner_ids": scanners,
        "exposure_offsets_ev": exposures,
        "variant_scanner_support": variant_scanner_support,
    }
    passed = all(checks.values())
    stable = {"metrics": metrics, "checks": checks, "rows": rows}
    return {
        "schema": (
            "neuro-film.u5-r2bd0-color-precision-chart-connectivity-report.v1"
        ),
        "experiment_id": config["experiment_id"],
        **stable,
        "stable_evidence_id": canonical_sha256(stable),
        "connectivity_passed": passed,
        "decision": (
            "pass_structure_rights_block_pixel_fit"
            if passed
            else "close_insufficient_connectivity"
        ),
        "rights_status": config["source"]["rights_status"],
        "pixel_analysis_allowed": (
            "internal_stress_or_nuisance_audit_only" if passed else "none"
        ),
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_clustering_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "ColorPrecisionChartSourceError",
    "canonical_sha256",
    "inspect_archive",
    "parse_pdfimages_list",
    "parse_pdf_text_pages",
    "sha256_file",
    "summarize_connectivity",
    "validate_contract",
]
