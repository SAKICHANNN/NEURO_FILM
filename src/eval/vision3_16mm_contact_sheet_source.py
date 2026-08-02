"""U5.R2BU6 exact contact-sheet source and role audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader

CONTRACT_SCHEMA = (
    "neuro_film.u5_r2bu6_vision3_16mm_contact_sheet_source_contract.v1"
)
OBSERVATIONS_SCHEMA = (
    "neuro_film.u5_r2bu6_vision3_16mm_contact_sheet_observations.v1"
)
REPORT_SCHEMA = "neuro_film.u5_r2bu6_vision3_16mm_contact_sheet_source_report.v1"


class Vision3ContactSheetSourceError(ValueError):
    """Raised when the frozen BU6 source contract or contact sheet drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise Vision3ContactSheetSourceError("BU6 source path must be relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    contact = payload.get("public_drive", {}).get("contact_sheet", {})
    if (
        payload.get("schema") != CONTRACT_SCHEMA
        or payload.get("experiment_id") != "U5.R2BU6"
        or payload.get("public_drive", {}).get("folder_id")
        != "1ebwiJ1GTqapP0LT8p1opOlxl2vE5c9iG"
        or contact.get("file_id") != "19X7RMbwReubb4wD6lh4Gh9qbAJ8Hs46z"
        or contact.get("http_content_length") != 121_524_686
        or contact.get("maximum_download_bytes") != 130_000_000
        or payload.get("rights", {}).get("explicit_reuse_license_found") is not False
        or payload.get("rights", {}).get("redistribution") is not False
        or payload.get("gates")
        != {
            "download_only_contact_sheet": True,
            "maximum_download_bytes": 130_000_000,
            "pdf_decode_required": True,
            "minimum_stock_ids": 2,
            "required_stock_ids": ["7207", "7219"],
            "minimum_distinct_condition_labels": 4,
            "paired_scene_roles_must_be_explicit": True,
            "scanner_and_process_must_remain_unknown_if_unstated": True,
            "image_payload_downloads": 0,
            "video_payload_downloads": 0,
        }
    ):
        raise Vision3ContactSheetSourceError("BU6 frozen contract drift")
    _relative_path(str(contact.get("destination", "")))
    return payload


def load_observations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sheet = payload.get("contact_sheet", {})
    anchors = payload.get("anchors", [])
    if (
        payload.get("schema") != OBSERVATIONS_SCHEMA
        or payload.get("experiment_id") != "U5.R2BU6"
        or sheet.get("sha256")
        != "7ba0d96372ec35f04f2d1ed13c4410a33a5050fc1b17471fc576b47e0a5abd97"
        or sheet.get("bytes") != 121_524_686
        or sheet.get("page_image_counts") != [15, 15, 15, 7]
        or sheet.get("embedded_image_count") != 52
        or len(anchors) != len({row.get("index") for row in anchors})
        or payload.get("scanner_id") is not None
        or payload.get("process_id") is not None
        or payload.get("external_image_payload_downloads") != 0
        or payload.get("external_video_payload_downloads") != 0
    ):
        raise Vision3ContactSheetSourceError("BU6 observation binding drift")
    return payload


def _decode_contact_sheet(
    path: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, str]]:
    if (
        not path.is_file()
        or path.stat().st_size != int(expected["bytes"])
        or sha256_file(path) != expected["sha256"]
    ):
        raise Vision3ContactSheetSourceError("BU6 contact-sheet integrity mismatch")
    reader = PdfReader(path)
    metadata = reader.metadata

    def metadata_text(key: str) -> str:
        if metadata is None:
            return ""
        value = metadata.raw_get(key)
        if hasattr(value, "get_object"):
            value = value.get_object()
        return str(value or "")
    page_counts = [len(page.images) for page in reader.pages]
    image_hashes: dict[int, str] = {}
    widths: set[int] = set()
    heights: set[int] = set()
    modes: set[str] = set()
    index = 0
    for page in reader.pages:
        ordered_images = sorted(
            page.images,
            key=lambda image: int(image.indirect_reference.idnum),
        )
        for embedded in ordered_images:
            image_object = embedded.indirect_reference.get_object()
            width = int(image_object["/Width"])
            height = int(image_object["/Height"])
            bits = int(image_object["/BitsPerComponent"])
            decoded = image_object.get_data()
            if bits != 16 or len(decoded) != width * height * 3 * 2:
                raise Vision3ContactSheetSourceError(
                    "BU6 embedded image is not packed 16-bit RGB"
                )
            # pypdf's PIL convenience conversion treats these ICCBased 16-bit
            # samples as byte-interleaved RGB. Decode the PDF samples directly.
            samples = np.frombuffer(decoded, dtype=">u2").reshape(height, width, 3)
            widths.add(width)
            heights.add(height)
            modes.add("RGB16BE")
            image_hashes[index] = hashlib.sha256(samples.tobytes()).hexdigest()
            index += 1
    facts = {
        "page_count": len(reader.pages),
        "page_image_counts": page_counts,
        "embedded_image_count": index,
        "embedded_image_widths": sorted(widths),
        "embedded_image_heights": sorted(heights),
        "decoded_modes": sorted(modes),
        "metadata_title": metadata_text("/Title"),
        "metadata_author": metadata_text("/Author"),
    }
    return facts, image_hashes


def evaluate_contact_sheet_source(
    root: Path, contract: dict[str, Any], observations: dict[str, Any]
) -> dict[str, Any]:
    contact = contract["public_drive"]["contact_sheet"]
    expected = observations["contact_sheet"]
    pdf_path = root / _relative_path(contact["destination"])
    pdf_facts, image_hashes = _decode_contact_sheet(pdf_path, expected)
    anchors = observations["anchors"]
    indexed = {int(row["index"]): row for row in anchors}
    anchor_indices = sorted(indexed)
    if not anchor_indices or anchor_indices[-1] >= pdf_facts["embedded_image_count"]:
        raise Vision3ContactSheetSourceError("BU6 anchor index outside contact sheet")
    stock_ids = sorted({str(row["stock_id"]) for row in anchors})
    condition_labels = sorted(
        {str(label) for row in anchors for label in row.get("filtration", [])}
    )
    paired_roles: list[dict[str, Any]] = []
    for pair in observations["paired_roles"]:
        indices = [int(value) for value in pair["indices"]]
        if len(indices) != 2 or any(value not in indexed for value in indices):
            raise Vision3ContactSheetSourceError("BU6 paired-role index drift")
        rows = [indexed[value] for value in indices]
        explicit_match = (
            {str(row["stock_id"]) for row in rows} == {"7207", "7219"}
            and len({str(row["key_fill"]) for row in rows}) == 1
            and len({str(row["f_stop"]) for row in rows}) == 1
            and len({str(row["exposure"]) for row in rows}) == 1
        )
        paired_roles.append(
            {
                "role_id": pair["role_id"],
                "indices": indices,
                "explicit_label_match": explicit_match,
            }
        )
    expected_pdf_facts = {
        "page_count": expected["page_count"],
        "page_image_counts": expected["page_image_counts"],
        "embedded_image_count": expected["embedded_image_count"],
        "embedded_image_widths": [expected["embedded_image_width"]],
        "embedded_image_heights": [expected["embedded_image_height"]],
        "decoded_modes": ["RGB16BE"],
        "metadata_title": expected["metadata_title"],
        "metadata_author": expected["metadata_author"],
    }
    gates = {
        "download_only_contact_sheet": observations["external_image_payload_downloads"]
        == 0
        and observations["external_video_payload_downloads"] == 0,
        "maximum_download_bytes": pdf_path.stat().st_size
        <= int(contract["gates"]["maximum_download_bytes"]),
        "pdf_decode_required": pdf_facts == expected_pdf_facts,
        "minimum_stock_ids": len(stock_ids)
        >= int(contract["gates"]["minimum_stock_ids"]),
        "required_stock_ids": stock_ids
        == sorted(contract["gates"]["required_stock_ids"]),
        "minimum_distinct_condition_labels": len(condition_labels)
        >= int(contract["gates"]["minimum_distinct_condition_labels"]),
        "paired_scene_roles_must_be_explicit": bool(paired_roles)
        and all(row["explicit_label_match"] for row in paired_roles),
        "scanner_and_process_must_remain_unknown_if_unstated": observations[
            "scanner_id"
        ]
        is None
        and observations["process_id"] is None,
        "image_payload_downloads": observations["external_image_payload_downloads"]
        == int(contract["gates"]["image_payload_downloads"]),
        "video_payload_downloads": observations["external_video_payload_downloads"]
        == int(contract["gates"]["video_payload_downloads"]),
    }
    stable_payload = {
        "experiment_id": contract["experiment_id"],
        "contact_sheet_bytes": pdf_path.stat().st_size,
        "contact_sheet_sha256": expected["sha256"],
        "pdf_facts": pdf_facts,
        "anchor_count": len(anchors),
        "anchor_pixel_sha256": {
            str(index): image_hashes[index] for index in anchor_indices
        },
        "stock_ids": stock_ids,
        "distinct_condition_labels": condition_labels,
        "paired_roles": paired_roles,
        "scanner_id": observations["scanner_id"],
        "process_id": observations["process_id"],
        "external_image_payload_downloads": observations[
            "external_image_payload_downloads"
        ],
        "external_video_payload_downloads": observations[
            "external_video_payload_downloads"
        ],
        "gate_results": gates,
    }
    gate_pass = all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        **stable_payload,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable_payload)).hexdigest(),
        "gate_pass": gate_pass,
        "decision": (
            "open_minimum_sufficient_tiff_integrity_and_role_audit"
            if gate_pass
            else "close_public_vision3_16mm_contact_sheet_source"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
