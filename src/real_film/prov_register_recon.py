"""Bounded metadata-only PROV negative-register source reconnaissance."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any


class ProvRegisterReconError(ValueError):
    """Raised when the PROV catalogue response violates the frozen contract."""


def normalize_solr_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    _remove_qtime(normalized)
    return normalized


def _remove_qtime(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "responseHeader" and isinstance(child, dict):
                child.pop("QTime", None)
            _remove_qtime(child)
    elif isinstance(value, list):
        for child in value:
            _remove_qtime(child)


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            normalize_solr_response(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def summarize_catalogues(
    register_payload: Mapping[str, Any],
    collection_payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    register_response = register_payload.get("response")
    collection_response = collection_payload.get("response")
    if not isinstance(register_response, Mapping) or not isinstance(collection_response, Mapping):
        raise ProvRegisterReconError("PROV response is missing a Solr response object")
    register_count = int(register_response.get("numFound", -1))
    collection_count = int(collection_response.get("numFound", -1))
    bounds = config["expected_discovery_bounds"]
    if not int(bounds["negative_register_minimum_items"]) <= register_count <= int(
        bounds["negative_register_maximum_items"]
    ):
        raise ProvRegisterReconError("negative-register item count is outside the frozen bound")
    if collection_count < int(bounds["digitised_collection_minimum_items"]):
        raise ProvRegisterReconError("digitised-collection item count is below the frozen bound")
    docs = register_response.get("docs")
    if not isinstance(docs, list) or len(docs) != register_count:
        raise ProvRegisterReconError("negative-register query did not return every bounded item")

    formats = Counter(str(row.get("format", "unknown")) for row in docs)
    consignments = Counter(str(row.get("consignment_id", "unknown")) for row in docs)
    digital_fields: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for row in docs:
        keys = sorted(
            key
            for key in row
            if "iiif" in key.casefold()
            or "digital" in key.casefold()
            or "veo" in key.casefold()
        )
        if keys or str(row.get("format", "")).casefold() in {"digital", "digital and physical"}:
            digital_fields.append(
                {
                    "identifier": row.get("identifier.PROV_ACM.id"),
                    "fields": keys,
                    "format": row.get("format"),
                }
            )
        items.append(
            {
                "identifier": row.get("identifier.PROV_ACM.id"),
                "title": row.get("title"),
                "consignment_id": row.get("consignment_id"),
                "format": row.get("format"),
                "negative_number_range": row.get("description.aggregate"),
                "rights_statement": row.get("rights_statement"),
                "rights_status": row.get("rights_status"),
                "location": row.get("location"),
            }
        )

    facets = collection_payload.get("facet_counts", {})
    format_values = facets.get("facet_fields", {}).get("format", [])
    if not isinstance(format_values, list) or len(format_values) % 2:
        raise ProvRegisterReconError("digitised-collection format facet is invalid")
    collection_formats = {
        str(format_values[index]): int(format_values[index + 1])
        for index in range(0, len(format_values), 2)
    }
    register_machine_accessible = bool(digital_fields)
    return {
        "negative_register": {
            "series_id": 17690,
            "item_count": register_count,
            "format_counts": dict(sorted(formats.items())),
            "consignment_counts": dict(sorted(consignments.items())),
            "digital_or_iiif_records": digital_fields,
            "items": items,
        },
        "digitised_negative_collection": {
            "series_id": 17684,
            "item_count": collection_count,
            "format_counts": collection_formats,
        },
        "register_contents_machine_accessible": register_machine_accessible,
        "decision": (
            "open_separate_bounded_register_content_join_audit"
            if register_machine_accessible
            else "closed_register_catalogue_is_physical_only"
        ),
    }
