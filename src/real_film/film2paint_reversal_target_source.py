"""Zero-pixel source gate for the FILM2PAINT reversal-target dataset."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pypdf import PdfReader

BinaryReader = Callable[[str, int], bytes]
ArticleTextReader = Callable[[bytes], str]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def _read_bounded(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "neuro-film-source-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}: {url}")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError(f"response exceeds frozen byte ceiling: {url}")
    return payload


def _extract_pdf_text(payload: bytes) -> str:
    reader = PdfReader(io.BytesIO(payload))
    return " ".join(" ".join(page.extract_text() or "" for page in reader.pages).split())


def _json(payload: bytes, role: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{role} must be a JSON object")
    return value


def _metadata_values(item: dict[str, Any], key: str) -> list[str]:
    rows = item.get("metadata", {}).get(key, [])
    if not isinstance(rows, list):
        return []
    return [str(row.get("value", "")) for row in rows if isinstance(row, dict)]


def _request_specs(config: dict[str, Any]) -> list[tuple[str, str, int]]:
    repository = config["repository"]
    base = repository["base_url"].rstrip("/")
    item_uuid = repository["item_uuid"]
    specs = [
        ("article", config["article"]["url"], 3_000_000),
        ("item", f"{base}/items/{item_uuid}", 100_000),
        ("bundles", f"{base}/items/{item_uuid}/bundles?size=100", 100_000),
    ]
    for row in repository["expected_bundles"]:
        specs.append(
            (
                f"bitstreams:{row['name']}",
                f"{base}/bundles/{row['uuid']}/bitstreams?size=100",
                100_000,
            )
        )
    return specs


def _bundle_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = payload.get("_embedded", {}).get("bundles", [])
    return sorted(
        ({"name": str(row["name"]), "uuid": str(row["uuid"])} for row in rows),
        key=lambda row: (row["name"], row["uuid"]),
    )


def _bitstream_rows(
    payloads: dict[str, bytes], expected_bundles: Sequence[dict[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in expected_bundles:
        role = f"bitstreams:{bundle['name']}"
        payload = _json(payloads[role], role)
        bitstreams = payload.get("_embedded", {}).get("bitstreams", [])
        for row in bitstreams:
            checksum = row.get("checkSum", {})
            rows.append(
                {
                    "bundle": bundle["name"],
                    "name": str(row["name"]),
                    "uuid": str(row["uuid"]),
                    "bytes": int(row["sizeBytes"]),
                    "checksum_algorithm": str(checksum["checkSumAlgorithm"]),
                    "checksum": str(checksum["value"]),
                }
            )
    return sorted(rows, key=lambda row: (row["bundle"], row["name"], row["uuid"]))


def _contains_any(text: str, fragments: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(fragment.casefold() in lowered for fragment in fragments)


def run_source_audit(
    config_path: Path,
    *,
    binary_reader: BinaryReader = _read_bounded,
    article_text_reader: ArticleTextReader = _extract_pdf_text,
    reverse_request_order: bool = False,
) -> dict[str, Any]:
    """Run the frozen source gate without requesting any dataset bitstream."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    specs = _request_specs(config)
    execution_specs = list(reversed(specs)) if reverse_request_order else specs
    payloads = {
        role: binary_reader(url, max_bytes)
        for role, url, max_bytes in execution_specs
    }

    article_payload = payloads["article"]
    article_text = article_text_reader(article_payload)
    article_phrase_results = {
        fragment: fragment in article_text
        for fragment in config["article"]["required_text_fragments"]
    }
    article_identity_matches = (
        len(article_payload) == config["article"]["bytes"]
        and _sha256(article_payload) == config["article"]["sha256"]
    )

    item = _json(payloads["item"], "item")
    bundles = _bundle_rows(_json(payloads["bundles"], "bundles"))
    expected_bundles = sorted(
        config["repository"]["expected_bundles"],
        key=lambda row: (row["name"], row["uuid"]),
    )
    bitstreams = _bitstream_rows(
        payloads, config["repository"]["expected_bundles"]
    )
    expected_bitstreams = sorted(
        config["repository"]["expected_bitstreams"],
        key=lambda row: (row["bundle"], row["name"], row["uuid"]),
    )
    item_identity_matches = (
        str(item.get("uuid")) == config["repository"]["item_uuid"]
        and config["repository"]["doi"]
        in _metadata_values(item, "dc.identifier.doi")
    )
    repository_inventory_matches = (
        item_identity_matches
        and bundles == expected_bundles
        and bitstreams == expected_bitstreams
    )

    markers = config["admission_markers"]
    dataset_rows = [
        row
        for row in bitstreams
        if any(row["name"].casefold().endswith(ext) for ext in markers["dataset_extensions"])
    ]
    repository_semantic_text = json.dumps(
        {"item_metadata": item.get("metadata", {}), "bitstreams": bitstreams},
        sort_keys=True,
        ensure_ascii=True,
    )
    manifest_present = bool(dataset_rows) and any(
        _contains_any(row["name"], markers["manifest_name_fragments"])
        for row in bitstreams
    )
    dataset_licence_present = bool(dataset_rows) and _contains_any(
        repository_semantic_text, markers["dataset_licence_fragments"]
    )
    group_roles_present = bool(dataset_rows) and all(
        fragment.casefold() in repository_semantic_text.casefold()
        for fragment in markers["group_role_fragments"]
    )
    zero_keys = (
        "repository_bitstream_content_requests",
        "film_target_scan_requests",
        "image_derivative_requests",
        "account_or_copy_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    )
    gates = {
        "official_article_identity_and_method_facts_match": article_identity_matches
        and all(article_phrase_results.values()),
        "official_repository_inventory_matches": repository_inventory_matches,
        "public_film_target_dataset_payload_present": bool(dataset_rows),
        "dataset_specific_fitting_and_derived_output_licence_present": dataset_licence_present,
        "exact_dataset_manifest_and_checksums_present": manifest_present,
        "independent_groups_and_confirmation_role_present": group_roles_present,
        "zero_forbidden_media_pixel_fit_render_score": all(
            config["operation_limits"][key] == 0 for key in zero_keys
        ),
    }
    passed = all(gates.values())
    source_identity = {
        "article_bytes": len(article_payload),
        "article_sha256": _sha256(article_payload),
        "article_required_text_results": article_phrase_results,
        "item_uuid": str(item.get("uuid")),
        "item_doi_values": _metadata_values(item, "dc.identifier.doi"),
        "bundles": bundles,
        "bitstreams": bitstreams,
    }
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3p-film2paint-reversal-target-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "source_identity": source_identity,
        "source_identity_sha256": _canonical_sha256(source_identity),
        "admission": {
            "target_stock": config["target_stock"],
            "dataset_bitstreams": dataset_rows,
            "document_bitstream_licence": config["repository"][
                "document_bitstream_licence"
            ],
            "dataset_licence_is_independently_present": dataset_licence_present,
            "required_dataset_properties": config["required_dataset_properties"],
        },
        "gates": gates,
        "operation_counts": config["operation_limits"],
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_payload = {
        key: report[key]
        for key in (
            "schema",
            "experiment_id",
            "decision",
            "source_identity_sha256",
            "admission",
            "gates",
            "operation_counts",
            "claim_ceiling",
        )
    }
    report["stable_evidence_id"] = _canonical_sha256(stable_payload)
    return report
