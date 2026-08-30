"""Zero-pixel Parker Day DADDY PLEASE Ektar 100 source admission."""

from __future__ import annotations

import hashlib
import html
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class DaddyPleaseEktarSourceError(ValueError):
    """Raised when the frozen source or structured metadata fails closed."""


Fetcher = Callable[[str], bytes]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _default_fetcher(url: str) -> bytes:
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": "K-MCFM-source-audit/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if int(response.status) != 200:
                    raise DaddyPleaseEktarSourceError(
                        f"source returned HTTP {response.status}"
                    )
                return response.read()
        except (
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
            urllib.error.URLError,
        ):
            if attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))
    raise AssertionError("unreachable")


def _visible_text(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(html.unescape(text).split())


def _parse_dl_fields(payload: bytes) -> dict[str, str]:
    text = payload.decode("utf-8")
    fields: dict[str, str] = {}
    for match in re.finditer(
        r"<dt>(?P<key>[^<]+)</dt>\s*"
        r"<dd(?:\s[^>]*)?>(?P<value>(?:(?!<dt>).)*?)</dd>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        key = _visible_text(match.group("key"))
        value = _visible_text(match.group("value"))
        if key and key not in fields:
            fields[key] = value
    return fields


def _read_cbor_length(payload: bytes, offset: int, additional: int) -> tuple[int, int]:
    if additional < 24:
        return additional, offset
    width_by_additional = {24: 1, 25: 2, 26: 4, 27: 8}
    width = width_by_additional.get(additional)
    if width is None or offset + width > len(payload):
        raise DaddyPleaseEktarSourceError("unsupported or truncated CBOR length")
    return int.from_bytes(payload[offset : offset + width], "big"), offset + width


def _read_cbor_item(payload: bytes, offset: int) -> tuple[object, int]:
    if offset >= len(payload):
        raise DaddyPleaseEktarSourceError("truncated CBOR item")
    initial = payload[offset]
    offset += 1
    major = initial >> 5
    length, offset = _read_cbor_length(payload, offset, initial & 31)
    if major == 0:
        return length, offset
    if major == 3:
        end = offset + length
        if end > len(payload):
            raise DaddyPleaseEktarSourceError("truncated CBOR text")
        try:
            return payload[offset:end].decode("utf-8"), end
        except UnicodeDecodeError as error:
            raise DaddyPleaseEktarSourceError("invalid CBOR UTF-8") from error
    if major == 5:
        result: dict[str, object] = {}
        for _ in range(length):
            key, offset = _read_cbor_item(payload, offset)
            value, offset = _read_cbor_item(payload, offset)
            if not isinstance(key, str) or key in result:
                raise DaddyPleaseEktarSourceError("invalid CBOR map key")
            result[key] = value
        return result, offset
    raise DaddyPleaseEktarSourceError(f"unsupported CBOR major type {major}")


def _decode_child_metadata(payload: bytes) -> dict[str, str]:
    try:
        encoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DaddyPleaseEktarSourceError("invalid metadata JSON") from error
    if not isinstance(encoded, str):
        raise DaddyPleaseEktarSourceError("metadata API did not return a hex string")
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as error:
        raise DaddyPleaseEktarSourceError("invalid metadata hex") from error
    value, offset = _read_cbor_item(raw, 0)
    if offset != len(raw) or not isinstance(value, dict):
        raise DaddyPleaseEktarSourceError("metadata CBOR is not one complete map")
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise DaddyPleaseEktarSourceError("metadata map must contain strings only")
    return {str(key): str(item) for key, item in value.items()}


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    return _sha256((("\n".join(ids)) + "\n").encode("utf-8"))


def _ranked_sample_ids(ids: Sequence[str], count: int) -> list[str]:
    return sorted(ids, key=lambda item: (_sha256(item.encode("utf-8")), item))[:count]


def _parse_child_page(payload: bytes, parent_id: str, child_id: str) -> dict[str, Any]:
    text = payload.decode("utf-8")
    parent_link = f"/inscription/{parent_id}"
    if parent_link not in text:
        raise DaddyPleaseEktarSourceError(f"child {child_id} lacks the frozen parent")
    fields = _parse_dl_fields(payload)
    try:
        content_length = int(fields["content length"].removesuffix(" bytes"))
        content_type = fields["content type"]
    except (KeyError, ValueError) as error:
        raise DaddyPleaseEktarSourceError(
            "invalid child content declaration"
        ) from error
    return {
        "id": child_id,
        "parent_id": parent_id,
        "content_type": content_type,
        "content_length": content_length,
    }


def run_daddy_please_ektar_source_audit(
    config_path: Path,
    *,
    reverse: bool = False,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Run the frozen parent/index/metadata-only source audit."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    sample_contract = config["sample"]
    fetch = fetcher or _default_fetcher
    base_url = source["base_url"]
    parent_id = source["parent_id"]

    parent_payload = fetch(source["parent_slug_url"])
    parent_fields = _parse_dl_fields(parent_payload)
    required_parent_fields = source["required_parent_fields"]
    parent_field_gates = {
        key: parent_fields.get(key) == value
        for key, value in sorted(required_parent_fields.items())
    }

    page_numbers = list(range(int(source["child_page_count"])))
    request_pages = list(reversed(page_numbers)) if reverse else page_numbers
    pages: dict[int, dict[str, Any]] = {}
    for page in request_pages:
        suffix = "" if page == 0 else f"/{page}"
        payload = fetch(f"{base_url}/r/children/{parent_id}{suffix}")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DaddyPleaseEktarSourceError("invalid child index JSON") from error
        if (
            not isinstance(decoded, dict)
            or decoded.get("page") != page
            or not isinstance(decoded.get("ids"), list)
            or not all(isinstance(item, str) for item in decoded["ids"])
        ):
            raise DaddyPleaseEktarSourceError("invalid child index page")
        pages[page] = decoded

    child_ids = [item for page in page_numbers for item in pages[page]["ids"]]
    if len(child_ids) != len(set(child_ids)):
        raise DaddyPleaseEktarSourceError("duplicate child inscription ID")

    metadata_request_ids = list(reversed(child_ids)) if reverse else child_ids

    def load_metadata(child_id: str) -> tuple[str, dict[str, str]]:
        payload = fetch(f"{base_url}/r/metadata/{child_id}")
        return child_id, _decode_child_metadata(payload)

    with ThreadPoolExecutor(max_workers=int(source["metadata_workers"])) as pool:
        metadata_pairs = list(pool.map(load_metadata, metadata_request_ids))
    metadata_by_id = dict(metadata_pairs)
    required_metadata_keys = set(source["required_child_metadata_keys"])
    model_counts = dict(
        sorted(
            Counter(
                metadata_by_id[child_id].get("MODEL", "") for child_id in child_ids
            ).items()
        )
    )
    metadata_entries = [
        {"id": child_id, "metadata": metadata_by_id[child_id]}
        for child_id in sorted(child_ids)
    ]

    selected_ids = _ranked_sample_ids(child_ids, int(sample_contract["count"]))
    request_sample_ids = list(reversed(selected_ids)) if reverse else selected_ids
    sample_entries = [
        _parse_child_page(
            fetch(f"{base_url}/inscription/{child_id}"), parent_id, child_id
        )
        for child_id in request_sample_ids
    ]
    sample_entries.sort(key=lambda entry: entry["id"])
    sample_total_bytes = sum(int(entry["content_length"]) for entry in sample_entries)

    operation_counts = dict(config["operation_limits"])
    expected_model_count = int(source["expected_model_count"])
    expected_photos_per_model = int(source["expected_photos_per_model"])
    audit_gates = {
        "parent_project_stock_camera_license_fields_exact": all(
            parent_field_gates.values()
        ),
        "child_page_count_and_terminal_flags_exact": all(
            len(pages[page]["ids"]) == int(source["expected_ids_per_page"])
            and bool(pages[page].get("more")) == (page < page_numbers[-1])
            for page in page_numbers
        ),
        "child_count_unique_exact": len(child_ids)
        == int(source["expected_child_count"]),
        "ordered_child_id_sha256_exact": _ordered_ids_sha256(child_ids)
        == source["expected_ordered_child_id_sha256"],
        "child_metadata_keys_and_strings_exact": all(
            set(metadata_by_id[child_id]) == required_metadata_keys
            and all(metadata_by_id[child_id].values())
            for child_id in child_ids
        ),
        "model_group_count_exact": len(model_counts) == expected_model_count,
        "photos_per_model_exact": bool(model_counts)
        and all(count == expected_photos_per_model for count in model_counts.values()),
        "sample_ids_exact": selected_ids == sample_contract["expected_ids"],
        "sample_parent_type_length_exact": all(
            entry["parent_id"] == parent_id
            and entry["content_type"] == sample_contract["required_content_type"]
            and int(entry["content_length"]) > 0
            for entry in sample_entries
        ),
        "sample_total_content_bytes_exact": sample_total_bytes
        == int(sample_contract["expected_total_content_bytes"]),
        "zero_image_body_pixel_fit_render_score": all(
            operation_counts[key] == 0
            for key in (
                "image_body_requests",
                "image_range_requests",
                "pixel_decodes",
                "fit_calls",
                "render_calls",
                "score_calls",
            )
        ),
    }

    group_evidence = config["group_evidence"]
    minimums = config["admission_minimums"]
    admission_gates = {
        "target_stock_project_identity_exact": all(parent_field_gates.values()),
        "public_child_assets_and_group_metadata_complete": all(audit_gates.values()),
        **{
            key: int(group_evidence[key]) >= int(minimums[key])
            for key in sorted(minimums)
        },
    }
    passed = all(audit_gates.values()) and all(admission_gates.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3t-daddy-please-ektar-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "source": {
            "base_url": base_url,
            "parent_id": parent_id,
            "parent_slug_url": source["parent_slug_url"],
        },
        "parent_fields": {
            key: parent_fields.get(key) for key in sorted(required_parent_fields)
        },
        "parent_field_gates": parent_field_gates,
        "child_inventory": {
            "page_count": len(page_numbers),
            "count": len(child_ids),
            "ordered_id_sha256": _ordered_ids_sha256(child_ids),
            "first_id": child_ids[0] if child_ids else None,
            "last_id": child_ids[-1] if child_ids else None,
        },
        "metadata_summary": {
            "entry_count": len(metadata_entries),
            "sha256": _json_sha256(metadata_entries),
            "model_count": len(model_counts),
            "model_counts_sha256": _json_sha256(model_counts),
            "minimum_photos_per_model": min(model_counts.values())
            if model_counts
            else 0,
            "maximum_photos_per_model": max(model_counts.values())
            if model_counts
            else 0,
        },
        "sample": {
            "selection": sample_contract["selection"],
            "ids": selected_ids,
            "count": len(sample_entries),
            "total_content_bytes": sample_total_bytes,
            "entries_sha256": _json_sha256(sample_entries),
            "entries": sample_entries,
        },
        "group_evidence": group_evidence,
        "admission_minimums": minimums,
        "audit_gates": audit_gates,
        "admission_gates": admission_gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _json_sha256(report)
    return report
