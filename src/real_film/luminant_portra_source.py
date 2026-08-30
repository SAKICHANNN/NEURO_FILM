"""Zero-pixel Luminant Halide Portra 400 source admission."""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


class LuminantPortraSourceError(ValueError):
    """Raised when the frozen source or transport fails closed."""


FetchResult = tuple[int, Mapping[str, str], bytes]
Fetcher = Callable[[str, str], FetchResult]

_ALT_PATTERN = re.compile(
    r'alt="\#(?P<id>\d{5})\s+-\s+(?P<date>[^\"]+?)\s+-\s+'
    r'(?P<film>[^\"]+?)\s+-\s+(?P<place>[^\"]*)"'
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _canonical_report_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _default_fetcher(url: str, method: str) -> FetchResult:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "K-MCFM-source-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read() if method == "GET" else b""
        return int(response.status), dict(response.headers.items()), payload


def _header(headers: Mapping[str, str], name: str) -> str:
    lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
    return lowered.get(name.casefold(), "")


def _parse_inventory(
    page_payloads: Sequence[bytes], accepted_labels: set[str], base_url: str
) -> list[dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    for payload in page_payloads:
        text = payload.decode("utf-8")
        for match in _ALT_PATTERN.finditer(text):
            film = html.unescape(match.group("film")).strip()
            if film not in accepted_labels:
                continue
            image_id = match.group("id")
            item = {
                "id": image_id,
                "date": html.unescape(match.group("date")).strip(),
                "film": film,
                "place": html.unescape(match.group("place")).strip(),
                "full_url": f"{base_url}/img/{image_id}/{image_id}_full.jpg",
            }
            if image_id in items and items[image_id] != item:
                raise LuminantPortraSourceError(
                    f"conflicting duplicate image identity {image_id}"
                )
            items[image_id] = item
    return [items[image_id] for image_id in sorted(items)]


def _ranked_sample_ids(inventory: Sequence[dict[str, str]], count: int) -> list[str]:
    ranked = sorted(
        inventory,
        key=lambda item: (_sha256(item["id"].encode("utf-8")), item["id"]),
    )
    return [item["id"] for item in ranked[:count]]


def run_luminant_portra_source_audit(
    config_path: Path,
    *,
    reverse: bool = False,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Run the frozen HTML/HEAD-only source admission audit."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    fetch = fetcher or _default_fetcher
    source = config["source"]
    target = config["target"]
    head_contract = config["head_sample"]

    status, _, about_payload = fetch(source["about_url"], "GET")
    if status != 200:
        raise LuminantPortraSourceError("about page did not return HTTP 200")
    about_text = about_payload.decode("utf-8")
    about_phrase_gates = {
        phrase: phrase in html.unescape(about_text)
        for phrase in source["required_about_phrases"]
    }

    page_numbers = list(range(int(source["first_page"]), int(source["last_page"]) + 1))
    request_page_numbers = list(reversed(page_numbers)) if reverse else page_numbers
    page_payloads: dict[int, bytes] = {}
    for page in request_page_numbers:
        url = source["page_url_template"].format(page=page)
        page_status, _, payload = fetch(url, "GET")
        if page_status != 200:
            raise LuminantPortraSourceError(
                f"gallery page {page} did not return HTTP 200"
            )
        page_payloads[page] = payload
    inventory = _parse_inventory(
        [page_payloads[page] for page in page_numbers],
        set(target["accepted_labels"]),
        source["base_url"],
    )
    label_counts = dict(sorted(Counter(item["film"] for item in inventory).items()))
    inventory_sha256 = _json_sha256(inventory)

    selected_ids = _ranked_sample_ids(inventory, int(head_contract["count"]))
    inventory_by_id = {item["id"]: item for item in inventory}
    request_ids = list(reversed(selected_ids)) if reverse else selected_ids
    head_entries: list[dict[str, Any]] = []
    for image_id in request_ids:
        item = inventory_by_id[image_id]
        head_status, headers, body = fetch(item["full_url"], "HEAD")
        if body:
            raise LuminantPortraSourceError(
                "HEAD transport unexpectedly returned a body"
            )
        try:
            content_length = int(_header(headers, "Content-Length"))
        except ValueError as error:
            raise LuminantPortraSourceError("invalid HEAD Content-Length") from error
        head_entries.append(
            {
                "id": image_id,
                "status": head_status,
                "content_length": content_length,
                "content_type": _header(headers, "Content-Type"),
                "etag": _header(headers, "ETag"),
                "last_modified": _header(headers, "Last-Modified"),
                "accept_ranges": _header(headers, "Accept-Ranges"),
            }
        )
    head_entries.sort(key=lambda entry: entry["id"])
    head_metadata_sha256 = _json_sha256(head_entries)
    selected_total_bytes = sum(entry["content_length"] for entry in head_entries)

    operation_counts = dict(config["operation_limits"])
    audit_gates = {
        "about_photochemical_optical_statements_exact": all(
            about_phrase_gates.values()
        ),
        "site_license_cc_by_4_0": source["license_id"] == "CC-BY-4.0",
        "inventory_count_exact": len(inventory)
        == int(target["expected_inventory_count"]),
        "label_counts_exact": label_counts == target["expected_label_counts"],
        "inventory_sha256_exact": inventory_sha256
        == target["expected_inventory_sha256"],
        "head_sample_ids_exact": selected_ids == head_contract["expected_ids"],
        "head_status_content_type_ranges_valid": all(
            entry["status"] == 200
            and entry["content_length"] > 0
            and entry["content_type"] == head_contract["required_content_type"]
            and entry["accept_ranges"] == head_contract["required_accept_ranges"]
            for entry in head_entries
        ),
        "head_total_bytes_exact": selected_total_bytes
        == int(head_contract["expected_total_bytes"]),
        "head_metadata_sha256_exact": head_metadata_sha256
        == head_contract["expected_metadata_sha256"],
        "zero_image_body_pixel_fit_render_score": all(
            operation_counts[key] == 0
            for key in (
                "image_get_requests",
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
        "target_stock_label_exact": bool(inventory),
        "commercial_compatible_data_rights": source["license_id"] == "CC-BY-4.0",
        "public_original_asset_preflight": all(audit_gates.values()),
        **{
            key: int(group_evidence[key]) >= int(minimums[key])
            for key in sorted(minimums)
        },
    }
    passed = all(audit_gates.values()) and all(admission_gates.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3s-luminant-portra-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "source": {
            "base_url": source["base_url"],
            "about_url": source["about_url"],
            "author_identity": source["author_identity"],
            "license_id": source["license_id"],
            "license_url": source["license_url"],
            "page_range": [source["first_page"], source["last_page"]],
        },
        "about_phrase_gates": about_phrase_gates,
        "inventory": {
            "count": len(inventory),
            "label_counts": label_counts,
            "sha256": inventory_sha256,
            "first_id": inventory[0]["id"] if inventory else None,
            "last_id": inventory[-1]["id"] if inventory else None,
        },
        "head_sample": {
            "selection": head_contract["selection"],
            "ids": selected_ids,
            "count": len(head_entries),
            "total_bytes": selected_total_bytes,
            "metadata_sha256": head_metadata_sha256,
            "entries": head_entries,
        },
        "group_evidence": group_evidence,
        "admission_minimums": minimums,
        "audit_gates": audit_gates,
        "admission_gates": admission_gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_report_sha256(report)
    return report
