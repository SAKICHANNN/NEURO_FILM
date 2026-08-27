"""Metadata-only admission for the On Landscape three-stock comparison."""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

HtmlReader = Callable[[str], bytes]


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


def _read_html(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "neuro-film-source-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}: {url}")
        return response.read()


def _text(payload: bytes) -> str:
    return payload.decode("utf-8", errors="strict")


def _comparison_inventory(article_html: str) -> list[dict[str, str]]:
    rows: set[tuple[str, str, str]] = set()
    list_pattern = re.compile(
        r'<ul\s+urlbase="[^"]*/film-comparison/(?P<group>[^"]+)"[^>]*>'
        r"(?P<body>.*?)</ul>",
        re.IGNORECASE | re.DOTALL,
    )
    item_pattern = re.compile(
        r'<li\s+class="swapper"\s+img="(?P<asset>[^"]+)"[^>]*>'
        r"(?P<label>.*?)</li>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in list_pattern.finditer(article_html):
        group = html.unescape(match.group("group")).strip()
        for item in item_pattern.finditer(match.group("body")):
            asset = html.unescape(item.group("asset")).strip()
            label = re.sub(r"<[^>]+>", "", item.group("label"))
            rows.add((group, asset, html.unescape(label).strip()))
    return [
        {"group": group, "asset": asset, "label": label}
        for group, asset, label in sorted(rows)
    ]


def run_source_audit(
    config_path: Path,
    *,
    html_reader: HtmlReader = _read_html,
    request_order: tuple[str, str] = ("article", "terms"),
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(request_order) != {"article", "terms"} or len(request_order) != 2:
        raise ValueError("request_order must contain article and terms exactly once")

    payloads: dict[str, bytes] = {}
    for role in request_order:
        payloads[role] = html_reader(config[role]["url"])
    article_text = _text(payloads["article"])
    terms_text = _text(payloads["terms"])
    inventory = _comparison_inventory(article_text)

    phrase_results = {
        phrase: phrase in article_text
        for phrase in config["article"]["required_phrases"]
    }
    required_stock_results: dict[str, bool] = {}
    for stock in config["article"]["required_stocks"]:
        required_stock_results[stock["film_stock_id"]] = any(
            row["group"] == config["article"]["required_group"]
            and row["label"] == stock["label"]
            and row["asset"] in stock["accepted_asset_names"]
            for row in inventory
        )

    restriction_results = {
        phrase: phrase in terms_text
        for phrase in config["terms"]["required_restriction_phrases"]
    }
    permission_results = {
        permission: permission in terms_text
        for permission in config["terms"]["required_explicit_permissions"]
    }
    gates = {
        "article_capture_scan_statements_present": all(phrase_results.values()),
        "same_group_three_stock_inventory_present": all(
            required_stock_results.values()
        ),
        "official_restrictions_observed": all(restriction_results.values()),
        "explicit_fitting_release_commercial_permissions_present": all(
            permission_results.values()
        ),
        "zero_image_requests": config["network_limits"]["image_requests"] == 0,
        "zero_pixel_fit_render_score": all(
            config["network_limits"][key] == 0
            for key in ("pixel_decodes", "fit_calls", "render_calls", "score_calls")
        ),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3k-onlandscape-three-stock-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": (
            config["decision_if_pass"] if passed else config["decision_if_fail"]
        ),
        "article": {
            "url": config["article"]["url"],
            "bytes": len(payloads["article"]),
            "sha256": _sha256(payloads["article"]),
            "required_phrase_results": phrase_results,
            "comparison_inventory_count": len(inventory),
            "comparison_inventory_sha256": _canonical_sha256(inventory),
            "required_stock_results": required_stock_results,
        },
        "terms": {
            "url": config["terms"]["url"],
            "bytes": len(payloads["terms"]),
            "sha256": _sha256(payloads["terms"]),
            "restriction_results": restriction_results,
            "explicit_permission_results": permission_results,
        },
        "gates": gates,
        "html_requests": 2,
        "image_requests": 0,
        "pixel_decodes": 0,
        "fit_calls": 0,
        "render_calls": 0,
        "score_calls": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_payload = {
        "schema": report["schema"],
        "experiment_id": report["experiment_id"],
        "decision": report["decision"],
        "inventory_sha256": report["article"]["comparison_inventory_sha256"],
        "required_stock_results": required_stock_results,
        "gates": gates,
        "network_counts": {
            key: report[key]
            for key in (
                "html_requests",
                "image_requests",
                "pixel_decodes",
                "fit_calls",
                "render_calls",
                "score_calls",
            )
        },
        "claim_ceiling": report["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(stable_payload)
    return report
