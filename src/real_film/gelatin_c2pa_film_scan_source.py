"""Zero-pixel admission for signed Gelatin Labs physical-film scans."""

from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from collections.abc import Callable, Sequence
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


def _visible_text(payload: bytes) -> str:
    source = payload.decode("utf-8", errors="strict")
    without_script = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        " ",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    without_tags = re.sub(r"<[^>]+>", " ", without_script)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _validate_order(order: Sequence[str]) -> tuple[str, ...]:
    expected = {"credentials", "terms"}
    if len(order) != 2 or set(order) != expected:
        raise ValueError("request_order must contain credentials and terms once")
    return tuple(order)


def run_source_audit(
    config_path: Path,
    *,
    html_reader: HtmlReader = _read_html,
    request_order: Sequence[str] = ("credentials", "terms"),
) -> dict[str, Any]:
    """Run the frozen gate without requesting scans, images or verifier data."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    order = _validate_order(request_order)
    urls = {
        role: config["sources"][role]["url"]
        for role in ("credentials", "terms")
    }
    payloads = {role: html_reader(urls[role]) for role in order}
    texts = {role: _visible_text(payloads[role]) for role in urls}

    phrase_results = {
        role: {
            phrase: phrase in texts[role]
            for phrase in config["sources"][role]["required_phrases"]
        }
        for role in urls
    }
    credentials_text = texts["credentials"]
    observed_stocks = sorted(
        row["film_stock_id"]
        for row in config["target_stocks"]
        if row["public_label"] in credentials_text
    )
    required_stocks = sorted(row["film_stock_id"] for row in config["target_stocks"])
    combined_text = f"{texts['credentials']} {texts['terms']}"
    training_forbidden_phrase = config["sources"]["credentials"][
        "training_forbidden_phrase"
    ]
    customer_rights_phrase = config["sources"]["terms"][
        "customer_rights_phrase"
    ]
    training_forbidden = training_forbidden_phrase in credentials_text
    customer_owns_photographs = customer_rights_phrase in texts["terms"]
    artifact_results = {
        artifact: artifact in combined_text
        for artifact in config["admission_requirements"]["required_public_artifacts"]
    }
    zero_operation_keys = tuple(
        key for key in config["network_limits"] if key != "html_requests"
    )
    gates = {
        "official_c2pa_provenance_fields_present": all(
            phrase_results["credentials"].values()
        ),
        "official_customer_rights_boundary_present": all(
            phrase_results["terms"].values()
        ),
        "complete_target_stock_coverage_present": observed_stocks == required_stocks,
        "public_assets_manifest_groups_and_fitting_rights_present": all(
            artifact_results.values()
        ),
        "training_mining_authorized": not training_forbidden,
        "zero_scan_verifier_pixel_fit_render_score": all(
            config["network_limits"][key] == 0 for key in zero_operation_keys
        ),
    }
    passed = all(gates.values())

    operation_counts = dict(config["network_limits"])
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3o-gelatin-c2pa-film-scan-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "official_sources": {
            "urls": urls,
            "required_phrase_results": phrase_results,
            "semantic_source_identity_sha256": _canonical_sha256(phrase_results),
        },
        "admission": {
            "observed_target_stock_ids": observed_stocks,
            "required_target_stock_ids": required_stocks,
            "public_artifact_results": artifact_results,
            "customer_owns_photographs": customer_owns_photographs,
            "cawg_training_and_mining_not_allowed": training_forbidden,
        },
        "gates": gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    stable_payload = {
        "schema": report["schema"],
        "experiment_id": report["experiment_id"],
        "decision": report["decision"],
        "semantic_source_identity_sha256": report["official_sources"][
            "semantic_source_identity_sha256"
        ],
        "admission": report["admission"],
        "gates": gates,
        "operation_counts": operation_counts,
        "claim_ceiling": report["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(stable_payload)
    return report
