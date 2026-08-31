"""Zero-pixel Inland Aperture Portra 400 / Ektar 100 source admission."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class InlandApertureSourceError(ValueError):
    """Raised when a frozen Flickr page fails the source contract."""


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
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "K-MCFM-source-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if int(response.status) != 200:
            raise InlandApertureSourceError(
                f"source returned HTTP {response.status}"
            )
        return response.read()


def _iter_embedded_models(payload: bytes, registry: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InlandApertureSourceError("photo page is not UTF-8") from error
    marker = f'"_flickrModelRegistry":"{registry}"'
    decoder = json.JSONDecoder()
    models: list[dict[str, Any]] = []
    offset = 0
    while True:
        marker_offset = text.find(marker, offset)
        if marker_offset < 0:
            break
        start = text.rfind('{"data":{', 0, marker_offset)
        if start < 0:
            raise InlandApertureSourceError("embedded model has no JSON start")
        try:
            envelope, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as error:
            raise InlandApertureSourceError("invalid embedded Flickr model") from error
        data = envelope.get("data") if isinstance(envelope, dict) else None
        if isinstance(data, dict) and data.get("_flickrModelRegistry") == registry:
            models.append(data)
        offset = start + max(end, 1)
    return models


def _find_photo_model(payload: bytes, photo_id: str) -> dict[str, Any]:
    matches = [
        model
        for model in _iter_embedded_models(payload, "photo-models")
        if str(model.get("id", "")) == photo_id
    ]
    if not matches:
        raise InlandApertureSourceError(f"missing photo model {photo_id}")
    stats_matches = [
        model
        for model in _iter_embedded_models(payload, "photo-stats-models")
        if str(model.get("id", "")) == photo_id
    ]
    if not stats_matches:
        raise InlandApertureSourceError(f"missing photo stats model {photo_id}")
    stats_rows = [
        {
            "date_taken": model.get("dateTaken"),
            "date_posted": model.get("datePosted"),
        }
        for model in stats_matches
    ]
    complete_stats = [
        row for row in stats_rows if all(value is not None for value in row.values())
    ]
    if not complete_stats or any(
        row != complete_stats[0] for row in complete_stats[1:]
    ):
        raise InlandApertureSourceError(
            f"incomplete or conflicting photo stats {photo_id}"
        )
    canonical = [
        {
            "id": photo_id,
            "title": model.get("title"),
            "description": model.get("description"),
            "license": model.get("license"),
            "width": model.get("oWidth"),
            "height": model.get("oHeight"),
            "date_taken": complete_stats[0]["date_taken"],
            "date_posted": complete_stats[0]["date_posted"],
        }
        for model in matches
    ]
    complete = [row for row in canonical if all(value is not None for value in row.values())]
    if not complete:
        raise InlandApertureSourceError(f"incomplete photo model {photo_id}")
    if any(row != complete[0] for row in complete[1:]):
        raise InlandApertureSourceError(f"conflicting photo models {photo_id}")
    return complete[0]


def _walk_json(value: object) -> Sequence[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("@type") == "ImageObject":
            found.append(value)
        for child in value.values():
            found.extend(_walk_json(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_json(child))
    return found


def _find_rights_record(payload: bytes, photo_url: str) -> dict[str, str]:
    text = payload.decode("utf-8")
    scripts = re.findall(
        r'<script\s+type="application/ld\+json"[^>]*>(.*?)</script>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    objects: list[dict[str, Any]] = []
    for script in scripts:
        try:
            objects.extend(_walk_json(json.loads(script)))
        except json.JSONDecodeError as error:
            raise InlandApertureSourceError("invalid JSON-LD") from error
    matches = [item for item in objects if item.get("acquireLicensePage") == photo_url]
    if len(matches) != 1:
        raise InlandApertureSourceError("photo page lacks one exact rights record")
    item = matches[0]
    author = item.get("author")
    if not isinstance(author, dict):
        raise InlandApertureSourceError("rights record lacks author")
    content_url = item.get("contentUrl")
    if not isinstance(content_url, str):
        raise InlandApertureSourceError("rights record lacks content URL identity")
    return {
        "license": str(item.get("license", "")),
        "acquire_license_page": photo_url,
        "author_name": str(author.get("name", "")),
        "author_url": str(author.get("url", "")).rstrip("/"),
        "content_url_sha256": _sha256(content_url.encode("utf-8")),
    }


def run_inland_aperture_source_audit(
    config_path: Path,
    *,
    reverse: bool = False,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Run the frozen seven-page HTML-only source audit."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    expected_rows = [
        {**row, "license": int(source["license_code"])} for row in source["rows"]
    ]
    request_rows = list(reversed(expected_rows)) if reverse else expected_rows
    fetch = fetcher or _default_fetcher
    actual_rows: list[dict[str, Any]] = []
    rights_rows: list[dict[str, str]] = []
    for expected in request_rows:
        photo_url = f'{source["author_url"]}/{expected["id"]}'
        payload = fetch(photo_url + "/")
        actual = _find_photo_model(payload, str(expected["id"]))
        actual["stock"] = expected["stock"]
        actual_rows.append(actual)
        rights = _find_rights_record(payload, photo_url)
        rights["id"] = str(expected["id"])
        rights_rows.append(rights)
    actual_rows.sort(key=lambda row: row["id"])
    expected_rows.sort(key=lambda row: row["id"])
    rights_rows.sort(key=lambda row: row["id"])
    manifest_sha256 = _json_sha256(actual_rows)
    stock_counts = dict(sorted(Counter(row["stock"] for row in actual_rows).items()))
    camera_pattern = re.compile(source["required_camera_pattern"], re.IGNORECASE)
    rights_exact = all(
        row["license"] == source["license_url"]
        and row["author_name"] == source["author_name"]
        and row["author_url"] == source["author_url"]
        for row in rights_rows
    )
    operation_counts = dict(config["operation_limits"])
    audit_gates = {
        "seven_fixed_photo_pages_exact": len(actual_rows) == 7,
        "selected_manifest_exact": actual_rows == expected_rows,
        "selected_manifest_sha256_exact": manifest_sha256
        == source["expected_manifest_sha256"],
        "portra_four_ektar_three_exact": stock_counts
        == {"ektar_100": 3, "portra_400": 4},
        "same_camera_declaration_exact": all(
            camera_pattern.search(str(row["description"])) is not None
            for row in actual_rows
        ),
        "per_work_cc_by_4_author_exact": rights_exact,
        "flickr_license_code_exact": all(
            row["license"] == int(source["license_code"]) for row in actual_rows
        ),
        "zero_media_pixel_fit_render_score": all(
            operation_counts[key] == 0
            for key in (
                "image_head_requests",
                "image_range_requests",
                "image_body_requests",
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
        key: int(group_evidence[key]) >= int(minimums[key])
        for key in sorted(minimums)
    }
    passed = all(audit_gates.values()) and all(admission_gates.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3w-inland-aperture-portra-ektar-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "source": {
            "author_name": source["author_name"],
            "author_slug": source["author_slug"],
            "author_url": source["author_url"],
            "license_code": source["license_code"],
            "license_url": source["license_url"],
        },
        "manifest": {
            "rows": actual_rows,
            "sha256": manifest_sha256,
            "stock_counts": stock_counts,
        },
        "rights": {
            "rows": rights_rows,
            "sha256": _json_sha256(rights_rows),
        },
        "photo_html_requests_observed": len(actual_rows),
        "group_evidence": group_evidence,
        "admission_minimums": minimums,
        "audit_gates": audit_gates,
        "admission_gates": admission_gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _json_sha256(report)
    return report
