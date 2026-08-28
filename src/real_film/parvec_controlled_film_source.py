"""Zero-pixel admission for the Parvec controlled-film colour study."""

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
    expected = {"project", "watch", "oembed"}
    if len(order) != 3 or set(order) != expected:
        raise ValueError("request_order must contain project, watch and oembed once")
    return tuple(order)


def run_source_audit(
    config_path: Path,
    *,
    html_reader: HtmlReader = _read_html,
    request_order: Sequence[str] = ("project", "watch", "oembed"),
) -> dict[str, Any]:
    """Run the frozen source gate without requesting presets, media or pixels."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    order = _validate_order(request_order)
    urls = {
        "project": config["project"]["url"],
        "watch": config["video"]["watch_url"],
        "oembed": config["video"]["oembed_url"],
    }
    payloads: dict[str, bytes] = {}
    for role in order:
        payloads[role] = html_reader(urls[role])

    project_text = _visible_text(payloads["project"])
    watch_text = payloads["watch"].decode("utf-8", errors="strict")
    oembed = json.loads(payloads["oembed"].decode("utf-8", errors="strict"))

    project_phrase_results = {
        phrase: phrase in project_text for phrase in config["project"]["required_phrases"]
    }
    chapter_results = {
        row["film_stock_id"]: row["title"] in watch_text
        for row in config["video"]["required_chapters"]
    }
    missing_stock = config["video"]["required_missing_stock"]
    missing_stock_present = missing_stock["title"] in watch_text
    observed_stock_ids = sorted(
        [stock_id for stock_id, present in chapter_results.items() if present]
        + ([missing_stock["film_stock_id"]] if missing_stock_present else [])
    )

    search_text = f"{project_text} {watch_text}"
    artifact_results = {
        artifact: artifact in search_text
        for artifact in config["admission_requirements"]["required_public_artifacts"]
    }
    identity = {
        "title_exact": oembed.get("title") == config["video"]["required_title"],
        "author_exact": oembed.get("author_name")
        == config["video"]["required_author"],
        "official_channel_exact": oembed.get("author_url")
        == "https://www.youtube.com/@parvec",
    }
    required_stock_ids = sorted(
        config["admission_requirements"]["required_stock_ids"]
    )
    gates = {
        "official_project_statements_present": all(project_phrase_results.values()),
        "official_video_identity_present": all(identity.values()),
        "protocol_chapter_present": config["video"]["required_protocol_chapter"]
        in watch_text,
        "portra_ektar_chapters_present": all(chapter_results.values()),
        "complete_three_stock_coverage_present": observed_stock_ids
        == required_stock_ids,
        "public_machine_readable_measurements_and_rights_present": all(
            artifact_results.values()
        ),
        "zero_media_preset_pixel_fit_render_score": all(
            config["network_limits"][key] == 0
            for key in (
                "video_stream_requests",
                "thumbnail_requests",
                "preset_requests",
                "form_submissions",
                "image_requests",
                "pixel_decodes",
                "fit_calls",
                "render_calls",
                "score_calls",
            )
        ),
    }
    passed = all(gates.values())

    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3n-parvec-controlled-film-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": (
            config["decision_if_pass"] if passed else config["decision_if_fail"]
        ),
        "official_sources": {
            "project_url": urls["project"],
            "watch_url": urls["watch"],
            "oembed_url": urls["oembed"],
            "project_phrase_results": project_phrase_results,
            "video_identity_results": identity,
            "protocol_chapter_present": gates["protocol_chapter_present"],
            "target_chapter_results": chapter_results,
            "missing_stock_result": {
                "film_stock_id": missing_stock["film_stock_id"],
                "chapter_present": missing_stock_present,
            },
            "semantic_source_identity_sha256": _canonical_sha256(
                {
                    "project_phrase_results": project_phrase_results,
                    "video_identity_results": identity,
                    "protocol_chapter_present": gates["protocol_chapter_present"],
                    "target_chapter_results": chapter_results,
                    "missing_stock_present": missing_stock_present,
                }
            ),
        },
        "admission": {
            "observed_stock_ids": observed_stock_ids,
            "required_stock_ids": required_stock_ids,
            "public_artifact_results": artifact_results,
        },
        "gates": gates,
        "operation_counts": {
            "html_requests": 3,
            "video_stream_requests": 0,
            "thumbnail_requests": 0,
            "preset_requests": 0,
            "form_submissions": 0,
            "image_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
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
        "operation_counts": report["operation_counts"],
        "claim_ceiling": report["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(stable_payload)
    return report
