"""Zero-pixel admission audit for the FilmMatch Portra 400 source."""

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
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


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


def _drive_folder_ids(payload: bytes) -> list[str]:
    source = payload.decode("utf-8", errors="strict")
    return sorted(
        set(
            re.findall(
                r"drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)",
                source,
            )
        )
    )


def _validate_order(order: Sequence[str], roles: set[str]) -> tuple[str, ...]:
    if len(order) != len(roles) or set(order) != roles:
        raise ValueError("request_order must contain each frozen source exactly once")
    return tuple(order)


def run_source_audit(
    config_path: Path,
    *,
    html_reader: HtmlReader = _read_html,
    request_order: Sequence[str] = ("portramatch", "learnmore", "shooting_charts"),
) -> dict[str, Any]:
    """Run the frozen source gate without requesting media or Drive content."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    roles = set(config["sources"])
    order = _validate_order(request_order, roles)
    payloads = {
        role: html_reader(config["sources"][role]["url"])
        for role in order
    }
    texts = {role: _visible_text(payloads[role]) for role in roles}
    source_rows: dict[str, Any] = {}
    method_results: dict[str, dict[str, bool]] = {}
    for role in sorted(roles):
        source = config["sources"][role]
        payload = payloads[role]
        source_rows[role] = {
            "url": source["url"],
            "bytes": len(payload),
            "sha256": _sha256(payload),
            "drive_folder_ids": _drive_folder_ids(payload),
        }
        folded = texts[role].casefold()
        method_results[role] = {
            phrase: phrase.casefold() in folded
            for phrase in source["required_phrases"]
        }

    combined_text = " ".join(texts[role] for role in sorted(roles)).casefold()
    observed_drive_ids = sorted(
        {folder for row in source_rows.values() for folder in row["drive_folder_ids"]}
    )
    portra_specific_ids = sorted(config["public_links"]["portra_specific_dataset_folder_ids"])
    marker_results = {
        category: {
            marker: marker.casefold() in combined_text
            for marker in markers
        }
        for category, markers in config["admission_markers"].items()
    }
    exact_sources = all(
        source_rows[role]["bytes"] == config["sources"][role]["bytes"]
        and source_rows[role]["sha256"] == config["sources"][role]["sha256"]
        for role in roles
    )
    method_facts = all(
        value
        for role_results in method_results.values()
        for value in role_results.values()
    )
    public_portra_payload = bool(portra_specific_ids) and all(
        folder in observed_drive_ids for folder in portra_specific_ids
    )
    rights_present = all(marker_results["dataset_rights"].values())
    manifest_present = all(marker_results["manifest"].values())
    group_roles_present = all(marker_results["group_roles"].values())
    zero_keys = tuple(key for key in config["operation_limits"] if key != "html_requests")
    gates = {
        "official_page_identities_match": exact_sources,
        "controlled_portra_method_facts_present": method_facts,
        "public_portra_specific_observation_payload_present": public_portra_payload,
        "dataset_fitting_and_derived_parameter_rights_present": rights_present,
        "exact_asset_manifest_and_checksums_present": manifest_present,
        "exact_groups_and_sealed_confirmation_present": group_roles_present,
        "zero_media_drive_pixel_fit_render_score": all(
            config["operation_limits"][key] == 0 for key in zero_keys
        ),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3r-filmmatch-portra-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "source_identity": {
            "pages": source_rows,
            "required_phrase_results": method_results,
            "observed_drive_folder_ids": observed_drive_ids,
        },
        "admission": {
            "target_stock": config["target_stock"],
            "known_practice_charts_folder_id": config["public_links"][
                "known_practice_charts_folder_id"
            ],
            "known_profiling_tools_folder_id": config["public_links"][
                "known_profiling_tools_folder_id"
            ],
            "portra_specific_dataset_folder_ids": portra_specific_ids,
            "marker_results": marker_results,
        },
        "gates": gates,
        "operation_counts": dict(config["operation_limits"]),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["source_identity_sha256"] = _canonical_sha256(report["source_identity"])
    stable_payload = {
        "schema": report["schema"],
        "experiment_id": report["experiment_id"],
        "decision": report["decision"],
        "source_identity_sha256": report["source_identity_sha256"],
        "admission": report["admission"],
        "gates": gates,
        "operation_counts": report["operation_counts"],
        "claim_ceiling": report["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(stable_payload)
    return report
