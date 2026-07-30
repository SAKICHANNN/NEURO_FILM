"""Bounded official-URL preflight for a fresh FiveK pair population."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin

from lxml import html


class FiveKFreshPairPreflightError(ValueError):
    """Raised when the official source or preflight contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def validate_contract(root: Path, config: Mapping[str, Any]) -> None:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKFreshPairPreflightError("preflight contract is not frozen")
    for section, keys in (
        (
            "official_source",
            ("page", "license", "license_file_list"),
        ),
        ("exclusion", ("retained_manifest",)),
    ):
        payload = config[section]
        for key in keys:
            path = root / str(payload[key])
            if not path.is_file() or _sha256(path) != payload[f"{key}_sha256"]:
                raise FiveKFreshPairPreflightError(
                    f"frozen source drift: {section}.{key}"
                )
    network = config["network_preflight"]
    if (
        config.get("image_download_allowed")
        or config.get("training_allowed")
        or config.get("production_integration_allowed")
        or network.get("pixel_payload_download_allowed")
        or network.get("method") != "HEAD"
    ):
        raise FiveKFreshPairPreflightError("metadata-only boundary drift")


def select_pair_names(
    *,
    licensed_names: list[str],
    retained_names: set[str],
    seed: int,
    count: int,
) -> list[str]:
    eligible = sorted(
        name
        for name in set(licensed_names)
        if name and name not in retained_names
    )
    if len(eligible) < count:
        raise FiveKFreshPairPreflightError(
            "insufficient disjoint licensed names"
        )
    selected = random.Random(seed).sample(eligible, count)
    return sorted(selected)


def _page_links(page: Path) -> dict[str, dict[str, str]]:
    document = html.fromstring(page.read_bytes())
    rows: dict[str, dict[str, str]] = {}
    for anchor in document.xpath('//a[starts-with(@href, "img/dng/")]'):
        href = str(anchor.get("href"))
        source_name = Path(href).stem
        table_row = anchor.xpath("ancestor::tr[1]")
        if not table_row:
            continue
        expert = table_row[0].xpath(
            './/a[starts-with(@href, "img/tiff16_c/")]/@href'
        )
        rows[source_name] = {
            "dng_href": href,
            "expert_c_href": str(expert[0]) if len(expert) == 1 else "",
            "page_metadata": " ".join(table_row[0].itertext()).split(),
        }
    return rows


def _head(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "Neuro-Film-Research/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return {
                "url": url,
                "final_url": response.geturl(),
                "status": int(response.status),
                "content_length": int(length) if length is not None else None,
                "error": None,
            }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {
            "url": url,
            "final_url": None,
            "status": None,
            "content_length": None,
            "error": type(exc).__name__,
        }


def run_preflight(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validate_contract(root, config)
    official = config["official_source"]
    licensed_names = [
        value.strip()
        for value in (
            root / str(official["license_file_list"])
        ).read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    exclusion = config["exclusion"]
    with (root / str(exclusion["retained_manifest"])).open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        retained_rows = list(csv.DictReader(handle))
    if len(retained_rows) != exclusion["expected_excluded_rows"]:
        raise FiveKFreshPairPreflightError("retained exclusion count drift")
    retained_names = {str(row["source_name"]) for row in retained_rows}
    selection = config["selection"]
    selected = select_pair_names(
        licensed_names=licensed_names,
        retained_names=retained_names,
        seed=int(selection["seed"]),
        count=int(selection["pair_count"]),
    )
    links = _page_links(root / str(official["page"]))
    base_url = str(official["base_url"])
    rows = []
    missing_links = 0
    urls = []
    for source_name in selected:
        link = links.get(source_name)
        if (
            link is None
            or not link["dng_href"]
            or not link["expert_c_href"]
        ):
            missing_links += 1
            continue
        dng_url = urljoin(base_url, link["dng_href"])
        expert_url = urljoin(base_url, link["expert_c_href"])
        rows.append(
            {
                "source_name": source_name,
                "license_partition": "Adobe",
                "dng_url": dng_url,
                "expert_c_url": expert_url,
                "page_metadata": link["page_metadata"],
            }
        )
        urls.extend([dng_url, expert_url])
    network = config["network_preflight"]
    if len(urls) > int(network["maximum_requests"]):
        raise FiveKFreshPairPreflightError("request budget exceeded")
    timeout = int(network["timeout_seconds"])
    with ThreadPoolExecutor(max_workers=8) as executor:
        facts = list(executor.map(lambda url: _head(url, timeout), urls))
    facts_by_url = {row["url"]: row for row in facts}
    for row in rows:
        row["dng_head"] = facts_by_url[row["dng_url"]]
        row["expert_c_head"] = facts_by_url[row["expert_c_url"]]
    required_status = int(network["required_status"])
    non_200 = sum(
        fact["status"] != required_status for fact in facts
    )
    missing_length = sum(
        fact["content_length"] is None for fact in facts
    )
    expected_bytes = sum(
        int(fact["content_length"] or 0) for fact in facts
    )
    licensed_set = set(licensed_names)
    observed = {
        "selected_pairs": len(selected),
        "selected_ids_in_exact_license_partition": sum(
            name in licensed_set for name in selected
        ),
        "selected_ids_overlapping_retained_128": sum(
            name in retained_names for name in selected
        ),
        "missing_page_links": missing_links,
        "non_200_assets": non_200,
        "assets_missing_content_length": missing_length,
        "expected_bytes": expected_bytes,
        "expected_bytes_at_or_below_limit": expected_bytes
        <= int(network["maximum_expected_bytes"]),
    }
    gates = {
        key: observed[key] == expected
        for key, expected in config["pass_gates"].items()
        if key != "repeat_selection_byte_identity"
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "official_page_sha256": official["page_sha256"],
        "license_sha256": official["license_sha256"],
        "license_file_list_sha256": official[
            "license_file_list_sha256"
        ],
        "selection_seed": selection["seed"],
        "rows": rows,
        "expected_bytes": expected_bytes,
        "rights_scope": official["rights_scope"],
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(_canonical_bytes(manifest))
    stable_payload = {
        "manifest_sha256": _sha256(manifest_path),
        "observed": observed,
        "gates": gates,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        **stable_payload,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_payload)
        ).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": report["manifest_sha256"],
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "report": report,
    }


__all__ = [
    "FiveKFreshPairPreflightError",
    "run_preflight",
    "select_pair_names",
    "validate_contract",
]
