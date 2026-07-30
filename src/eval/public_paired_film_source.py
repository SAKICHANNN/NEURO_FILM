"""Bounded author-controlled audit for public digital/film paired data."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from src.eval.filmmatch_paired_source import canonical_sha256


class PublicPairedFilmSourceError(RuntimeError):
    """Raised when the live source escapes the frozen metadata-only contract."""


def _fetch(url: str, config: Mapping[str, Any]) -> bytes:
    network = config["network_contract"]
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": str(network["user_agent"]),
            "Accept": "application/vnd.github+json, text/html;q=0.9",
        },
        method="GET",
    )
    maximum = int(network["maximum_response_bytes"])
    with urllib.request.urlopen(  # noqa: S310 - config binds exact HTTPS URLs
        request, timeout=float(network["timeout_seconds"])
    ) as response:
        final = urllib.parse.urlsplit(str(response.geturl()))
        if final.scheme != "https":
            raise PublicPairedFilmSourceError("source redirected off HTTPS")
        payload = response.read(maximum + 1)
    if len(payload) > maximum:
        raise PublicPairedFilmSourceError("response exceeds byte cap")
    return payload


def _json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicPairedFilmSourceError(f"invalid JSON: {label}") from exc


def _html_links(payload: bytes) -> list[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicPairedFilmSourceError("project page is not UTF-8") from exc
    return sorted(
        {
            match.group(1)
            for match in re.finditer(
                r"""href=["'](https://[^"'#]+)["']""",
                text,
                flags=re.IGNORECASE,
            )
        }
    )


def _audit_sillystill(
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    fetcher: Callable[[str, Mapping[str, Any]], bytes],
) -> dict[str, Any]:
    repository = _json(
        fetcher(str(source["repository_api_url"]), config), "repository"
    )
    default_branch = str(repository.get("default_branch", ""))
    commits_url = (
        str(source["repository_api_url"]).rstrip("/")
        + f"/commits/{urllib.parse.quote(default_branch, safe='')}"
    )
    commit = _json(fetcher(commits_url, config), "default commit")
    tree_sha = str(commit.get("commit", {}).get("tree", {}).get("sha", ""))
    tree_url = str(source["tree_api_template"]).format(tree_sha=tree_sha)
    tree = _json(fetcher(tree_url, config), "repository tree")
    paths = sorted(
        str(row.get("path"))
        for row in tree.get("tree", [])
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    )
    dataset_paths = sorted(
        path
        for path in paths
        if any(
            path == prefix.rstrip("/") or path.startswith(prefix)
            for prefix in source["required_dataset_path_prefixes"]
        )
    )
    root_licences = sorted(
        path
        for path in paths
        if path in source["required_root_license_paths"]
    )
    repository_license = repository.get("license")
    explicit_license = bool(
        isinstance(repository_license, Mapping)
        and repository_license.get("spdx_id") not in {None, "", "NOASSERTION"}
    )
    return {
        "source_id": source["source_id"],
        "repository_head": str(commit.get("sha", "")),
        "default_branch": default_branch,
        "tree_sha": tree_sha,
        "tree_truncated": bool(tree.get("truncated")),
        "tree_entry_count": len(paths),
        "dataset_paths": dataset_paths,
        "root_license_paths": root_licences,
        "repository_spdx_id": (
            repository_license.get("spdx_id")
            if isinstance(repository_license, Mapping)
            else None
        ),
        "downloadable_pair_surface_present": bool(dataset_paths),
        "explicit_dataset_reuse_license_present": bool(
            root_licences and explicit_license
        ),
        "stock_pair_topology_claim_present": True,
    }


def _audit_emulating_emulsion(
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    fetcher: Callable[[str, Mapping[str, Any]], bytes],
) -> dict[str, Any]:
    links = _html_links(fetcher(str(source["project_page_url"]), config))
    repositories = _json(
        fetcher(str(source["author_repositories_api_url"]), config),
        "author repositories",
    )
    if not isinstance(repositories, list):
        raise PublicPairedFilmSourceError("author repositories are not a list")
    names = sorted(
        str(row.get("name", ""))
        for row in repositories
        if isinstance(row, Mapping)
    )
    repository_patterns = [
        str(value).lower()
        for value in source["required_repository_name_patterns"]
    ]
    matching_repositories = sorted(
        name
        for name in names
        if any(pattern in name.lower() for pattern in repository_patterns)
    )
    data_patterns = [
        str(value).lower()
        for value in source["required_public_data_link_patterns"]
    ]
    data_links = sorted(
        link
        for link in links
        if any(pattern in link.lower() for pattern in data_patterns)
    )
    explicit_license_links = sorted(
        link
        for link in links
        if "creativecommons.org/licenses/" in link.lower()
        or link.lower().endswith("/license")
    )
    return {
        "source_id": source["source_id"],
        "project_link_count": len(links),
        "author_public_repository_count": len(names),
        "matching_public_repositories": matching_repositories,
        "public_data_links": data_links,
        "explicit_dataset_license_links": explicit_license_links,
        "downloadable_pair_surface_present": bool(
            matching_repositories and data_links
        ),
        "explicit_dataset_reuse_license_present": bool(
            data_links and explicit_license_links
        ),
        "stock_pair_topology_claim_present": True,
    }


def audit_public_sources(
    config: Mapping[str, Any],
    *,
    fetcher: Callable[[str, Mapping[str, Any]], bytes] = _fetch,
) -> dict[str, Any]:
    """Audit only the frozen repository/API/page surfaces."""

    if (
        config.get("schema_version")
        != "u5-r2bi0-public-paired-film-source-refresh-v1"
        or config.get("dataset_download_allowed")
        or config.get("image_request_allowed")
        or config.get("external_contact_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("training_allowed")
    ):
        raise PublicPairedFilmSourceError("contract boundary drift")
    rows = []
    for source in config["sources"]:
        if source["source_id"] == "sillystill-cinestill800t":
            row = _audit_sillystill(source, config, fetcher)
        elif source["source_id"] == "emulating-emulsion-velvia100":
            row = _audit_emulating_emulsion(source, config, fetcher)
        else:
            raise PublicPairedFilmSourceError("unknown source id")
        row["eligible"] = bool(
            row["downloadable_pair_surface_present"]
            and row["explicit_dataset_reuse_license_present"]
            and row["stock_pair_topology_claim_present"]
        )
        rows.append(row)
    eligible = [row["source_id"] for row in rows if row["eligible"]]
    report = {
        "schema_version": "u5-r2bi0-public-paired-film-source-report-v1",
        "experiment_id": config["experiment_id"],
        "sources": rows,
        "eligible_sources": eligible,
        "eligible_source_count": len(eligible),
        "decision": (
            "eligible_source_found_separate_acquisition_required"
            if eligible
            else "no_eligible_source_current_public_surface_closed"
        ),
        "network_facts": {
            "image_or_dataset_payload_requests": 0,
            "external_contacts": 0,
            "maximum_requests": config["network_contract"]["maximum_requests"],
        },
        "operator_fitting_opened": False,
        "training_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "PublicPairedFilmSourceError",
    "audit_public_sources",
]
