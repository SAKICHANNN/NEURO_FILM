"""Zero-pixel admission audit for the public Noritsu Portra profile route."""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

JsonReader = Callable[[str], bytes]


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


def _read_json(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "neuro-film-source-audit/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}: {url}")
        return response.read()


def _api_url(repo: str, kind: str, value: str) -> str:
    escaped = "/".join(urllib.parse.quote(part, safe="") for part in value.split("/"))
    return f"https://api.github.com/repos/{repo}/{kind}/{escaped}"


def _decode_document(payload: bytes) -> dict[str, Any]:
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise TypeError("GitHub response must be a JSON object")
    return document


def _decode_blob(document: dict[str, Any]) -> bytes:
    if document.get("encoding") != "base64" or not isinstance(
        document.get("content"), str
    ):
        raise ValueError("expected GitHub base64 content response")
    return base64.b64decode(document["content"], validate=True)


def _validate_order(order: Sequence[str], roles: set[str]) -> tuple[str, ...]:
    if len(order) != len(roles) or set(order) != roles:
        raise ValueError("asset_order must contain every frozen asset exactly once")
    return tuple(order)


def run_source_audit(
    config_path: Path,
    *,
    json_reader: JsonReader = _read_json,
    asset_order: Sequence[str] = (
        "readme",
        "license",
        "portra_profile",
        "paired_audit",
    ),
) -> dict[str, Any]:
    """Run the frozen Git/source gate without reading the profile or pair bodies."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    roles = set(config["assets"])
    order = _validate_order(asset_order, roles)

    repository_rows: dict[str, Any] = {}
    for role in sorted(config["repositories"]):
        source = config["repositories"][role]
        repo = source["repo"]
        repo_doc = _decode_document(json_reader(f"https://api.github.com/repos/{repo}"))
        commit_doc = _decode_document(
            json_reader(_api_url(repo, "commits", source["commit"]))
        )
        repository_rows[role] = {
            "repo": repo,
            "license_spdx": (repo_doc.get("license") or {}).get("spdx_id"),
            "commit": commit_doc.get("sha"),
            "tree": ((commit_doc.get("commit") or {}).get("tree") or {}).get("sha"),
        }

    asset_rows: dict[str, Any] = {}
    phrase_results: dict[str, dict[str, bool]] = {}
    for role in order:
        asset = config["assets"][role]
        repo = config["repositories"][asset["repo_role"]]["repo"]
        document = _decode_document(
            json_reader(_api_url(repo, "contents", asset["path"]))
        )
        row: dict[str, Any] = {
            "repo": repo,
            "path": asset["path"],
            "bytes": document.get("size"),
            "git_blob": document.get("sha"),
            "body_read": False,
        }
        results: dict[str, bool] = {}
        if asset["decode_text"]:
            body = _decode_blob(document)
            text = body.decode("utf-8", errors="strict").casefold()
            row.update({"body_read": True, "sha256": _sha256(body)})
            results = {
                phrase: phrase.casefold() in text
                for phrase in asset["required_phrases"]
            }
        asset_rows[role] = row
        phrase_results[role] = results

    exact_repositories = all(
        repository_rows[role]["license_spdx"] == "MIT"
        and repository_rows[role]["commit"] == source["commit"]
        and repository_rows[role]["tree"] == source["tree"]
        for role, source in config["repositories"].items()
    )
    exact_assets = all(
        asset_rows[role]["bytes"] == asset["bytes"]
        and asset_rows[role]["git_blob"] == asset["git_blob"]
        and (
            not asset["decode_text"]
            or asset_rows[role].get("sha256") == asset["sha256"]
        )
        for role, asset in config["assets"].items()
    )
    physical_pair_statements = all(
        value for results in phrase_results.values() for value in results.values()
    )
    pair_public = bool(config["published_pair_assets"])
    pair_rights = bool(config["published_pair_rights"])
    group_roles = bool(config["published_group_roles"])
    pair_manifest = bool(config["published_pair_manifest"])
    zero_keys = tuple(
        key for key in config["operation_limits"] if key != "github_json_requests"
    )
    gates = {
        "official_repository_identities_match": exact_repositories,
        "official_asset_identities_match": exact_assets,
        "real_portra_machine_pair_statements_present": physical_pair_statements,
        "owner_pair_payload_publicly_addressable": pair_public,
        "owner_pair_fitting_and_derived_parameter_rights_present": pair_rights,
        "independent_roll_process_scanner_and_confirmation_groups_present": group_roles,
        "public_pair_manifest_and_checksums_present": pair_manifest,
        "zero_profile_pair_media_pixel_fit_render_score": all(
            config["operation_limits"][key] == 0 for key in zero_keys
        )
        and not asset_rows["portra_profile"]["body_read"],
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3v-noritsu-portra-physical-profile-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "source_identity": {
            "repositories": repository_rows,
            "assets": asset_rows,
            "required_phrase_results": phrase_results,
        },
        "admission": {
            "target_stock": config["target_stock"],
            "published_pair_assets": config["published_pair_assets"],
            "published_pair_rights": config["published_pair_rights"],
            "published_group_roles": config["published_group_roles"],
            "published_pair_manifest": config["published_pair_manifest"],
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
