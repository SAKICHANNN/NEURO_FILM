"""Shared deterministic builder for additive reference-match review manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

if __package__:
    from scripts.build_reference_match_integration_manifest import (
        IntegrationManifestError,
        _changed_paths,
        _commit,
        _git,
        _public_exports,
        _show,
        _tree,
        _tree_entry,
    )
else:
    from build_reference_match_integration_manifest import (
        IntegrationManifestError,
        _changed_paths,
        _commit,
        _git,
        _public_exports,
        _show,
        _tree,
        _tree_entry,
    )


def encode_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def strict_schema_entry(
    *,
    repo: Path,
    payload: str,
    payload_tree: dict[str, dict[str, str]],
    path: str,
) -> dict[str, str]:
    content = _show(repo, payload, path)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise IntegrationManifestError(
            f"contract schema is not JSON: {path}"
        ) from exc
    if parsed.get("additionalProperties") is not False:
        raise IntegrationManifestError(f"schema is not strict: {path}")
    schema_id = parsed.get("$id")
    if not isinstance(schema_id, str) or not schema_id:
        raise IntegrationManifestError(f"schema has no identity: {path}")
    entry = _tree_entry(payload_tree, path)
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    entry["schema_id"] = schema_id
    return entry


def build_versioned_manifest(
    *,
    repo: Path,
    payload_commit: str,
    base_commit: str,
    main_commit: str,
    schema_id: str,
    payload_scope: str,
    required_public_exports: tuple[str, ...],
    required_contract_schema_paths: tuple[str, ...],
    prior_manifest_path: str,
    prior_manifest_schema_id: str,
    verification_commands: tuple[str, ...],
    claim_ceiling: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    payload = _commit(repo, payload_commit)
    base = _commit(repo, base_commit)
    main = _commit(repo, main_commit)
    if _git(repo, "merge-base", payload, main).decode("ascii").strip() != base:
        raise IntegrationManifestError(
            "pinned base is not the payload/main merge base"
        )
    payload_paths = _changed_paths(repo, base, payload)
    main_paths = _changed_paths(repo, base, main)
    overlap = tuple(sorted(set(payload_paths) & set(main_paths)))
    if overlap:
        raise IntegrationManifestError(
            f"payload/main path overlap: {list(overlap)}"
        )
    payload_tree = _tree(repo, payload)
    files = tuple(_tree_entry(payload_tree, path) for path in payload_paths)
    if "src/color_match/__init__.py" not in payload_tree:
        raise IntegrationManifestError("missing public export module")
    exports = _public_exports(
        _show(repo, payload, "src/color_match/__init__.py")
    )
    missing_exports = sorted(set(required_public_exports) - set(exports))
    if missing_exports:
        raise IntegrationManifestError(
            f"missing public exports: {missing_exports}"
        )
    schemas = [
        strict_schema_entry(
            repo=repo,
            payload=payload,
            payload_tree=payload_tree,
            path=path,
        )
        for path in required_contract_schema_paths
    ]
    prior_content = _show(repo, payload, prior_manifest_path)
    try:
        prior_parsed = json.loads(prior_content)
    except json.JSONDecodeError as exc:
        raise IntegrationManifestError(
            "prior integration manifest is not JSON"
        ) from exc
    prior_entry = _tree_entry(payload_tree, prior_manifest_path)
    prior_entry["sha256"] = hashlib.sha256(prior_content).hexdigest()
    prior_entry["schema_id"] = str(prior_parsed.get("schema_id", ""))
    if prior_entry["schema_id"] != prior_manifest_schema_id:
        raise IntegrationManifestError(
            "prior integration manifest identity is invalid"
        )
    return {
        "schema_id": schema_id,
        "payload_scope": payload_scope,
        "payload_commit": payload,
        "base_commit": base,
        "main_commit": main,
        "payload_changed_file_count": len(files),
        "payload_files": list(files),
        "main_changed_file_count": len(main_paths),
        "overlap_paths": list(overlap),
        "required_public_exports": list(required_public_exports),
        "contract_schemas": schemas,
        "supersedes_manifest": prior_entry,
        "verification_commands": list(verification_commands),
        "claim_ceiling": claim_ceiling,
    }


def validate_manifest_schema(
    manifest: Any,
    *,
    schema_path: Path,
) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(manifest),
            key=lambda item: tuple(
                str(part) for part in item.absolute_path
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationManifestError(
            f"integration manifest schema unavailable: {schema_path}"
        ) from exc
    if errors:
        first = errors[0]
        location = "/".join(
            str(part) for part in first.absolute_path
        )
        prefix = f"{location}: " if location else ""
        raise IntegrationManifestError(
            f"integration manifest schema violation: "
            f"{prefix}{first.message}"
        )


def validate_versioned_manifest(
    *,
    repo: Path,
    manifest: dict[str, Any],
    schema_path: Path,
    rebuild: Callable[..., dict[str, Any]],
) -> None:
    validate_manifest_schema(manifest, schema_path=schema_path)
    rebuilt = rebuild(
        repo=repo,
        payload_commit=str(manifest["payload_commit"]),
        base_commit=str(manifest["base_commit"]),
        main_commit=str(manifest["main_commit"]),
    )
    if manifest != rebuilt:
        raise IntegrationManifestError(
            "manifest differs from commit-derived integration evidence"
        )


__all__ = [
    "build_versioned_manifest",
    "encode_json",
    "strict_schema_entry",
    "validate_manifest_schema",
    "validate_versioned_manifest",
]
