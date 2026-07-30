"""Build the non-mutating P64 reference-match integration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


SCHEMA_ID = "neuro-film.reference-match-main-integration-manifest.v2"
CLAIM_CEILING = "review-ready-not-merged"
PAYLOAD_SCOPE = "P1-P63"
PRIOR_MANIFEST_PATH = (
    "configs/reference_match_main_integration_manifest_v1.json"
)
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v2.schema.json"
)
REQUIRED_PUBLIC_EXPORTS = (
    "make_shared_reference_operator_v1",
    "guard_shared_numeric_batch_v1",
    "authorize_shared_product_staging_v1",
    "commit_external_shared_staging_v1",
    "verify_external_shared_staging_v1",
    "build_shared_reference_composition_v1",
    "commit_shared_filmfx_staging_v1",
    "verify_shared_filmfx_staging_v1",
    "authorize_shared_local_delivery_v1",
    "commit_shared_local_delivery_v1",
    "verify_shared_local_delivery_v1",
    "make_successor_runtime_evidence_v1",
    "bind_successor_runtime_evidence_v1",
    "validate_successor_runtime_evidence_v1",
    "qualify_shared_product_authorization_runtime_v1",
    "validate_runtime_qualified_shared_authorization_v1",
    "commit_runtime_qualified_external_shared_staging_v1",
    "validate_runtime_qualified_external_shared_staging_run_v1",
    "verify_runtime_qualified_external_shared_staging_v1",
    "validate_runtime_qualified_external_shared_staging_verification_v1",
)
REQUIRED_CONTRACT_SCHEMA_PATHS = (
    "configs/schemas/reference_match_successor_declaration_v1.schema.json",
    "configs/schemas/reference_match_successor_runtime_evidence_v1.schema.json",
    "configs/schemas/reference_runtime_qualified_shared_staging_run_v1.schema.json",
    "configs/schemas/"
    "reference_runtime_qualified_shared_staging_verification_v1.schema.json",
    "configs/schemas/reference_shared_composition_v1.schema.json",
    "configs/schemas/reference_shared_filmfx_run_v1.schema.json",
    "configs/schemas/reference_shared_filmfx_staging_verification_v1.schema.json",
    "configs/schemas/reference_shared_local_delivery_authorization_v1.schema.json",
    "configs/schemas/reference_shared_local_delivery_v1.schema.json",
    "configs/schemas/reference_shared_local_delivery_verification_v1.schema.json",
    "configs/schemas/reference_shared_numeric_batch_guard_v1.schema.json",
    "configs/schemas/reference_shared_operator_batch_v1.schema.json",
    "configs/schemas/reference_shared_product_staging_authorization_v1.schema.json",
    "configs/schemas/reference_shared_runtime_qualification_v1.schema.json",
)
VERIFICATION_COMMANDS = (
    "git merge-tree --write-tree <payload_commit> <main_commit>",
    "python -m pytest -q -k color_match",
    "python -m pytest -q",
)


def _strict_schema_entry(
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


def build_manifest(
    *,
    repo: Path,
    payload_commit: str,
    base_commit: str,
    main_commit: str,
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
    missing_exports = sorted(set(REQUIRED_PUBLIC_EXPORTS) - set(exports))
    if missing_exports:
        raise IntegrationManifestError(
            f"missing public exports: {missing_exports}"
        )
    schemas = [
        _strict_schema_entry(
            repo=repo,
            payload=payload,
            payload_tree=payload_tree,
            path=path,
        )
        for path in REQUIRED_CONTRACT_SCHEMA_PATHS
    ]
    prior_content = _show(repo, payload, PRIOR_MANIFEST_PATH)
    try:
        prior_parsed = json.loads(prior_content)
    except json.JSONDecodeError as exc:
        raise IntegrationManifestError(
            "prior integration manifest is not JSON"
        ) from exc
    prior_entry = _tree_entry(payload_tree, PRIOR_MANIFEST_PATH)
    prior_entry["sha256"] = hashlib.sha256(prior_content).hexdigest()
    prior_entry["schema_id"] = str(prior_parsed.get("schema_id", ""))
    if (
        prior_entry["schema_id"]
        != "neuro-film.reference-match-main-integration-manifest.v1"
    ):
        raise IntegrationManifestError(
            "prior integration manifest identity is invalid"
        )
    return {
        "schema_id": SCHEMA_ID,
        "payload_scope": PAYLOAD_SCOPE,
        "payload_commit": payload,
        "base_commit": base,
        "main_commit": main,
        "payload_changed_file_count": len(files),
        "payload_files": list(files),
        "main_changed_file_count": len(main_paths),
        "overlap_paths": list(overlap),
        "required_public_exports": list(REQUIRED_PUBLIC_EXPORTS),
        "contract_schemas": schemas,
        "supersedes_manifest": prior_entry,
        "verification_commands": list(VERIFICATION_COMMANDS),
        "claim_ceiling": CLAIM_CEILING,
    }


def encode_manifest(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def validate_manifest_schema(
    manifest: Any,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
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


def validate_manifest(
    *,
    repo: Path,
    manifest: dict[str, Any],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> None:
    validate_manifest_schema(manifest, schema_path=schema_path)
    required = {"payload_commit", "base_commit", "main_commit"}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise IntegrationManifestError(
            "manifest commit binding is incomplete"
        )
    rebuilt = build_manifest(
        repo=repo,
        payload_commit=str(manifest["payload_commit"]),
        base_commit=str(manifest["base_commit"]),
        main_commit=str(manifest["main_commit"]),
    )
    if manifest != rebuilt:
        raise IntegrationManifestError(
            "manifest differs from commit-derived integration evidence"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--payload")
    parser.add_argument("--base")
    parser.add_argument("--main")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args()
    if args.verify_manifest is not None:
        manifest = json.loads(
            args.verify_manifest.read_text(encoding="utf-8")
        )
        validate_manifest(repo=args.repo, manifest=manifest)
        return 0
    if args.payload is None or args.base is None or args.main is None:
        parser.error("--payload, --base and --main are required to build")
    encoded = encode_manifest(
        build_manifest(
            repo=args.repo,
            payload_commit=args.payload,
            base_commit=args.base,
            main_commit=args.main,
        )
    )
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(
            encoded,
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
