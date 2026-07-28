"""Build the non-mutating P68 integration manifest and strict schema."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
    from scripts.build_reference_match_integration_manifest_v3 import (
        REQUIRED_CONTRACT_SCHEMA_PATHS as V3_SCHEMA_PATHS,
        REQUIRED_PUBLIC_EXPORTS as V3_PUBLIC_EXPORTS,
        _strict_schema_entry,
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
    from build_reference_match_integration_manifest_v3 import (
        REQUIRED_CONTRACT_SCHEMA_PATHS as V3_SCHEMA_PATHS,
        REQUIRED_PUBLIC_EXPORTS as V3_PUBLIC_EXPORTS,
        _strict_schema_entry,
    )


SCHEMA_ID = "neuro-film.reference-match-main-integration-manifest.v4"
CLAIM_CEILING = "review-ready-not-merged"
PAYLOAD_SCOPE = "P1-P67"
PRIOR_MANIFEST_PATH = (
    "configs/reference_match_main_integration_manifest_v3.json"
)
PRIOR_MANIFEST_SCHEMA_ID = (
    "neuro-film.reference-match-main-integration-manifest.v3"
)
ROOT = Path(__file__).resolve().parents[1]
V3_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v3.schema.json"
)
DEFAULT_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v4.schema.json"
)
DECODE_SCHEMA_PATH = (
    "configs/schemas/"
    "reference_runtime_qualified_shared_staging_decode_v1.schema.json"
)
DECODE_EXPORTS = (
    "decode_runtime_qualified_shared_staging_bytes_v1",
    "runtime_qualified_shared_staging_decode_record_from_json",
    "runtime_qualified_shared_staging_decode_record_to_json",
    "validate_runtime_qualified_shared_staging_decode_record_v1",
    "validate_runtime_qualified_shared_staging_decoded_batch_v1",
)
REQUIRED_PUBLIC_EXPORTS = (*V3_PUBLIC_EXPORTS, *DECODE_EXPORTS)
REQUIRED_CONTRACT_SCHEMA_PATHS = (
    *V3_SCHEMA_PATHS[:5],
    DECODE_SCHEMA_PATH,
    *V3_SCHEMA_PATHS[5:],
)
VERIFICATION_COMMANDS = (
    "git merge-tree --write-tree <payload_commit> <main_commit>",
    "python -m pytest -q -k color_match",
    "python -m pytest -q",
)


def build_schema(*, v3_schema_path: Path = V3_SCHEMA_PATH) -> dict[str, Any]:
    try:
        schema = json.loads(v3_schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationManifestError(
            f"v3 integration schema unavailable: {v3_schema_path}"
        ) from exc
    result = deepcopy(schema)
    result["$id"] = (
        "https://neuro-film.local/schemas/"
        "reference-match-main-integration-manifest-v4"
    )
    result["title"] = "Neuro-Film Reference Match Main Integration Manifest V4"
    properties = result["properties"]
    properties["schema_id"]["const"] = SCHEMA_ID
    properties["payload_scope"]["const"] = PAYLOAD_SCOPE
    properties["required_public_exports"]["const"] = list(
        REQUIRED_PUBLIC_EXPORTS
    )
    contract = properties["contract_schemas"]
    contract["minItems"] = len(REQUIRED_CONTRACT_SCHEMA_PATHS)
    contract["maxItems"] = len(REQUIRED_CONTRACT_SCHEMA_PATHS)
    contract["prefixItems"].insert(
        5,
        {
            "allOf": [
                {"$ref": "#/$defs/contract_schema"},
                {"properties": {"path": {"const": DECODE_SCHEMA_PATH}}},
            ]
        },
    )
    result["$defs"]["contract_schema"]["properties"]["path"]["enum"].insert(
        5, DECODE_SCHEMA_PATH
    )
    prior = result["$defs"]["prior_manifest"]["properties"]
    prior["path"]["const"] = PRIOR_MANIFEST_PATH
    prior["schema_id"]["const"] = PRIOR_MANIFEST_SCHEMA_ID
    Draft202012Validator.check_schema(result)
    return result


def encode_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


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
    if prior_entry["schema_id"] != PRIOR_MANIFEST_SCHEMA_ID:
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
    parser.add_argument("--output-schema", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args()
    if args.output_schema is not None:
        args.output_schema.write_text(
            encode_json(build_schema()),
            encoding="utf-8",
            newline="\n",
        )
        return 0
    if args.verify_manifest is not None:
        manifest = json.loads(
            args.verify_manifest.read_text(encoding="utf-8")
        )
        validate_manifest(repo=args.repo, manifest=manifest)
        return 0
    if args.payload is None or args.base is None or args.main is None:
        parser.error("--payload, --base and --main are required to build")
    encoded = encode_json(
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
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
