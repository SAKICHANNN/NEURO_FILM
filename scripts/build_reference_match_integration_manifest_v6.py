"""Build the non-mutating P73 integration manifest and strict schema."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from scripts.build_reference_match_integration_manifest import (
        IntegrationManifestError,
    )
    from scripts.build_reference_match_integration_manifest_v5 import (
        REQUIRED_CONTRACT_SCHEMA_PATHS as V5_SCHEMA_PATHS,
        REQUIRED_PUBLIC_EXPORTS as V5_PUBLIC_EXPORTS,
    )
    from scripts.reference_match_integration_manifest_common import (
        build_versioned_manifest,
        encode_json,
        validate_manifest_schema as validate_schema_common,
        validate_versioned_manifest,
    )
else:
    from build_reference_match_integration_manifest import (
        IntegrationManifestError,
    )
    from build_reference_match_integration_manifest_v5 import (
        REQUIRED_CONTRACT_SCHEMA_PATHS as V5_SCHEMA_PATHS,
        REQUIRED_PUBLIC_EXPORTS as V5_PUBLIC_EXPORTS,
    )
    from reference_match_integration_manifest_common import (
        build_versioned_manifest,
        encode_json,
        validate_manifest_schema as validate_schema_common,
        validate_versioned_manifest,
    )


SCHEMA_ID = "neuro-film.reference-match-main-integration-manifest.v6"
CLAIM_CEILING = "review-ready-not-merged"
PAYLOAD_SCOPE = "P1-P72"
PRIOR_MANIFEST_PATH = (
    "configs/reference_match_main_integration_manifest_v5.json"
)
PRIOR_MANIFEST_SCHEMA_ID = (
    "neuro-film.reference-match-main-integration-manifest.v5"
)
ROOT = Path(__file__).resolve().parents[1]
V5_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v5.schema.json"
)
DEFAULT_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v6.schema.json"
)
COLOR_ATTESTATION_SCHEMA_PATH = (
    "configs/schemas/reference_runtime_staging_color_attestation_v1.schema.json"
)
ATTESTED_MATCH_VIEW_SCHEMA_PATH = (
    "configs/schemas/"
    "reference_runtime_staging_attested_match_view_bridge_v2.schema.json"
)
NEW_PUBLIC_EXPORTS = (
    "attest_runtime_staging_srgb_metadata_v1",
    "runtime_staging_color_attestation_record_from_json",
    "runtime_staging_color_attestation_record_to_json",
    "validate_runtime_staging_color_attestation_record_v1",
    "validate_runtime_staging_color_attested_decoded_batch_v1",
    "prepare_runtime_staging_attested_match_views_v2",
    "runtime_staging_attested_match_view_bridge_record_from_json",
    "runtime_staging_attested_match_view_bridge_record_to_json",
    "validate_runtime_staging_attested_match_view_bridge_record_v2",
    "validate_runtime_staging_attested_prepared_match_view_batch_v2",
)
REQUIRED_PUBLIC_EXPORTS = (*V5_PUBLIC_EXPORTS, *NEW_PUBLIC_EXPORTS)
REQUIRED_CONTRACT_SCHEMA_PATHS = (
    *V5_SCHEMA_PATHS[:6],
    COLOR_ATTESTATION_SCHEMA_PATH,
    ATTESTED_MATCH_VIEW_SCHEMA_PATH,
    *V5_SCHEMA_PATHS[6:],
)
VERIFICATION_COMMANDS = (
    "git merge-tree --write-tree <payload_commit> <main_commit>",
    "python -m pytest -q -k color_match",
    "python -m pytest -q",
)


def _schema_prefix(path: str) -> dict[str, Any]:
    return {
        "allOf": [
            {"$ref": "#/$defs/contract_schema"},
            {"properties": {"path": {"const": path}}},
        ]
    }


def build_schema(*, v5_schema_path: Path = V5_SCHEMA_PATH) -> dict[str, Any]:
    try:
        schema = json.loads(v5_schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationManifestError(
            f"v5 integration schema unavailable: {v5_schema_path}"
        ) from exc
    result = deepcopy(schema)
    result["$id"] = (
        "https://neuro-film.local/schemas/"
        "reference-match-main-integration-manifest-v6"
    )
    result["title"] = "Neuro-Film Reference Match Main Integration Manifest V6"
    properties = result["properties"]
    properties["schema_id"]["const"] = SCHEMA_ID
    properties["payload_scope"]["const"] = PAYLOAD_SCOPE
    properties["required_public_exports"]["const"] = list(
        REQUIRED_PUBLIC_EXPORTS
    )
    contract = properties["contract_schemas"]
    contract["minItems"] = len(REQUIRED_CONTRACT_SCHEMA_PATHS)
    contract["maxItems"] = len(REQUIRED_CONTRACT_SCHEMA_PATHS)
    contract["prefixItems"][6:6] = [
        _schema_prefix(COLOR_ATTESTATION_SCHEMA_PATH),
        _schema_prefix(ATTESTED_MATCH_VIEW_SCHEMA_PATH),
    ]
    enum = result["$defs"]["contract_schema"]["properties"]["path"]["enum"]
    enum[6:6] = [
        COLOR_ATTESTATION_SCHEMA_PATH,
        ATTESTED_MATCH_VIEW_SCHEMA_PATH,
    ]
    prior = result["$defs"]["prior_manifest"]["properties"]
    prior["path"]["const"] = PRIOR_MANIFEST_PATH
    prior["schema_id"]["const"] = PRIOR_MANIFEST_SCHEMA_ID
    Draft202012Validator.check_schema(result)
    return result


def build_manifest(
    *,
    repo: Path,
    payload_commit: str,
    base_commit: str,
    main_commit: str,
) -> dict[str, Any]:
    return build_versioned_manifest(
        repo=repo,
        payload_commit=payload_commit,
        base_commit=base_commit,
        main_commit=main_commit,
        schema_id=SCHEMA_ID,
        payload_scope=PAYLOAD_SCOPE,
        required_public_exports=REQUIRED_PUBLIC_EXPORTS,
        required_contract_schema_paths=REQUIRED_CONTRACT_SCHEMA_PATHS,
        prior_manifest_path=PRIOR_MANIFEST_PATH,
        prior_manifest_schema_id=PRIOR_MANIFEST_SCHEMA_ID,
        verification_commands=VERIFICATION_COMMANDS,
        claim_ceiling=CLAIM_CEILING,
    )


def validate_manifest_schema(
    manifest: Any,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> None:
    validate_schema_common(manifest, schema_path=schema_path)


def validate_manifest(
    *,
    repo: Path,
    manifest: dict[str, Any],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> None:
    validate_versioned_manifest(
        repo=repo,
        manifest=manifest,
        schema_path=schema_path,
        rebuild=build_manifest,
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
