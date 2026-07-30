"""Build the non-mutating P81 integration manifest and strict schema."""

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
    from scripts.build_reference_match_integration_manifest_v9 import (
        REQUIRED_CONTRACT_SCHEMA_PATHS,
        REQUIRED_PUBLIC_EXPORTS,
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
    from build_reference_match_integration_manifest_v9 import (
        REQUIRED_CONTRACT_SCHEMA_PATHS,
        REQUIRED_PUBLIC_EXPORTS,
    )
    from reference_match_integration_manifest_common import (
        build_versioned_manifest,
        encode_json,
        validate_manifest_schema as validate_schema_common,
        validate_versioned_manifest,
    )


SCHEMA_ID = "neuro-film.reference-match-main-integration-manifest.v10"
CLAIM_CEILING = "review-ready-not-merged"
PAYLOAD_SCOPE = "P1-P80"
PRIOR_MANIFEST_PATH = (
    "configs/reference_match_main_integration_manifest_v9.json"
)
PRIOR_MANIFEST_SCHEMA_ID = (
    "neuro-film.reference-match-main-integration-manifest.v9"
)
ROOT = Path(__file__).resolve().parents[1]
V9_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v9.schema.json"
)
DEFAULT_SCHEMA_PATH = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v10.schema.json"
)
VERIFICATION_COMMANDS = (
    "git merge-tree --write-tree <payload_commit> <main_commit>",
    "python -m pytest -q -k color_match",
    "python -m pytest -q",
)


def build_schema(*, v9_schema_path: Path = V9_SCHEMA_PATH) -> dict[str, Any]:
    try:
        schema = json.loads(v9_schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationManifestError(
            f"v9 integration schema unavailable: {v9_schema_path}"
        ) from exc
    result = deepcopy(schema)
    result["$id"] = (
        "https://neuro-film.local/schemas/"
        "reference-match-main-integration-manifest-v10"
    )
    result["title"] = "Neuro-Film Reference Match Main Integration Manifest V10"
    properties = result["properties"]
    properties["schema_id"]["const"] = SCHEMA_ID
    properties["payload_scope"]["const"] = PAYLOAD_SCOPE
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
