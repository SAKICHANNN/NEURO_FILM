"""Build the P170 integration manifest."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from scripts.build_reference_match_integration_manifest import (
        IntegrationManifestError,
    )
    from scripts.build_reference_match_integration_manifest_v41 import (
        V41_REQUIRED_CONTRACT_SCHEMA_PATHS,
        V41_REQUIRED_PUBLIC_EXPORTS,
    )
    from scripts.reference_match_integration_manifest_common import (
        build_versioned_manifest,
        encode_json,
        validate_versioned_manifest,
    )
    from scripts.reference_match_integration_manifest_common import (
        validate_manifest_schema as validate_schema_common,
    )
else:
    from build_reference_match_integration_manifest import IntegrationManifestError
    from build_reference_match_integration_manifest_v41 import (
        V41_REQUIRED_CONTRACT_SCHEMA_PATHS,
        V41_REQUIRED_PUBLIC_EXPORTS,
    )
    from reference_match_integration_manifest_common import (
        build_versioned_manifest,
        encode_json,
        validate_versioned_manifest,
    )
    from reference_match_integration_manifest_common import (
        validate_manifest_schema as validate_schema_common,
    )


SCHEMA_ID = "neuro-film.reference-match-main-integration-manifest.v42"
CLAIM_CEILING = "review-ready-not-merged"
PAYLOAD_SCOPE = "P1-P170"
PRIOR_MANIFEST_PATH = "configs/reference_match_main_integration_manifest_v41.json"
PRIOR_MANIFEST_SCHEMA_ID = (
    "neuro-film.reference-match-main-integration-manifest.v41"
)
ROOT = Path(__file__).resolve().parents[1]
V41_SCHEMA_PATH = (
    ROOT
    / "configs/schemas/reference_match_main_integration_manifest_v41.schema.json"
)
DEFAULT_SCHEMA_PATH = (
    ROOT
    / "configs/schemas/reference_match_main_integration_manifest_v42.schema.json"
)
V42_REQUIRED_PUBLIC_EXPORTS = (
    *V41_REQUIRED_PUBLIC_EXPORTS,
    "REFERENCE_MATCH_PRODUCT_CAPABILITIES_CLAIM_CEILING",
    "REFERENCE_MATCH_PRODUCT_CAPABILITIES_ID",
    "reference_file_supported_input_rails",
    "reference_match_product_capabilities_payload",
)
V42_REQUIRED_CONTRACT_SCHEMA_PATHS = (
    *V41_REQUIRED_CONTRACT_SCHEMA_PATHS,
    "configs/schemas/reference_match_product_capabilities_v1.schema.json",
)
VERIFICATION_COMMANDS = (
    "git merge-tree --write-tree <payload_commit> <main_commit>",
    "python -m pytest -q -k color_match",
    "python -m pytest -q",
)


def build_schema(
    *,
    v41_schema_path: Path = V41_SCHEMA_PATH,
) -> dict[str, Any]:
    try:
        result = deepcopy(
            json.loads(v41_schema_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationManifestError(
            "v41 integration schema unavailable"
        ) from exc
    result["$id"] = (
        "https://neuro-film.local/schemas/"
        "reference-match-main-integration-manifest-v42"
    )
    result["title"] = "Neuro-Film Reference Match Main Integration Manifest V42"
    result["properties"]["schema_id"]["const"] = SCHEMA_ID
    result["properties"]["payload_scope"]["const"] = PAYLOAD_SCOPE
    result["properties"]["required_public_exports"]["const"] = list(
        V42_REQUIRED_PUBLIC_EXPORTS
    )
    product_schema_path = (
        "configs/schemas/reference_match_product_capabilities_v1.schema.json"
    )
    result["$defs"]["contract_schema"]["properties"]["path"]["enum"].append(
        product_schema_path
    )
    result["properties"]["contract_schemas"]["prefixItems"].append(
        {
            "allOf": [
                {"$ref": "#/$defs/contract_schema"},
                {
                    "properties": {
                        "path": {"const": product_schema_path}
                    }
                },
            ]
        }
    )
    result["properties"]["contract_schemas"]["minItems"] = len(
        V42_REQUIRED_CONTRACT_SCHEMA_PATHS
    )
    result["properties"]["contract_schemas"]["maxItems"] = len(
        V42_REQUIRED_CONTRACT_SCHEMA_PATHS
    )
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
        required_public_exports=V42_REQUIRED_PUBLIC_EXPORTS,
        required_contract_schema_paths=V42_REQUIRED_CONTRACT_SCHEMA_PATHS,
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
    if args.output_schema:
        args.output_schema.write_text(
            encode_json(build_schema()),
            encoding="utf-8",
            newline="\n",
        )
        return 0
    if args.verify_manifest:
        validate_manifest(
            repo=args.repo,
            manifest=json.loads(
                args.verify_manifest.read_text(encoding="utf-8")
            ),
        )
        return 0
    if not all((args.payload, args.base, args.main)):
        parser.error("--payload, --base and --main are required")
    value = encode_json(
        build_manifest(
            repo=args.repo,
            payload_commit=args.payload,
            base_commit=args.base,
            main_commit=args.main,
        )
    )
    if args.output:
        args.output.write_text(value, encoding="utf-8", newline="\n")
    else:
        print(value, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
