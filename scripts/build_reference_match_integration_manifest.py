"""Build a deterministic, commit-bound reference-match integration manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from jsonschema import Draft202012Validator


SCHEMA_ID = "neuro-film.reference-match-main-integration-manifest.v1"
CLAIM_CEILING = "review-ready-not-merged"
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "schemas"
    / "reference_match_main_integration_manifest_v1.schema.json"
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
)


class IntegrationManifestError(ValueError):
    """Raised when the pinned integration evidence is incomplete."""


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise IntegrationManifestError(message or "git command failed")
    return completed.stdout


def _commit(repo: Path, value: str) -> str:
    resolved = _git(repo, "rev-parse", "--verify", f"{value}^{{commit}}")
    return resolved.decode("ascii").strip()


def _changed_paths(repo: Path, base: str, head: str) -> tuple[str, ...]:
    raw = _git(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        base,
        head,
        "--",
    )
    return tuple(
        sorted(line for line in raw.decode("utf-8").splitlines() if line)
    )


def _tree(repo: Path, commit: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    raw = _git(repo, "ls-tree", "-r", commit).decode("utf-8")
    for line in raw.splitlines():
        metadata, path = line.split("\t", 1)
        mode, object_type, blob = metadata.split(" ", 2)
        if object_type == "blob":
            result[path] = {"path": path, "mode": mode, "blob": blob}
    return result


def _tree_entry(
    tree: dict[str, dict[str, str]],
    path: str,
) -> dict[str, str]:
    try:
        return dict(tree[path])
    except KeyError as exc:
        raise IntegrationManifestError(
            f"missing payload path: {path}"
        ) from exc


def _show(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "show", f"{commit}:{path}")


def _public_exports(init_bytes: bytes) -> tuple[str, ...]:
    module = ast.parse(init_bytes.decode("utf-8"))
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                break
            return tuple(value)
    raise IntegrationManifestError("src.color_match.__all__ is not literal")


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
    schema_paths = tuple(
        path
        for path in payload_paths
        if path.startswith("configs/schemas/reference_shared")
        and path.endswith(".schema.json")
    )
    if not schema_paths:
        raise IntegrationManifestError("shared schema inventory is empty")
    schemas = []
    for path in schema_paths:
        content = _show(repo, payload, path)
        parsed = json.loads(content)
        if parsed.get("additionalProperties") is not False:
            raise IntegrationManifestError(f"schema is not strict: {path}")
        entry = _tree_entry(payload_tree, path)
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        entry["schema_id"] = str(parsed.get("$id", ""))
        schemas.append(entry)
    return {
        "schema_id": SCHEMA_ID,
        "payload_commit": payload,
        "base_commit": base,
        "main_commit": main,
        "payload_changed_file_count": len(files),
        "payload_files": list(files),
        "main_changed_file_count": len(main_paths),
        "overlap_paths": list(overlap),
        "required_public_exports": list(REQUIRED_PUBLIC_EXPORTS),
        "shared_schemas": schemas,
        "verification_commands": [
            "git merge-tree --write-tree <payload_commit> <main_commit>",
            "python -m pytest -q tests/test_color_match*.py",
            "python -m pytest -q",
        ],
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
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationManifestError(
            f"integration manifest schema unavailable: {schema_path}"
        ) from exc
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path)
        prefix = f"{location}: " if location else ""
        raise IntegrationManifestError(
            f"integration manifest schema violation: {prefix}{first.message}"
        )


def validate_manifest(
    *,
    repo: Path,
    manifest: dict[str, Any],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> None:
    validate_manifest_schema(manifest, schema_path=schema_path)
    required = {
        "payload_commit",
        "base_commit",
        "main_commit",
    }
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise IntegrationManifestError("manifest commit binding is incomplete")
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
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
