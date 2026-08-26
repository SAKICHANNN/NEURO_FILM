"""Portable no-pixel recovery bundles with explicit unresolved path bindings."""

from __future__ import annotations

import copy
import io
import os
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .recipe_recovery_bundle import (
    MAXIMUM_MEMBERS,
    MAXIMUM_UNCOMPRESSED_BYTES,
    PAYLOAD_PREFIX,
    RecipeRecoveryBundleError,
    _canonical_json,
    _encode_bundle,
    _normalized_member,
    _root_relative,
    _sha256,
    _strict_json,
    _write_create_only,
)
from .render_contract import (
    RenderContractError,
    load_render_profile,
    sha256_file,
    validate_render_profile,
    validate_render_recipe,
)

PORTABLE_RECIPE_SCHEMA_ID = "kmcfm.portable-render-recipe.v1"
PORTABLE_BUNDLE_SCHEMA_ID = "kmcfm.portable-recipe-recovery-bundle.v1"
PORTABLE_RECIPE_MEMBER = "portable_recipe.json"
PORTABLE_MANIFEST_MEMBER = "manifest.json"
PORTABLE_BUNDLE_FORMAT = "deterministic-zip-deflate-v1"
_INPUT_BINDING = "unresolved-user-input"
_OUTPUT_BINDING = "unresolved-user-output"
_HASH = frozenset("0123456789abcdef")


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(value))


def _path_independent_semantic_sha256(recipe: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(recipe))
    value["input"].pop("path", None)
    value["output"].pop("path", None)
    return _semantic_sha256(value)


def portable_recipe_from_strict(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Replace only strict input/output paths with explicit unresolved bindings."""

    try:
        validate_render_recipe(recipe)
    except RenderContractError as exc:
        raise RecipeRecoveryBundleError(
            "strict source recipe validation failed"
        ) from exc
    result = copy.deepcopy(dict(recipe))
    source_schema = str(result["schema_id"])
    source_semantic_sha256 = _path_independent_semantic_sha256(result)
    result["schema_id"] = PORTABLE_RECIPE_SCHEMA_ID
    result["source_recipe"] = {
        "schema_id": source_schema,
        "semantic_sha256": source_semantic_sha256,
    }
    input_path = result["input"].pop("path")
    output_path = result["output"].pop("path")
    if (
        not isinstance(input_path, str)
        or not input_path
        or not isinstance(output_path, str)
        or not output_path
    ):
        raise RecipeRecoveryBundleError("strict recipe paths are invalid")
    result["input"]["path_binding"] = _INPUT_BINDING
    result["output"]["path_binding"] = _OUTPUT_BINDING
    validate_portable_recipe(result)
    return result


def validate_portable_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate the portable shape without resolving or reading either path."""

    if (
        not isinstance(recipe, Mapping)
        or recipe.get("schema_id") != PORTABLE_RECIPE_SCHEMA_ID
    ):
        raise RecipeRecoveryBundleError("portable recipe schema is unsupported")
    expected = {
        "schema_id",
        "source_recipe",
        "software",
        "input",
        "output",
        "profile",
        "assets",
        "render",
        "claim",
    }
    if set(recipe) != expected:
        raise RecipeRecoveryBundleError("portable recipe fields mismatch")
    source = recipe.get("source_recipe")
    if not isinstance(source, Mapping) or set(source) != {
        "schema_id",
        "semantic_sha256",
    }:
        raise RecipeRecoveryBundleError("portable source-recipe identity is invalid")
    digest = source.get("semantic_sha256")
    if (
        source.get("schema_id") != "kmcfm.render-recipe.v1"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in _HASH for char in digest)
    ):
        raise RecipeRecoveryBundleError("portable source-recipe identity is invalid")
    for field, binding in (("input", _INPUT_BINDING), ("output", _OUTPUT_BINDING)):
        value = recipe.get(field)
        if (
            not isinstance(value, Mapping)
            or value.get("path_binding") != binding
            or "path" in value
        ):
            raise RecipeRecoveryBundleError(f"portable {field} binding is invalid")
    rebound = _bind_unchecked(recipe, "input-placeholder", "output-placeholder")
    try:
        validate_render_recipe(rebound)
    except RenderContractError as exc:
        raise RecipeRecoveryBundleError(
            "portable recipe semantics are invalid"
        ) from exc
    if _path_independent_semantic_sha256(rebound) != digest:
        raise RecipeRecoveryBundleError("portable recipe payload drift")


def _bind_unchecked(
    recipe: Mapping[str, Any], input_path: str, output_path: str
) -> dict[str, Any]:
    result = copy.deepcopy(dict(recipe))
    source = result.pop("source_recipe")
    result["schema_id"] = source["schema_id"]
    result["input"].pop("path_binding")
    result["output"].pop("path_binding")
    result["input"]["path"] = input_path
    result["output"]["path"] = output_path
    return result


def bind_portable_recipe(
    recipe: Mapping[str, Any], *, input_path: str, output_path: str
) -> dict[str, Any]:
    """Resolve both path slots explicitly and return one strict v1 recipe."""

    validate_portable_recipe(recipe)
    if (
        not isinstance(input_path, str)
        or not input_path
        or not isinstance(output_path, str)
        or not output_path
    ):
        raise RecipeRecoveryBundleError(
            "portable recipe bindings must be non-empty strings"
        )
    result = _bind_unchecked(recipe, input_path, output_path)
    try:
        validate_render_recipe(result)
    except RenderContractError as exc:
        raise RecipeRecoveryBundleError(
            "bound strict recipe validation failed"
        ) from exc
    if (
        _path_independent_semantic_sha256(result)
        != recipe["source_recipe"]["semantic_sha256"]
    ):
        raise RecipeRecoveryBundleError("bound recipe differs from source semantics")
    return result


def _read_bundle(path: Path) -> tuple[bytes, dict[str, bytes]]:
    payload = path.read_bytes()
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), "r")
    except zipfile.BadZipFile as exc:
        raise RecipeRecoveryBundleError(
            "portable bundle is not a valid ZIP archive"
        ) from exc
    with archive:
        infos = archive.infolist()
        if not 1 <= len(infos) <= MAXIMUM_MEMBERS:
            raise RecipeRecoveryBundleError(
                "portable bundle member count is outside the limit"
            )
        names = [
            _normalized_member(info.filename, "portable bundle member")
            for info in infos
        ]
        if len(names) != len(set(names)):
            raise RecipeRecoveryBundleError(
                "portable bundle contains duplicate members"
            )
        total = 0
        for info in infos:
            mode = info.external_attr >> 16
            if info.is_dir() or info.flag_bits & 0x1:
                raise RecipeRecoveryBundleError(
                    "portable bundle directories/encryption are forbidden"
                )
            if (
                info.create_system == 3
                and stat.S_IFMT(mode) != 0
                and not stat.S_ISREG(mode)
            ):
                raise RecipeRecoveryBundleError(
                    "portable bundle non-regular members are forbidden"
                )
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise RecipeRecoveryBundleError(
                    "portable bundle compression is unsupported"
                )
            total += info.file_size
            if total > MAXIMUM_UNCOMPRESSED_BYTES:
                raise RecipeRecoveryBundleError(
                    "portable bundle uncompressed size exceeds the limit"
                )
        return payload, {name: archive.read(name) for name in names}


def _validate_manifest(
    contents: Mapping[str, bytes],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if (
        PORTABLE_MANIFEST_MEMBER not in contents
        or PORTABLE_RECIPE_MEMBER not in contents
    ):
        raise RecipeRecoveryBundleError("portable bundle required members are missing")
    manifest = _strict_json(contents[PORTABLE_MANIFEST_MEMBER], "portable manifest")
    expected_keys = {
        "schema_id",
        "format",
        "recipe_member",
        "profile_member",
        "profile_root",
        "members",
        "privacy",
        "claim",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_id") != PORTABLE_BUNDLE_SCHEMA_ID
        or manifest.get("format") != PORTABLE_BUNDLE_FORMAT
        or manifest.get("recipe_member") != PORTABLE_RECIPE_MEMBER
        or manifest.get("profile_root") != PAYLOAD_PREFIX.rstrip("/")
    ):
        raise RecipeRecoveryBundleError(
            "portable bundle manifest identity is unsupported"
        )
    privacy = manifest.get("privacy")
    if privacy != {
        "includes_input_bytes": False,
        "includes_machine_local_paths": False,
        "includes_output_bytes": False,
        "network_required": False,
        "telemetry": False,
    }:
        raise RecipeRecoveryBundleError(
            "portable bundle privacy contract is unsupported"
        )
    claim = manifest.get("claim")
    if claim != {
        "calibrated_reference_allowed": False,
        "evidence_grade": "look-approximation",
        "output_label": "film-inspired",
    }:
        raise RecipeRecoveryBundleError(
            "portable bundle claim differs from Look Approximation"
        )
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise RecipeRecoveryBundleError("portable bundle member ledger is invalid")
    expected_names = {PORTABLE_MANIFEST_MEMBER}
    roles: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "role",
            "member",
            "sha256",
            "bytes",
        }:
            raise RecipeRecoveryBundleError("portable bundle member row is invalid")
        role = row["role"]
        member = _normalized_member(row["member"], "portable ledger member")
        digest, size = row["sha256"], row["bytes"]
        if not isinstance(role, str) or role in roles:
            raise RecipeRecoveryBundleError(
                "portable bundle member roles must be unique"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in _HASH for char in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise RecipeRecoveryBundleError(
                "portable bundle member identity is invalid"
            )
        content = contents.get(member)
        if content is None or len(content) != size or _sha256(content) != digest:
            raise RecipeRecoveryBundleError(
                f"portable bundle member identity mismatch: {member}"
            )
        roles.add(role)
        expected_names.add(member)
    if set(contents) != expected_names or not {"portable_recipe", "profile"} <= roles:
        raise RecipeRecoveryBundleError("portable bundle member inventory mismatch")
    return manifest, claim


def inspect_portable_recipe_recovery_bundle(path: Path) -> dict[str, Any]:
    """Validate one portable bundle without resolving paths or reading pixels."""

    bundle_bytes, contents = _read_bundle(path)
    manifest, claim = _validate_manifest(contents)
    portable = _strict_json(contents[PORTABLE_RECIPE_MEMBER], "portable recipe")
    validate_portable_recipe(portable)
    profile_member = _normalized_member(
        manifest["profile_member"], "portable profile member"
    )
    if not profile_member.startswith(PAYLOAD_PREFIX) or profile_member not in contents:
        raise RecipeRecoveryBundleError("portable bundle profile member is invalid")
    profile = _strict_json(contents[profile_member], "portable profile")
    try:
        validate_render_profile(profile)
    except RenderContractError as exc:
        raise RecipeRecoveryBundleError(
            "portable bundled profile validation failed"
        ) from exc
    if (
        _sha256(contents[profile_member]) != portable["profile"]["sha256"]
        or portable["profile"]["profile_id"] != profile["profile_id"]
        or portable["profile"]["profile_version"] != profile["profile_version"]
        or portable["assets"] != profile["assets"]
    ):
        raise RecipeRecoveryBundleError("portable bundled profile identity differs")
    for asset in profile["assets"]:
        member = f"{PAYLOAD_PREFIX}{asset['path']}"
        if member not in contents or _sha256(contents[member]) != asset["sha256"]:
            raise RecipeRecoveryBundleError(
                f"portable bundled asset mismatch: {asset['path']}"
            )
    return {
        "schema_id": PORTABLE_BUNDLE_SCHEMA_ID,
        "bundle_sha256": _sha256(bundle_bytes),
        "bundle_bytes": len(bundle_bytes),
        "member_count": len(contents),
        "portable_recipe_sha256": _sha256(contents[PORTABLE_RECIPE_MEMBER]),
        "source_recipe_semantic_sha256": portable["source_recipe"]["semantic_sha256"],
        "profile_sha256": _sha256(contents[profile_member]),
        "input_sha256": portable["input"]["sha256"],
        "output_sha256": portable["output"]["sha256"],
        "style": portable["render"]["style"],
        "privacy": dict(manifest["privacy"]),
        "claim": dict(claim),
    }


def bind_portable_recipe_recovery_bundle(
    *,
    bundle_path: Path,
    input_path: Path,
    output_path: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    """Publish one strict private recipe from explicit caller bindings.

    The selected input is hash-verified as an opaque file. No image decode,
    output creation, rendering, path discovery, or network access occurs.
    """

    inspection = inspect_portable_recipe_recovery_bundle(bundle_path)
    _, contents = _read_bundle(bundle_path)
    portable = _strict_json(contents[PORTABLE_RECIPE_MEMBER], "portable recipe")
    validate_portable_recipe(portable)
    if not input_path.is_file():
        raise RecipeRecoveryBundleError("bound input must be an existing file")
    if sha256_file(input_path) != portable["input"]["sha256"]:
        raise RecipeRecoveryBundleError("bound input hash differs from recipe")
    for path, label in ((output_path, "output"), (recipe_path, "recipe")):
        if os.path.lexists(path):
            raise RecipeRecoveryBundleError(f"bound {label} destination already exists")
    if not recipe_path.parent.is_dir():
        raise RecipeRecoveryBundleError(
            "bound recipe parent directory does not exist"
        )
    canonical = {
        os.path.normcase(os.path.abspath(path))
        for path in (input_path, output_path, recipe_path)
    }
    if len(canonical) != 3:
        raise RecipeRecoveryBundleError("bound paths must be distinct")
    bound = bind_portable_recipe(
        portable,
        input_path=str(input_path),
        output_path=str(output_path),
    )
    encoded = _canonical_json(bound)
    _write_create_only(recipe_path, encoded)
    try:
        published = recipe_path.read_bytes()
        if published != encoded:
            raise RecipeRecoveryBundleError("published bound recipe identity drifted")
        decoded = _strict_json(published, "published bound recipe")
        validate_render_recipe(decoded)
        if _path_independent_semantic_sha256(decoded) != inspection[
            "source_recipe_semantic_sha256"
        ]:
            raise RecipeRecoveryBundleError(
                "published bound recipe semantic identity drifted"
            )
    except Exception:
        recipe_path.unlink(missing_ok=True)
        raise
    return {
        "schema_id": "kmcfm.explicit-recipe-binding.v1",
        "bundle_sha256": inspection["bundle_sha256"],
        "recipe_sha256": _sha256(encoded),
        "recipe_bytes": len(encoded),
        "source_recipe_semantic_sha256": inspection[
            "source_recipe_semantic_sha256"
        ],
        "input_sha256": inspection["input_sha256"],
        "expected_output_sha256": inspection["output_sha256"],
        "style": inspection["style"],
        "claim": inspection["claim"],
        "rendered": False,
    }


def build_portable_recipe_recovery_bundle(
    *, recipe_path: Path, profile_path: Path, root: Path, bundle_path: Path
) -> dict[str, Any]:
    """Create a deterministic portable bundle without input/output reads."""

    recipe = _strict_json(recipe_path.read_bytes(), "strict source recipe")
    portable = portable_recipe_from_strict(recipe)
    portable_bytes = _canonical_json(portable)
    profile_relative = _root_relative(profile_path, root, "profile")
    profile = load_render_profile(profile_path, root=root)
    profile_bytes = profile_path.read_bytes()
    if (
        _sha256(profile_bytes) != portable["profile"]["sha256"]
        or portable["profile"]["profile_id"] != profile["profile_id"]
        or portable["profile"]["profile_version"] != profile["profile_version"]
        or portable["assets"] != profile["assets"]
    ):
        raise RecipeRecoveryBundleError("portable source profile identity differs")
    profile_member = f"{PAYLOAD_PREFIX}{profile_relative}"
    payload: dict[str, bytes] = {
        PORTABLE_RECIPE_MEMBER: portable_bytes,
        profile_member: profile_bytes,
    }
    rows: list[dict[str, Any]] = [
        {
            "role": "portable_recipe",
            "member": PORTABLE_RECIPE_MEMBER,
            "sha256": _sha256(portable_bytes),
            "bytes": len(portable_bytes),
        },
        {
            "role": "profile",
            "member": profile_member,
            "sha256": _sha256(profile_bytes),
            "bytes": len(profile_bytes),
        },
    ]
    for asset in profile["assets"]:
        relative = str(asset["path"])
        content = (root / relative).read_bytes()
        if _sha256(content) != asset["sha256"]:
            raise RecipeRecoveryBundleError(
                f"portable profile asset hash mismatch: {relative}"
            )
        member = f"{PAYLOAD_PREFIX}{relative}"
        payload[member] = content
        rows.append(
            {
                "role": str(asset["role"]),
                "member": member,
                "sha256": _sha256(content),
                "bytes": len(content),
            }
        )
    rows.sort(key=lambda row: str(row["member"]))
    manifest = {
        "schema_id": PORTABLE_BUNDLE_SCHEMA_ID,
        "format": PORTABLE_BUNDLE_FORMAT,
        "recipe_member": PORTABLE_RECIPE_MEMBER,
        "profile_member": profile_member,
        "profile_root": PAYLOAD_PREFIX.rstrip("/"),
        "members": rows,
        "privacy": {
            "includes_input_bytes": False,
            "includes_machine_local_paths": False,
            "includes_output_bytes": False,
            "network_required": False,
            "telemetry": False,
        },
        "claim": {
            "calibrated_reference_allowed": False,
            "evidence_grade": portable["claim"]["evidence_grade"],
            "output_label": portable["claim"]["output_label"],
        },
    }
    payload[PORTABLE_MANIFEST_MEMBER] = _canonical_json(manifest)
    encoded = _encode_bundle(payload)
    _write_create_only(bundle_path, encoded)
    try:
        inspected = inspect_portable_recipe_recovery_bundle(bundle_path)
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise
    if inspected["bundle_sha256"] != _sha256(encoded):
        bundle_path.unlink(missing_ok=True)
        raise RecipeRecoveryBundleError("published portable bundle identity drifted")
    return inspected


__all__ = [
    "PORTABLE_BUNDLE_SCHEMA_ID",
    "PORTABLE_RECIPE_SCHEMA_ID",
    "bind_portable_recipe",
    "bind_portable_recipe_recovery_bundle",
    "build_portable_recipe_recovery_bundle",
    "inspect_portable_recipe_recovery_bundle",
    "portable_recipe_from_strict",
    "validate_portable_recipe",
]
