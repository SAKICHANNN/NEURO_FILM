"""Deterministic no-pixel recovery bundles for strict render recipes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .render_contract import (
    RenderContractError,
    load_render_profile,
    validate_render_profile,
    validate_render_recipe,
)

RECOVERY_BUNDLE_SCHEMA_ID = "kmcfm.recipe-recovery-bundle.v1"
RECOVERY_BUNDLE_FORMAT = "deterministic-zip-deflate-v1"
MANIFEST_MEMBER = "manifest.json"
RECIPE_MEMBER = "recipe.json"
PAYLOAD_PREFIX = "payload/"
MAXIMUM_MEMBERS = 16
MAXIMUM_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_HASH = frozenset("0123456789abcdef")


class RecipeRecoveryBundleError(ValueError):
    """Raised when a recipe recovery bundle violates its strict contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _strict_json(data: bytes, label: str) -> Mapping[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise RecipeRecoveryBundleError(f"{label} has a duplicate key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise RecipeRecoveryBundleError(f"{label} has a non-finite number: {value}")

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecipeRecoveryBundleError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise RecipeRecoveryBundleError(f"{label} must be a JSON object")
    return value


def _normalized_member(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RecipeRecoveryBundleError(f"{label} is not a normalized archive path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RecipeRecoveryBundleError(f"{label} is not a normalized archive path")
    if path.as_posix() != value:
        raise RecipeRecoveryBundleError(f"{label} is not a normalized archive path")
    return value


def _root_relative(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise RecipeRecoveryBundleError(
            f"{label} must exist inside the repository root"
        ) from exc
    return relative.as_posix()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _encode_bundle(members: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(name), members[name])
    return stream.getvalue()


def _write_create_only(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir():
        raise RecipeRecoveryBundleError("bundle parent directory does not exist")
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o644,
        )
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("bundle write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise RecipeRecoveryBundleError("bundle destination already exists") from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def build_recipe_recovery_bundle(
    *,
    recipe_path: Path,
    profile_path: Path,
    root: Path,
    bundle_path: Path,
) -> dict[str, Any]:
    """Create one deterministic bundle without reading input or output payloads."""

    recipe_bytes = recipe_path.read_bytes()
    recipe = _strict_json(recipe_bytes, "recipe")
    try:
        validate_render_recipe(recipe)
    except RenderContractError as exc:
        raise RecipeRecoveryBundleError("recipe validation failed") from exc

    profile_relative = _root_relative(profile_path, root, "profile")
    profile = load_render_profile(profile_path, root=root)
    profile_bytes = profile_path.read_bytes()
    if _sha256(profile_bytes) != recipe["profile"]["sha256"]:
        raise RecipeRecoveryBundleError(
            "recipe profile hash differs from selected profile"
        )
    if (
        recipe["profile"]["profile_id"] != profile["profile_id"]
        or recipe["profile"]["profile_version"] != profile["profile_version"]
        or recipe["assets"] != profile["assets"]
    ):
        raise RecipeRecoveryBundleError(
            "recipe profile identity or asset ledger differs"
        )

    payload: dict[str, bytes] = {
        RECIPE_MEMBER: recipe_bytes,
        f"{PAYLOAD_PREFIX}{profile_relative}": profile_bytes,
    }
    rows: list[dict[str, Any]] = [
        {
            "role": "recipe",
            "member": RECIPE_MEMBER,
            "sha256": _sha256(recipe_bytes),
            "bytes": len(recipe_bytes),
        },
        {
            "role": "profile",
            "member": f"{PAYLOAD_PREFIX}{profile_relative}",
            "sha256": _sha256(profile_bytes),
            "bytes": len(profile_bytes),
        },
    ]
    for asset in profile["assets"]:
        relative = str(asset["path"])
        asset_path = root / relative
        asset_bytes = asset_path.read_bytes()
        if _sha256(asset_bytes) != asset["sha256"]:
            raise RecipeRecoveryBundleError(f"profile asset hash mismatch: {relative}")
        member = f"{PAYLOAD_PREFIX}{relative}"
        payload[member] = asset_bytes
        rows.append(
            {
                "role": str(asset["role"]),
                "member": member,
                "sha256": _sha256(asset_bytes),
                "bytes": len(asset_bytes),
            }
        )

    rows.sort(key=lambda row: str(row["member"]))
    manifest: dict[str, Any] = {
        "schema_id": RECOVERY_BUNDLE_SCHEMA_ID,
        "format": RECOVERY_BUNDLE_FORMAT,
        "recipe_member": RECIPE_MEMBER,
        "profile_member": f"{PAYLOAD_PREFIX}{profile_relative}",
        "profile_root": PAYLOAD_PREFIX.rstrip("/"),
        "members": rows,
        "privacy": {
            "includes_input_bytes": False,
            "includes_output_bytes": False,
            "network_required": False,
            "telemetry": False,
        },
        "claim": {
            "output_label": recipe["claim"]["output_label"],
            "evidence_grade": recipe["claim"]["evidence_grade"],
            "calibrated_reference_allowed": False,
        },
    }
    payload[MANIFEST_MEMBER] = _canonical_json(manifest)
    encoded = _encode_bundle(payload)
    _write_create_only(bundle_path, encoded)
    try:
        inspected = inspect_recipe_recovery_bundle(bundle_path)
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise
    if inspected["bundle_sha256"] != _sha256(encoded):
        bundle_path.unlink(missing_ok=True)
        raise RecipeRecoveryBundleError("published bundle identity drifted")
    return inspected


def inspect_recipe_recovery_bundle(bundle_path: Path) -> dict[str, Any]:
    """Validate one bundle in memory without extracting or reading any pixels."""

    bundle_bytes = bundle_path.read_bytes()
    try:
        archive = zipfile.ZipFile(io.BytesIO(bundle_bytes), mode="r")
    except zipfile.BadZipFile as exc:
        raise RecipeRecoveryBundleError("bundle is not a valid ZIP archive") from exc
    with archive:
        infos = archive.infolist()
        if not 1 <= len(infos) <= MAXIMUM_MEMBERS:
            raise RecipeRecoveryBundleError("bundle member count is outside the limit")
        names = [_normalized_member(info.filename, "bundle member") for info in infos]
        if len(set(names)) != len(names):
            raise RecipeRecoveryBundleError("bundle contains duplicate members")
        if MANIFEST_MEMBER not in names:
            raise RecipeRecoveryBundleError("bundle manifest is missing")
        total = 0
        for info in infos:
            if info.is_dir() or info.flag_bits & 0x1:
                raise RecipeRecoveryBundleError(
                    "bundle directories/encryption are forbidden"
                )
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise RecipeRecoveryBundleError("bundle compression is unsupported")
            total += info.file_size
            if total > MAXIMUM_UNCOMPRESSED_BYTES:
                raise RecipeRecoveryBundleError(
                    "bundle uncompressed size exceeds the limit"
                )
        contents = {name: archive.read(name) for name in names}

    manifest = _strict_json(contents[MANIFEST_MEMBER], "bundle manifest")
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
    if set(manifest) != expected_keys:
        raise RecipeRecoveryBundleError("bundle manifest keys mismatch")
    if (
        manifest["schema_id"] != RECOVERY_BUNDLE_SCHEMA_ID
        or manifest["format"] != RECOVERY_BUNDLE_FORMAT
        or manifest["recipe_member"] != RECIPE_MEMBER
        or manifest["profile_root"] != PAYLOAD_PREFIX.rstrip("/")
    ):
        raise RecipeRecoveryBundleError("bundle manifest identity is unsupported")
    privacy = manifest["privacy"]
    if privacy != {
        "includes_input_bytes": False,
        "includes_output_bytes": False,
        "network_required": False,
        "telemetry": False,
    }:
        raise RecipeRecoveryBundleError("bundle privacy contract is unsupported")
    claim = manifest["claim"]
    if (
        not isinstance(claim, Mapping)
        or claim.get("calibrated_reference_allowed") is not False
    ):
        raise RecipeRecoveryBundleError("bundle claim exceeds the recovery boundary")
    if (
        claim.get("output_label") != "film-inspired"
        or claim.get("evidence_grade") != "look-approximation"
    ):
        raise RecipeRecoveryBundleError("bundle claim differs from Look Approximation")

    rows = manifest["members"]
    if not isinstance(rows, list) or not rows:
        raise RecipeRecoveryBundleError("bundle member ledger is invalid")
    expected_names = {MANIFEST_MEMBER}
    seen_roles: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "role",
            "member",
            "sha256",
            "bytes",
        }:
            raise RecipeRecoveryBundleError("bundle member row is invalid")
        role = row["role"]
        member = _normalized_member(row["member"], "ledger member")
        digest = row["sha256"]
        size = row["bytes"]
        if not isinstance(role, str) or role in seen_roles:
            raise RecipeRecoveryBundleError(
                "bundle member roles must be unique strings"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(value not in _HASH for value in digest)
        ):
            raise RecipeRecoveryBundleError("bundle member hash is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RecipeRecoveryBundleError("bundle member size is invalid")
        if (
            member not in contents
            or len(contents[member]) != size
            or _sha256(contents[member]) != digest
        ):
            raise RecipeRecoveryBundleError(
                f"bundle member identity mismatch: {member}"
            )
        seen_roles.add(role)
        expected_names.add(member)
    if set(contents) != expected_names:
        raise RecipeRecoveryBundleError("bundle contains an unexpected member")
    if "recipe" not in seen_roles or "profile" not in seen_roles:
        raise RecipeRecoveryBundleError("bundle recipe/profile roles are missing")

    recipe = _strict_json(contents[RECIPE_MEMBER], "bundled recipe")
    try:
        validate_render_recipe(recipe)
    except RenderContractError as exc:
        raise RecipeRecoveryBundleError("bundled recipe validation failed") from exc
    profile_member = _normalized_member(manifest["profile_member"], "profile member")
    if not profile_member.startswith(PAYLOAD_PREFIX) or profile_member not in contents:
        raise RecipeRecoveryBundleError("bundle profile member is invalid")
    profile = _strict_json(contents[profile_member], "bundled profile")
    try:
        validate_render_profile(profile)
    except RenderContractError as exc:
        raise RecipeRecoveryBundleError("bundled profile validation failed") from exc
    if _sha256(contents[profile_member]) != recipe["profile"]["sha256"]:
        raise RecipeRecoveryBundleError("bundled profile hash differs from recipe")
    if (
        recipe["profile"]["profile_id"] != profile["profile_id"]
        or recipe["profile"]["profile_version"] != profile["profile_version"]
        or recipe["assets"] != profile["assets"]
    ):
        raise RecipeRecoveryBundleError(
            "bundled profile identity or asset ledger differs"
        )
    for asset in profile["assets"]:
        member = f"{PAYLOAD_PREFIX}{asset['path']}"
        if member not in contents or _sha256(contents[member]) != asset["sha256"]:
            raise RecipeRecoveryBundleError(
                f"bundled profile asset mismatch: {asset['path']}"
            )
    return {
        "schema_id": RECOVERY_BUNDLE_SCHEMA_ID,
        "bundle_sha256": _sha256(bundle_bytes),
        "bundle_bytes": len(bundle_bytes),
        "member_count": len(contents),
        "recipe_sha256": _sha256(contents[RECIPE_MEMBER]),
        "profile_sha256": _sha256(contents[profile_member]),
        "input_sha256": recipe["input"]["sha256"],
        "output_sha256": recipe["output"]["sha256"],
        "style": recipe["render"]["style"],
        "privacy": dict(privacy),
        "claim": dict(claim),
    }


__all__ = [
    "RECOVERY_BUNDLE_FORMAT",
    "RECOVERY_BUNDLE_SCHEMA_ID",
    "RecipeRecoveryBundleError",
    "build_recipe_recovery_bundle",
    "inspect_recipe_recovery_bundle",
]
