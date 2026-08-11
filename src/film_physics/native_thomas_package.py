"""Hash-bound package boundary for the native Thomas export runtime."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .native_thomas_export_profile import load_native_thomas_export_profile

NATIVE_THOMAS_PACKAGE_SCHEMA = "neuro_film.native_thomas_export_package.v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_thomas_package_sha256(package: dict[str, Any]) -> str:
    validate_native_thomas_package(package)
    return hashlib.sha256(_canonical_bytes(package)).hexdigest()


def validate_native_thomas_package(package: dict[str, Any]) -> None:
    if set(package) != {
        "schema",
        "package_id",
        "profile_asset",
        "platform",
        "export_core",
        "execution_policy",
        "production_default_changed",
        "claim_ceiling",
    }:
        raise ValueError("native Thomas package fields drift")
    if (
        package["schema"] != NATIVE_THOMAS_PACKAGE_SCHEMA
        or package["package_id"] != "native-thomas-generic-p8cr-win-x64-v1"
        or package["platform"]
        != {
            "os": "windows",
            "architecture": "x86_64",
            "float": "ieee754-binary32",
        }
        or package["profile_asset"]
        != {
            "schema": "neuro_film.native_thomas_export_profile.v1",
            "profile_sha256": (
                "923a985aa90e029332c723a33afb793afe05cae777b3c07061d54828b91e21b4"
            ),
            "file_sha256": (
                "ce11da8e336b322342941ec639e4dac0fe3521224cf0fc907e9841cac51cf2c0"
            ),
        }
        or package["export_core"]
        != {
            "abi": "nf_thomas_rgb16_png_cached_parallel_f32_v1",
            "abi_version": 1,
            "sha256": (
                "e92a638b8cfcc5cfdcb178b76b58b015712be9723f492dc9434b8e1c13e3e914"
            ),
        }
        or package["execution_policy"]
        != {
            "row_partition": 128,
            "maximum_output_bytes": 80000000,
            "publication": "create-only-atomic-same-directory",
        }
        or package["production_default_changed"] is not False
    ):
        raise ValueError("native Thomas package identity drift")


@dataclass(frozen=True)
class ResolvedNativeThomasPackage:
    package_sha256: str
    profile: dict[str, Any]
    library_path: Path


def resolve_native_thomas_package(
    package: dict[str, Any],
    *,
    profile_path: Path,
    library_path: Path,
    require_host_match: bool = True,
) -> ResolvedNativeThomasPackage:
    """Resolve exact profile and prebuilt DLL assets without compiling them."""
    validate_native_thomas_package(package)
    if require_host_match and (
        sys.platform != "win32"
        or platform.machine().lower() not in {"amd64", "x86_64"}
    ):
        raise RuntimeError("native Thomas package host mismatch")
    profile_path = profile_path.absolute()
    library_path = library_path.absolute()
    profile = load_native_thomas_export_profile(
        profile_path,
        expected_profile_sha256=package["profile_asset"]["profile_sha256"],
    )
    if _sha256_file(profile_path) != package["profile_asset"]["file_sha256"]:
        raise ValueError("native Thomas profile asset hash mismatch")
    if not library_path.is_file() or _sha256_file(library_path) != package[
        "export_core"
    ]["sha256"]:
        raise ValueError("native Thomas export core hash mismatch")
    return ResolvedNativeThomasPackage(
        package_sha256=native_thomas_package_sha256(package),
        profile=profile,
        library_path=library_path,
    )


__all__ = [
    "NATIVE_THOMAS_PACKAGE_SCHEMA",
    "ResolvedNativeThomasPackage",
    "native_thomas_package_sha256",
    "resolve_native_thomas_package",
    "validate_native_thomas_package",
]
