"""Hash-bound opt-in package boundary for the native Standard runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping


NATIVE_STANDARD_PACKAGE_SCHEMA = (
    "neuro_film.native_standard_runtime_package.v1"
)
_COMPONENTS = (
    "domains",
    "gaussian",
    "adjacency",
    "gauge",
    "context",
    "display",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def native_standard_package_sha256(package: dict[str, Any]) -> str:
    validate_native_standard_package(package)
    return hashlib.sha256(_canonical_bytes(package)).hexdigest()


def validate_native_standard_package(
    package: dict[str, Any],
) -> None:
    if set(package) != {
        "schema",
        "package_id",
        "profile_artifact",
        "platform",
        "components",
        "execution_policy",
        "production_default_changed",
        "claim_ceiling",
    }:
        raise ValueError("native Standard package fields drift")
    target = package["platform"]
    policy = package["execution_policy"]
    artifact = package["profile_artifact"]
    components = package["components"]
    if (
        package["schema"] != NATIVE_STANDARD_PACKAGE_SCHEMA
        or package["package_id"]
        != "native-standard-physical-ao6-display-v4-win-x64"
        or package["production_default_changed"] is not False
        or target
        != {
            "os": "windows",
            "architecture": "x86_64",
            "float": "ieee754-binary32",
        }
        or set(artifact) != {"schema", "sha256", "bundle_sha256"}
        or artifact["schema"]
        != "neuro_film.u6_p8b_artifact_only_cpu_profile_artifact.v1"
        or not _is_sha256(artifact["sha256"])
        or not _is_sha256(artifact["bundle_sha256"])
        or set(components) != set(_COMPONENTS)
        or policy
        != {
            "tile_rows": 32,
            "pipeline_workers": 4,
            "max_in_flight": 4,
            "ordered_consumption": True,
            "single_final_quantization": True,
        }
    ):
        raise ValueError("native Standard package identity drift")
    expected_abis = {
        "domains": ("nf_physical_domains_f32_v1", 1),
        "gaussian": ("nf_gaussian_rgb_f32_v1", 1),
        "adjacency": ("nf_bounded_adjacency_f32_v1", 1),
        "gauge": ("nf_neutral_gauge_f32_v1", 1),
        "context": ("nf_ao6_context_f32_v2", 2),
        "display": ("nf_ao6_display_f32_v4", 4),
    }
    for name, (abi, version) in expected_abis.items():
        row = components[name]
        if (
            set(row) != {"abi", "abi_version", "sha256"}
            or row["abi"] != abi
            or row["abi_version"] != version
            or not _is_sha256(row["sha256"])
        ):
            raise ValueError(
                f"native Standard {name} component drift"
            )


def validate_package_profile_artifact(
    package: dict[str, Any], artifact: dict[str, Any]
) -> None:
    validate_native_standard_package(package)
    expected = package["profile_artifact"]
    actual_sha = hashlib.sha256(_canonical_bytes(artifact)).hexdigest()
    if (
        artifact.get("schema") != expected["schema"]
        or actual_sha != expected["sha256"]
        or artifact.get("bundle_sha256") != expected["bundle_sha256"]
    ):
        raise ValueError("native Standard profile artifact drift")


@dataclass(frozen=True)
class ResolvedNativeStandardLibraries:
    package_sha256: str
    paths: Mapping[str, Path]


def resolve_native_standard_libraries(
    package: dict[str, Any],
    library_paths: Mapping[str, Path],
    *,
    require_host_match: bool = True,
) -> ResolvedNativeStandardLibraries:
    """Resolve exact prebuilt libraries without compiling or loading them."""

    validate_native_standard_package(package)
    if require_host_match and (
        sys.platform != "win32"
        or platform.machine().lower() not in {"amd64", "x86_64"}
    ):
        raise RuntimeError("native Standard package host mismatch")
    if set(library_paths) != set(_COMPONENTS):
        raise ValueError("native Standard library inventory drift")
    resolved: dict[str, Path] = {}
    identities: set[str] = set()
    for name in _COMPONENTS:
        path = Path(library_paths[name]).absolute()
        if not path.is_file():
            raise ValueError(f"native Standard {name} library missing")
        identity = str(path.resolve()).casefold()
        if identity in identities:
            raise ValueError("native Standard library path reused")
        identities.add(identity)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != package["components"][name]["sha256"]:
            raise ValueError(
                f"native Standard {name} library hash mismatch"
            )
        resolved[name] = path
    return ResolvedNativeStandardLibraries(
        package_sha256=native_standard_package_sha256(package),
        paths=resolved,
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


__all__ = [
    "NATIVE_STANDARD_PACKAGE_SCHEMA",
    "ResolvedNativeStandardLibraries",
    "native_standard_package_sha256",
    "resolve_native_standard_libraries",
    "validate_native_standard_package",
    "validate_package_profile_artifact",
]
