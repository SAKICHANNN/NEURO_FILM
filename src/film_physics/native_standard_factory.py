"""Opt-in construction of the selected memory-bounded native runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .native_standard_package import (
    ResolvedNativeStandardLibraries,
    native_standard_package_sha256,
    resolve_native_standard_libraries,
    validate_package_profile_artifact,
)
from .native_standard_runtime_v2 import (
    MemoryBoundNativeStandardRuntime,
)


RUNTIME_IMPLEMENTATION_ID = "memory-bound-zero-copy-input-hash-v1"
RUNTIME_IMPLEMENTATION_SHA256 = (
    "245b3979d64b724bba6c836155a6948cfc21a16e345a920b85674e2b2429377c"
)


def create_opt_in_native_standard_runtime(
    *,
    package: dict[str, Any],
    artifact: dict[str, Any],
    library_paths: Mapping[str, Path],
) -> tuple[MemoryBoundNativeStandardRuntime, dict[str, Any]]:
    """Resolve exact libraries and construct the selected opt-in runtime."""

    validate_package_profile_artifact(package, artifact)
    resolved = resolve_native_standard_libraries(
        package, library_paths
    )
    runtime = MemoryBoundNativeStandardRuntime(
        package=package,
        artifact=artifact,
        resolved=resolved,
    )
    receipt = _factory_receipt(
        package=package,
        artifact=artifact,
        resolved=resolved,
    )
    return runtime, receipt


def _factory_receipt(
    *,
    package: dict[str, Any],
    artifact: dict[str, Any],
    resolved: ResolvedNativeStandardLibraries,
) -> dict[str, Any]:
    core = {
        "schema": "neuro_film.native_standard_runtime_factory_receipt.v1",
        "runtime_implementation_id": RUNTIME_IMPLEMENTATION_ID,
        "runtime_implementation_sha256": RUNTIME_IMPLEMENTATION_SHA256,
        "package_sha256": native_standard_package_sha256(package),
        "artifact_sha256": hashlib.sha256(
            json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "bundle_sha256": artifact["bundle_sha256"],
        "component_sha256": {
            name: package["components"][name]["sha256"]
            for name in sorted(resolved.paths)
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "opt-in exact-package Windows x64 memory-bounded native "
            "Standard runtime construction only; no render, decoder, "
            "encoder, durable commit, mobile, calibration or promotion"
        ),
    }
    return {
        **core,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }


__all__ = [
    "RUNTIME_IMPLEMENTATION_ID",
    "RUNTIME_IMPLEMENTATION_SHA256",
    "create_opt_in_native_standard_runtime",
]
