from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw
from src.film_physics.native_standard_package import (
    native_standard_package_sha256,
    resolve_native_standard_libraries,
    validate_native_standard_package,
    validate_package_profile_artifact,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "configs/u6_p8az_native_standard_package_v1.json"
DECISION = (
    ROOT
    / "configs/u6_p8az_native_standard_package_decision_v1.json"
)
PROFILE_CONFIG = (
    ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
)


def test_p8az_package_binds_profile_and_exact_prebuilt_libraries(
    tmp_path: Path,
) -> None:
    package = json.loads(PACKAGE.read_text())
    validate_native_standard_package(package)
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=json.loads(PROFILE_CONFIG.read_text())
    )
    validate_package_profile_artifact(package, artifact)

    p8aw._patch_runtime()
    builds = p8aq._build_components(
        {
            "component_dll_sha256": {
                name: row["sha256"]
                for name, row in package["components"].items()
            }
        },
        tmp_path / "binaries",
    )
    resolved = resolve_native_standard_libraries(
        package,
        {
            name: Path(row["dll_path"])
            for name, row in builds.items()
        },
    )
    assert resolved.package_sha256 == native_standard_package_sha256(
        package
    )
    assert set(resolved.paths) == set(package["components"])


def test_p8az_package_rejects_profile_and_library_drift(
    tmp_path: Path,
) -> None:
    package = json.loads(PACKAGE.read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=json.loads(PROFILE_CONFIG.read_text())
    )
    mutated = json.loads(json.dumps(artifact))
    mutated["reference_sampling_dpi"] = 3999
    with pytest.raises(ValueError, match="profile artifact drift"):
        validate_package_profile_artifact(package, mutated)

    paths = {}
    for name in package["components"]:
        path = tmp_path / f"{name}.dll"
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_native_standard_libraries(package, paths)


def test_p8az_decision_binds_parent_and_package() -> None:
    decision = json.loads(DECISION.read_text())
    package = json.loads(PACKAGE.read_text())
    assert _sha256(ROOT / decision["parent_decision"]) == decision[
        "parent_decision_sha256"
    ]
    assert _sha256(PACKAGE) == decision["package_file_sha256"]
    assert native_standard_package_sha256(package) == decision[
        "package_canonical_sha256"
    ]
    assert decision["result"]["related_native_profile_tests_passed"] == 93
    assert not decision["result"]["runtime_loading_or_rendering_implemented"]
    assert decision["next_leaf"].startswith("U6.P8BA")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
