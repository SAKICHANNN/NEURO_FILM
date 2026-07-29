from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw
from scripts.benchmark_u6_p8ax_native_ordered_pipeline_grid import (
    _worker as benchmark_worker,
)
from src.film_physics.native_standard_package import (
    resolve_native_standard_libraries,
)
from src.film_physics.native_standard_runtime import (
    NativeStandardRuntime,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "configs/u6_p8az_native_standard_package_v1.json"
PROFILE_CONFIG = (
    ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
)
RUNTIME_SOURCE = (
    ROOT / "src/film_physics/native_standard_runtime.py"
)
DECISION = (
    ROOT
    / "configs/u6_p8ba_native_standard_runtime_decision_v1.json"
)


def _runtime_fixture(
    tmp_path: Path,
) -> tuple[NativeStandardRuntime, dict[str, Path]]:
    package = json.loads(PACKAGE.read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=json.loads(PROFILE_CONFIG.read_text())
    )
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
    paths = {
        name: Path(row["dll_path"]) for name, row in builds.items()
    }
    resolved = resolve_native_standard_libraries(package, paths)
    return (
        NativeStandardRuntime(
            package=package,
            artifact=artifact,
            resolved=resolved,
        ),
        paths,
    )


def test_p8ba_runtime_matches_benchmark_chain_without_eval_dependency(
    tmp_path: Path,
) -> None:
    runtime, paths = _runtime_fixture(tmp_path)
    height, width, tile_rows = 64, 96, 8
    baseline_path = tmp_path / "baseline.json"
    benchmark_worker(
        profile_config=PROFILE_CONFIG,
        domains_dll=paths["domains"],
        gaussian_dll=paths["gaussian"],
        adjacency_dll=paths["adjacency"],
        gauge_dll=paths["gauge"],
        context_dll=paths["context"],
        display_dll=paths["display"],
        output=baseline_path,
        height=height,
        width=width,
        tile_rows=tile_rows,
        pipeline_workers=4,
        max_in_flight=4,
    )
    baseline = json.loads(baseline_path.read_text())
    source = np.ascontiguousarray(
        p8aq._source_rows(
            y0=0, y1=height, height=height, width=width
        ),
        dtype=np.float32,
    )
    digest = hashlib.sha256()
    consumed = 0

    def sink(y0: int, y1: int, rows: np.ndarray) -> None:
        nonlocal consumed
        assert y0 == consumed
        assert not rows.flags.writeable
        digest.update(rows.tobytes())
        consumed = y1

    receipt = runtime.render_to_sink(source, output_sink=sink)
    assert consumed == height
    assert digest.hexdigest() == baseline["output_sha256"]
    assert receipt["output"]["array_sha256"] == baseline[
        "output_sha256"
    ]
    assert receipt["execution"]["submitted_tiles"] == 2
    assert receipt["execution"]["consumed_tiles"] == 2
    assert source.flags.writeable

    tree = ast.parse(RUNTIME_SOURCE.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        module and module.startswith("src.eval") for module in imports
    )


def test_p8ba_runtime_rejects_hidden_input_copy_and_sink_failure(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime_fixture(tmp_path)
    source = np.zeros((8, 8, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        runtime.render_to_sink(
            source, output_sink=lambda y0, y1, rows: None
        )

    valid = np.zeros((8, 8, 3), dtype=np.float32)

    def fail_sink(y0: int, y1: int, rows: np.ndarray) -> None:
        raise RuntimeError("injected staging failure")

    with pytest.raises(RuntimeError, match="injected staging failure"):
        runtime.render_to_sink(valid, output_sink=fail_sink)
    assert valid.flags.writeable


def test_p8ba_decision_binds_exact_parent_package_and_sources() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["node"] == "U6.P8BA"
    assert decision["result"]["status"] == (
        "pass-opt-in-runtime-boundary"
    )
    assert decision["production_default_changed"] is False
    assert decision["output_contract"]["durable_commit"] is False

    bound_paths = {
        decision["parent_decision"]: decision[
            "parent_decision_sha256"
        ],
        decision["package"]: decision["package_file_sha256"],
        **{
            row["path"]: row["sha256"]
            for row in decision["implementation"].values()
        },
    }
    for relative_path, expected_sha in bound_paths.items():
        assert hashlib.sha256(
            (ROOT / relative_path).read_bytes()
        ).hexdigest() == expected_sha
