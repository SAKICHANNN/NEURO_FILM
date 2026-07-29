from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.film_physics.native_standard_factory import (
    RUNTIME_IMPLEMENTATION_SHA256,
    create_opt_in_native_standard_runtime,
)
from src.film_physics.native_standard_runtime_v2 import (
    MemoryBoundNativeStandardRuntime,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from tests.test_u6_p8ba_native_standard_runtime import (
    PACKAGE,
    PROFILE_CONFIG,
    ROOT,
    _runtime_fixture,
)


RUNTIME_SOURCE = (
    ROOT / "src/film_physics/native_standard_runtime_v2.py"
)
DECISION = (
    ROOT
    / "configs/u6_p8be_native_standard_runtime_factory_decision_v1.json"
)


def test_p8be_factory_binds_selected_runtime_and_renders_exact(
    tmp_path: Path,
) -> None:
    _, paths = _runtime_fixture(tmp_path)
    package = json.loads(PACKAGE.read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(PROFILE_CONFIG.read_text()),
    )
    runtime, factory_receipt = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths=paths,
    )
    assert isinstance(runtime, MemoryBoundNativeStandardRuntime)
    assert factory_receipt["runtime_implementation_sha256"] == (
        RUNTIME_IMPLEMENTATION_SHA256
    )
    assert hashlib.sha256(RUNTIME_SOURCE.read_bytes()).hexdigest() == (
        RUNTIME_IMPLEMENTATION_SHA256
    )
    source = np.ascontiguousarray(
        p8aq._source_rows(
            y0=0,
            y1=64,
            height=64,
            width=96,
        ),
        dtype=np.float32,
    )
    digest = hashlib.sha256()
    receipt = runtime.render_to_sink(
        source,
        output_sink=lambda y0, y1, rows: digest.update(
            rows.tobytes()
        ),
    )
    assert digest.hexdigest() == receipt["output"]["array_sha256"]
    assert not factory_receipt["production_default_changed"]


def test_p8be_decision_binds_factory_and_runtime_sources() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == (
        "pass-opt-in-factory-boundary"
    )
    for path_key, hash_key in (
        ("factory", "factory_sha256"),
        ("runtime", "runtime_sha256"),
    ):
        assert hashlib.sha256(
            (ROOT / decision[path_key]).read_bytes()
        ).hexdigest() == decision[hash_key]
    assert not decision["production_default_changed"]
    assert decision["next_leaf"].startswith("U6.P8BF")
