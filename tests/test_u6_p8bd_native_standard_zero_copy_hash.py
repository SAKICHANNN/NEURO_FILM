from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.film_physics.native_standard_runtime_v2 import (
    MemoryBoundNativeStandardRuntime,
    sha256_c_contiguous_array,
)
from src.film_physics.native_standard_package import (
    resolve_native_standard_libraries,
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


def test_p8bd_zero_copy_array_hash_matches_c_order_bytes() -> None:
    values = np.random.default_rng(2026072928).random(
        (33, 35, 3), dtype=np.float32
    )
    assert sha256_c_contiguous_array(values) == hashlib.sha256(
        values.tobytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="C-contiguous"):
        sha256_c_contiguous_array(values[:, ::2])


def test_p8bd_runtime_preserves_exact_output_and_receipt(
    tmp_path: Path,
) -> None:
    baseline, paths = _runtime_fixture(tmp_path)
    package = json.loads(PACKAGE.read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(PROFILE_CONFIG.read_text()),
    )
    candidate = MemoryBoundNativeStandardRuntime(
        package=package,
        artifact=artifact,
        resolved=resolve_native_standard_libraries(package, paths),
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

    def render(runtime: object) -> tuple[bytes, dict[str, object]]:
        chunks: list[bytes] = []
        receipt = runtime.render_to_sink(
            source,
            output_sink=lambda y0, y1, rows: chunks.append(
                rows.tobytes()
            ),
        )
        return b"".join(chunks), receipt

    baseline_bytes, baseline_receipt = render(baseline)
    candidate_bytes, candidate_receipt = render(candidate)
    assert candidate_bytes == baseline_bytes
    assert candidate_receipt == baseline_receipt
    assert source.flags.writeable
