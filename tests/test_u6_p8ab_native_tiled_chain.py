from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_native_ordered_chain_conformance import (
    build_msvc_native_adjacency_dll,
    build_msvc_native_domains_dll,
    build_msvc_native_gaussian_dll,
    render_loaded_ordered_chain,
)
from src.eval.physical_native_tiled_chain_conformance import (
    render_tiled_native_chain,
)
from src.film_physics.native_adjacency_profile import (
    compile_native_adjacency_profile_payload,
)
from src.film_physics.native_profile import (
    compile_native_domains_profile_payload,
)
from src.film_physics.native_spatial_profile import (
    compile_native_gaussian_profile_payload,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio"
        r"\Installer\vswhere.exe"
    ).is_file(),
    reason="MSVC Build Tools are unavailable",
)
def test_native_tiled_chain_is_exact_below_and_above_halo(
    tmp_path: Path,
) -> None:
    compiler_config = json.loads(
        (
            ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
        ).read_text()
    )
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=compiler_config
    )
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    builds = {
        "domains": build_msvc_native_domains_dll(
            root=ROOT, output_dir=tmp_path / "domains"
        ),
        "gaussian": build_msvc_native_gaussian_dll(
            root=ROOT, output_dir=tmp_path / "gaussian"
        ),
        "adjacency": build_msvc_native_adjacency_dll(
            root=ROOT, output_dir=tmp_path / "adjacency"
        ),
    }
    source = np.random.default_rng(20260729).random((29, 31, 3))
    common = {
        "domains_dll": Path(builds["domains"]["dll_path"]),
        "gaussian_dll": Path(builds["gaussian"]["dll_path"]),
        "adjacency_dll": Path(builds["adjacency"]["dll_path"]),
        "domains_payload": domains_payload,
        "spatial_payload": spatial_payload,
        "adjacency_payload": adjacency_payload,
    }
    full = render_loaded_ordered_chain(
        **common, source=source
    )["scanner_mtf"]
    for tile_rows in (1, 3, 5, 7, 16):
        for reverse in (False, True):
            tiled = render_tiled_native_chain(
                **common,
                source=source,
                tile_rows=tile_rows,
                reverse=reverse,
            )
            assert tiled.tobytes() == full.tobytes()
