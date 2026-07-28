from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.eval.physical_native_f32_conformance import (
    _build_component,
    render_tiled_f32_chain,
    stream_tiled_f32_chain,
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


def test_p8af_stream_matches_full_frame_rows(tmp_path: Path) -> None:
    config = json.loads(
        (
            ROOT / "configs/u6_p8af_native_standard_f32_streaming_v1.json"
        ).read_text()
    )
    profile_config = json.loads(
        (ROOT / config["profile_compiler_config"]).read_text()
    )
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=profile_config
    )
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    components = {
        "domains": (
            "nf_physical_domains_f32_v1",
            "native/film_physics/nf_physical_domains_f32_v1.c",
            "native/film_physics/nf_physical_domains_f32_v1.h",
        ),
        "gaussian": (
            "nf_gaussian_rgb_f32_v1",
            "native/film_physics/nf_gaussian_rgb_f32_v1.c",
            "native/film_physics/nf_gaussian_rgb_f32_v1.h",
        ),
        "adjacency": (
            "nf_bounded_adjacency_f32_v1",
            "native/film_physics/nf_bounded_adjacency_f32_v1.c",
            "native/film_physics/nf_bounded_adjacency_f32_v1.h",
        ),
    }
    builds = {}
    for name, (basename, source, header) in components.items():
        builds[name] = _build_component(
            root=ROOT,
            output_dir=tmp_path / name,
            component={"source": source, "header": header},
            basename=basename,
        )
    height = 17
    width = 19
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)

    def provider(y0: int, y1: int) -> np.ndarray:
        rows = np.empty((y1 - y0, width, 3), dtype=np.float32)
        rows[..., 0] = x[None, :]
        rows[..., 1] = y[y0:y1, None]
        rows[..., 2] = (
            np.float32(0.15)
            + np.float32(0.45) * x[None, :]
            + np.float32(0.35) * y[y0:y1, None]
        )
        np.clip(rows, 0.0, 1.0, out=rows)
        return rows

    source = provider(0, height)
    paths = {
        name: Path(row["dll_path"]) for name, row in builds.items()
    }
    expected = render_tiled_f32_chain(
        domains_dll=paths["domains"],
        gaussian_dll=paths["gaussian"],
        adjacency_dll=paths["adjacency"],
        domains_payload=domains_payload,
        spatial_payload=spatial_payload,
        adjacency_payload=adjacency_payload,
        source=source,
        tile_rows=5,
    )
    digest = hashlib.sha256()
    observed_rows = []

    def sink(y0: int, y1: int, rows: np.ndarray) -> None:
        observed_rows.append((y0, y1))
        digest.update(rows.tobytes())

    stream_tiled_f32_chain(
        domains_dll=paths["domains"],
        gaussian_dll=paths["gaussian"],
        adjacency_dll=paths["adjacency"],
        domains_payload=domains_payload,
        spatial_payload=spatial_payload,
        adjacency_payload=adjacency_payload,
        height=height,
        width=width,
        tile_rows=5,
        source_provider=provider,
        output_sink=sink,
    )
    assert observed_rows == [(0, 5), (5, 10), (10, 15), (15, 17)]
    assert digest.hexdigest() == hashlib.sha256(
        expected.tobytes()
    ).hexdigest()
