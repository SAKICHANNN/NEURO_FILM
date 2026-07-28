"""Halo-tiled parity audit for the ordered native scanner-linear chain."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_native_ordered_chain_conformance import (
    build_msvc_native_adjacency_dll,
    build_msvc_native_domains_dll,
    build_msvc_native_gaussian_dll,
    render_loaded_ordered_chain,
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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _load_exact_json(
    root: Path, relative: str, expected_sha256: str
) -> dict[str, Any]:
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"hash mismatch: {relative}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {relative}")
    return value


def build_tiled_source(config: dict[str, Any]) -> np.ndarray:
    height = int(config["source"]["height"])
    width = int(config["source"]["width"])
    seed = int(config["source"]["seed"])
    if height <= 0 or width <= 0:
        raise ValueError("tiled source dimensions must be positive")
    y, x = np.mgrid[0:height, 0:width]
    source = np.empty((height, width, 3), dtype=np.float64)
    source[..., 0] = (x + 2.0 * y) / (
        width - 1 + 2.0 * (height - 1)
    )
    source[..., 1] = ((x * 17 + y * 13) % 37) / 36.0
    source[..., 2] = np.random.default_rng(seed).random(
        (height, width)
    )
    source[0, 0] = (1.0, 0.0, 0.5)
    source[-1, -1] = (0.0, 1.0, 0.25)
    return source


def render_tiled_native_chain(
    *,
    domains_dll: Path,
    gaussian_dll: Path,
    adjacency_dll: Path,
    domains_payload: dict[str, Any],
    spatial_payload: dict[str, Any],
    adjacency_payload: dict[str, Any],
    source: np.ndarray,
    tile_rows: int,
    reverse: bool,
) -> np.ndarray:
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
    ):
        raise ValueError("tile_rows must be a positive integer")
    height = source.shape[0]
    halo = sum(
        int(row["maximum_radius"]) for row in spatial_payload["stages"]
    )
    starts = list(range(0, height, tile_rows))
    if reverse:
        starts.reverse()
    output = np.empty_like(source)
    for y0 in starts:
        y1 = min(height, y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(height, y1 + halo)
        rendered = render_loaded_ordered_chain(
            domains_dll=domains_dll,
            gaussian_dll=gaussian_dll,
            adjacency_dll=adjacency_dll,
            domains_payload=domains_payload,
            spatial_payload=spatial_payload,
            adjacency_payload=adjacency_payload,
            source=source[source_y0:source_y1],
        )["scanner_mtf"]
        output[y0:y1] = rendered[
            y0 - source_y0 : y1 - source_y0
        ]
    return output


def run_native_tiled_chain_conformance(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AB"):
        raise ValueError("P8AB parent decision drift")
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    expected_halo = sum(
        int(row["maximum_radius"]) for row in spatial_payload["stages"]
    )
    if expected_halo != int(config["expected_total_halo"]):
        raise ValueError("P8AB total halo drift")
    builds = {
        "domains": build_msvc_native_domains_dll(
            root=root, output_dir=output_dir / "domains"
        ),
        "gaussian": build_msvc_native_gaussian_dll(
            root=root, output_dir=output_dir / "gaussian"
        ),
        "adjacency": build_msvc_native_adjacency_dll(
            root=root, output_dir=output_dir / "adjacency"
        ),
    }
    for name, build in builds.items():
        expected = config["component_dll_sha256"][name]
        if build["dll_sha256"] != expected:
            raise ValueError(f"P8AB {name} DLL identity drift")
    source = build_tiled_source(config)
    full = render_loaded_ordered_chain(
        domains_dll=Path(builds["domains"]["dll_path"]),
        gaussian_dll=Path(builds["gaussian"]["dll_path"]),
        adjacency_dll=Path(builds["adjacency"]["dll_path"]),
        domains_payload=domains_payload,
        spatial_payload=spatial_payload,
        adjacency_payload=adjacency_payload,
        source=source,
    )["scanner_mtf"]
    full_sha = hashlib.sha256(full.tobytes()).hexdigest()
    if full_sha != config["expected_full_output_sha256"]:
        raise ValueError("P8AB full native output identity drift")
    rows = []
    for tile_rows in config["tile_rows"]:
        for order in ("forward", "reverse"):
            tiled = render_tiled_native_chain(
                domains_dll=Path(builds["domains"]["dll_path"]),
                gaussian_dll=Path(builds["gaussian"]["dll_path"]),
                adjacency_dll=Path(builds["adjacency"]["dll_path"]),
                domains_payload=domains_payload,
                spatial_payload=spatial_payload,
                adjacency_payload=adjacency_payload,
                source=source,
                tile_rows=int(tile_rows),
                reverse=order == "reverse",
            )
            maximum_error = float(np.max(np.abs(tiled - full)))
            exact = tiled.tobytes() == full.tobytes()
            if not exact:
                raise RuntimeError(
                    f"P8AB tile_rows={tile_rows} {order} is not exact; "
                    f"maximum error {maximum_error}"
                )
            rows.append(
                {
                    "tile_rows": int(tile_rows),
                    "order": order,
                    "byte_exact": True,
                    "maximum_absolute_error": maximum_error,
                    "output_sha256": hashlib.sha256(
                        tiled.tobytes()
                    ).hexdigest(),
                }
            )
    stable_core = {
        "schema": "neuro_film.u6_p8ab_native_tiled_chain.v1",
        "shape": list(source.shape),
        "total_halo": expected_halo,
        "full_output_sha256": full_sha,
        "rows": rows,
        "decision": (
            "pass halo-tiled scanner-linear native chain parity for all "
            "frozen row sizes and both execution orders; full-resolution "
            "resources and post-scan display stages remain open"
        ),
        "claim_ceiling": (
            "finite small-image halo-tiled native scanner-linear parity; "
            "not full-resolution performance, display look, calibration "
            "or product runtime"
        ),
        "production_default_changed": False,
    }
    return {
        **stable_core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_core)
        ).hexdigest(),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
