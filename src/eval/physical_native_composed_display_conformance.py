"""Compose native physical, gauge and AO6 residual around an external base."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.physical_native_ao6_residual_conformance import (
    _load as _load_residual,
    build_msvc_native_ao6_residual_dll,
)
from src.eval.physical_native_f32_conformance import (
    _build_component,
    _render_f32_chain,
    render_tiled_f32_chain,
)
from src.eval.physical_native_gauge_conformance import (
    _load_gauge,
    build_msvc_native_gauge_dll,
)
from src.film_physics.display_look import (
    build_source_context_display_look_stages,
)
from src.film_physics.native_adjacency_profile import (
    compile_native_adjacency_profile_payload,
)
from src.film_physics.native_ao6_residual_profile import (
    native_ao6_residual_profile_struct,
)
from src.film_physics.native_gauge_profile import (
    native_gauge_profile_struct,
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


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _build_all(
    *,
    root: Path,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    components = {
        "domains": (
            "native/film_physics/nf_physical_domains_f32_v1.c",
            "native/film_physics/nf_physical_domains_f32_v1.h",
            "nf_physical_domains_f32_v1",
        ),
        "gaussian": (
            "native/film_physics/nf_gaussian_rgb_f32_v1.c",
            "native/film_physics/nf_gaussian_rgb_f32_v1.h",
            "nf_gaussian_rgb_f32_v1",
        ),
        "adjacency": (
            "native/film_physics/nf_bounded_adjacency_f32_v1.c",
            "native/film_physics/nf_bounded_adjacency_f32_v1.h",
            "nf_bounded_adjacency_f32_v1",
        ),
    }
    builds = {}
    for name, (source, header, basename) in components.items():
        builds[name] = _build_component(
            root=root,
            output_dir=output_dir / name,
            component={"source": source, "header": header},
            basename=basename,
        )
    builds["gauge"] = build_msvc_native_gauge_dll(
        root=root, output_dir=output_dir / "gauge"
    )
    builds["residual"] = build_msvc_native_ao6_residual_dll(
        root=root, output_dir=output_dir / "residual"
    )
    return builds


def _apply_gauge(
    *,
    dll: Path,
    profile: Any,
    values: np.ndarray,
) -> np.ndarray:
    library = _load_gauge(dll)
    source = np.ascontiguousarray(values, dtype=np.float32)
    output = np.empty_like(source)
    status = library.nf_neutral_gauge_f32_apply_v1(
        ctypes.byref(profile),
        _pointer(source),
        source.shape[0] * source.shape[1],
        _pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"native composed gauge failed: {status}")
    return output


def _apply_residual(
    *,
    dll: Path,
    profile: Any,
    values: np.ndarray,
) -> np.ndarray:
    library = _load_residual(dll)
    source = np.ascontiguousarray(values, dtype=np.float32)
    output = np.empty_like(source)
    status = library.nf_ao6_residual_f32_apply_v1(
        ctypes.byref(profile),
        _pointer(source),
        source.shape[0] * source.shape[1],
        _pointer(output),
        None,
        None,
    )
    if status != 0:
        raise RuntimeError(f"native composed residual failed: {status}")
    return output


def build_composed_fixture(
    *,
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    compiler_config = _load_exact_json(
        root,
        config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    artifact = compile_standalone_profile_artifact(
        root=root, config=compiler_config
    )
    payloads = artifact["component_payloads"]
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    gauge_profile = native_gauge_profile_struct(
        payloads["neutral-axis-gauge"]
    )
    display_payload = payloads["ao6-source-context-display-look"]
    residual_profile = native_ao6_residual_profile_struct(
        display_payload
    )
    builds = _build_all(root=root, output_dir=output_dir)

    fixture = config["fixture"]
    height = int(fixture["height"])
    width = int(fixture["width"])
    random = np.random.default_rng(int(fixture["seed"]))
    source_encoded = random.random(
        (height, width, 3), dtype=np.float32
    )
    source_encoded[0, 0] = (0.0, 0.0, 0.0)
    source_encoded[0, 1] = (1.0, 1.0, 1.0)
    source_encoded[0, 2] = (0.0, 0.5, 1.0)
    source_linear = np.ascontiguousarray(
        encoded_srgb_to_linear(
            source_encoded.astype(np.float64)
        ),
        dtype=np.float32,
    )
    paths = {
        name: Path(build["dll_path"]) for name, build in builds.items()
    }
    full_stages, _ = _render_f32_chain(
        domains_dll=paths["domains"],
        gaussian_dll=paths["gaussian"],
        adjacency_dll=paths["adjacency"],
        domains_payload=domains_payload,
        spatial_payload=spatial_payload,
        adjacency_payload=adjacency_payload,
        source=source_linear,
        audit_failure_atomic=False,
    )
    physical = full_stages["scanner_mtf"]
    gauged = _apply_gauge(
        dll=paths["gauge"],
        profile=gauge_profile,
        values=physical,
    )
    gauged_encoded = linear_srgb_to_encoded(
        gauged.astype(np.float64)
    )
    apply_base, apply_python_residual = (
        build_source_context_display_look_stages(
            display_payload, source_encoded
        )
    )
    base_encoded = np.ascontiguousarray(
        apply_base(gauged_encoded), dtype=np.float32
    )
    base_linear = np.ascontiguousarray(
        encoded_srgb_to_linear(base_encoded.astype(np.float64)),
        dtype=np.float32,
    )
    native_residual_linear = _apply_residual(
        dll=paths["residual"],
        profile=residual_profile,
        values=base_linear,
    )
    native_output_encoded = linear_srgb_to_encoded(
        native_residual_linear.astype(np.float64)
    )
    python_output_encoded = apply_python_residual(base_encoded)

    tile_rows_values = [int(value) for value in config["tile_rows"]]
    partition_rows = []
    for tile_rows in tile_rows_values:
        for order in ("forward", "reverse"):
            tiled_physical = render_tiled_f32_chain(
                domains_dll=paths["domains"],
                gaussian_dll=paths["gaussian"],
                adjacency_dll=paths["adjacency"],
                domains_payload=domains_payload,
                spatial_payload=spatial_payload,
                adjacency_payload=adjacency_payload,
                source=source_linear,
                tile_rows=tile_rows,
                reverse=order == "reverse",
            )
            tiled_gauge = np.empty_like(gauged)
            starts = list(range(0, height, tile_rows))
            if order == "reverse":
                starts.reverse()
            for y0 in starts:
                y1 = min(height, y0 + tile_rows)
                tiled_gauge[y0:y1] = _apply_gauge(
                    dll=paths["gauge"],
                    profile=gauge_profile,
                    values=tiled_physical[y0:y1],
                )
            tiled_gauged_encoded = linear_srgb_to_encoded(
                tiled_gauge.astype(np.float64)
            )
            tiled_base = np.ascontiguousarray(
                apply_base(tiled_gauged_encoded), dtype=np.float32
            )
            tiled_base_linear = np.ascontiguousarray(
                encoded_srgb_to_linear(
                    tiled_base.astype(np.float64)
                ),
                dtype=np.float32,
            )
            tiled_residual = np.empty_like(native_residual_linear)
            for y0 in starts:
                y1 = min(height, y0 + tile_rows)
                tiled_residual[y0:y1] = _apply_residual(
                    dll=paths["residual"],
                    profile=residual_profile,
                    values=tiled_base_linear[y0:y1],
                )
            partition_rows.append(
                {
                    "tile_rows": tile_rows,
                    "order": order,
                    "physical_byte_exact": (
                        tiled_physical.tobytes() == physical.tobytes()
                    ),
                    "gauge_byte_exact": (
                        tiled_gauge.tobytes() == gauged.tobytes()
                    ),
                    "external_base_byte_exact": (
                        tiled_base.tobytes() == base_encoded.tobytes()
                    ),
                    "residual_byte_exact": (
                        tiled_residual.tobytes()
                        == native_residual_linear.tobytes()
                    ),
                }
            )
    component_payload_sha256 = {
        name: hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        for name, payload in payloads.items()
    }
    return {
        "source_encoded_sha256": hashlib.sha256(
            source_encoded.tobytes()
        ).hexdigest(),
        "physical_scanner_linear_sha256": hashlib.sha256(
            physical.tobytes()
        ).hexdigest(),
        "neutral_gauged_linear_sha256": hashlib.sha256(
            gauged.tobytes()
        ).hexdigest(),
        "external_base_encoded_sha256": hashlib.sha256(
            base_encoded.tobytes()
        ).hexdigest(),
        "native_residual_linear_sha256": hashlib.sha256(
            native_residual_linear.tobytes()
        ).hexdigest(),
        "native_output_encoded_sha256": hashlib.sha256(
            np.ascontiguousarray(native_output_encoded).tobytes()
        ).hexdigest(),
        "maximum_encoded_error_vs_python_residual": float(
            np.max(
                np.abs(
                    native_output_encoded.astype(np.float64)
                    - python_output_encoded.astype(np.float64)
                )
            )
        ),
        "partition_rows": partition_rows,
        "component_dll_sha256": {
            name: build["dll_sha256"] for name, build in builds.items()
        },
        "component_payload_sha256": component_payload_sha256,
        "shape": [height, width, 3],
    }


def run_native_composed_display_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AI"):
        raise ValueError("P8AI parent decision drift")
    fixture = build_composed_fixture(
        root=root, config=config, output_dir=output_dir
    )
    expected = config["expected_fixture"]
    for key in (
        "source_encoded_sha256",
        "physical_scanner_linear_sha256",
        "neutral_gauged_linear_sha256",
        "external_base_encoded_sha256",
        "native_residual_linear_sha256",
        "native_output_encoded_sha256",
    ):
        if fixture[key] != expected[key]:
            raise ValueError(f"P8AI fixture identity drift: {key}")
    if not all(
        all(
            row[key]
            for key in (
                "physical_byte_exact",
                "gauge_byte_exact",
                "external_base_byte_exact",
                "residual_byte_exact",
            )
        )
        for row in fixture["partition_rows"]
    ):
        raise RuntimeError("P8AI partition parity failed")
    tolerance = float(config["gates"]["maximum_encoded_error"])
    if fixture["maximum_encoded_error_vs_python_residual"] > tolerance:
        raise RuntimeError("P8AI Python residual parity failed")
    stable = {
        "schema": "neuro_film.u6_p8ai_native_composed_display.v1",
        "fixture": fixture,
        "maximum_encoded_error_tolerance": tolerance,
        "stage_ownership": config["stage_ownership"],
        "claim_ledger": config["claim_ledger"],
        "decision": (
            "pass the ordered native Standard physical, neutral-gauge and "
            "fixed AO6 residual composition around one frozen external "
            "source-context base fixture; the base itself remains Python "
            "reference code and no product path opens"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AJ freeze a native source-context base representation "
            "candidate or formally retain the Python reference boundary "
            "after a complexity and product-value audit"
        ),
    }
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
