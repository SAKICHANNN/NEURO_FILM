"""End-to-end native Standard physical plus AO6 display conformance."""

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
from src.eval.physical_native_ao6_context_conformance import (
    _load_context,
    _native_context,
    build_msvc_native_ao6_context_dll,
)
from src.eval.physical_native_ao6_display_conformance import (
    _load_display,
    build_msvc_native_ao6_display_dll,
)
from src.eval.physical_native_composed_display_conformance import (
    _apply_gauge,
    _build_all,
)
from src.eval.physical_native_f32_conformance import (
    _render_f32_chain,
    render_tiled_f32_chain,
)
from src.film_physics.display_look import (
    build_source_context_display_look,
)
from src.film_physics.native_adjacency_profile import (
    compile_native_adjacency_profile_payload,
)
from src.film_physics.native_ao6_base_profile import (
    native_ao6_base_display_payload_sha256,
    native_ao6_base_profile_struct,
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


def _apply_display_rows(
    *,
    library: ctypes.CDLL,
    base_profile: Any,
    context: Any,
    residual_profile: Any,
    encoded: np.ndarray,
    tile_rows: int,
    reverse: bool,
) -> np.ndarray:
    height, width, _ = encoded.shape
    output = np.empty_like(encoded)
    starts = list(range(0, height, tile_rows))
    if reverse:
        starts.reverse()
    for y0 in starts:
        y1 = min(height, y0 + tile_rows)
        input_rows = np.ascontiguousarray(
            encoded[y0:y1].reshape(-1, 3)
        )
        scratch = np.empty_like(input_rows)
        output_rows = np.empty_like(input_rows)
        status = library.nf_ao6_display_f32_apply_v1(
            ctypes.byref(base_profile),
            ctypes.byref(context),
            ctypes.byref(residual_profile),
            input_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            input_rows.shape[0],
            scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status != 0:
            raise RuntimeError(
                f"P8AM native display tile failed: {status}"
            )
        output[y0:y1] = output_rows.reshape(y1 - y0, width, 3)
    return output


def run_native_end_to_end_conformance(
    *, root: Path, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AM"):
        raise ValueError("P8AM parent decision drift")
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
    if (
        native_ao6_base_display_payload_sha256(display_payload)
        != config["expected_display_payload_sha256"]
    ):
        raise ValueError("P8AM display payload drift")
    base_profile = native_ao6_base_profile_struct(display_payload)
    residual_profile = native_ao6_residual_profile_struct(display_payload)

    builds = _build_all(root=root, output_dir=output_dir / "physical")
    context_build = build_msvc_native_ao6_context_dll(
        root=root, output_dir=output_dir / "context"
    )
    display_build = build_msvc_native_ao6_display_dll(
        root=root, output_dir=output_dir / "display"
    )
    paths = {
        name: Path(build["dll_path"]) for name, build in builds.items()
    }
    context_library = _load_context(Path(context_build["dll_path"]))
    display_library = _load_display(Path(display_build["dll_path"]))

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
        encoded_srgb_to_linear(source_encoded.astype(np.float64)),
        dtype=np.float32,
    )
    context, context_sha = _native_context(
        library=context_library,
        profile=base_profile,
        source=source_encoded,
        tile_rows=7,
    )
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
    gauged_encoded = np.ascontiguousarray(
        linear_srgb_to_encoded(gauged.astype(np.float64)),
        dtype=np.float32,
    )
    native_output = _apply_display_rows(
        library=display_library,
        base_profile=base_profile,
        context=context,
        residual_profile=residual_profile,
        encoded=gauged_encoded,
        tile_rows=height,
        reverse=False,
    )
    python_output = np.asarray(
        build_source_context_display_look(
            display_payload, source_encoded
        )(gauged_encoded),
        dtype=np.float32,
    )
    difference = np.abs(
        native_output.astype(np.float64) -
        python_output.astype(np.float64)
    )
    maximum_error = float(np.max(difference))
    if maximum_error > float(config["gates"]["maximum_output_error"]):
        raise RuntimeError(
            f"P8AM Python-reference error exceeds gate: {maximum_error}"
        )

    partition_rows = []
    for tile_rows in config["tile_rows"]:
        for order in ("forward", "reverse"):
            reverse = order == "reverse"
            tiled_physical = render_tiled_f32_chain(
                domains_dll=paths["domains"],
                gaussian_dll=paths["gaussian"],
                adjacency_dll=paths["adjacency"],
                domains_payload=domains_payload,
                spatial_payload=spatial_payload,
                adjacency_payload=adjacency_payload,
                source=source_linear,
                tile_rows=int(tile_rows),
                reverse=reverse,
            )
            tiled_gauge = np.empty_like(gauged)
            starts = list(range(0, height, int(tile_rows)))
            if reverse:
                starts.reverse()
            for y0 in starts:
                y1 = min(height, y0 + int(tile_rows))
                tiled_gauge[y0:y1] = _apply_gauge(
                    dll=paths["gauge"],
                    profile=gauge_profile,
                    values=tiled_physical[y0:y1],
                )
            tiled_encoded = np.ascontiguousarray(
                linear_srgb_to_encoded(
                    tiled_gauge.astype(np.float64)
                ),
                dtype=np.float32,
            )
            tiled_output = _apply_display_rows(
                library=display_library,
                base_profile=base_profile,
                context=context,
                residual_profile=residual_profile,
                encoded=tiled_encoded,
                tile_rows=int(tile_rows),
                reverse=reverse,
            )
            partition_rows.append(
                {
                    "tile_rows": int(tile_rows),
                    "order": order,
                    "physical_byte_exact": (
                        tiled_physical.tobytes() == physical.tobytes()
                    ),
                    "gauge_byte_exact": (
                        tiled_gauge.tobytes() == gauged.tobytes()
                    ),
                    "display_byte_exact": (
                        tiled_output.tobytes() == native_output.tobytes()
                    ),
                }
            )
    if not all(
        row["physical_byte_exact"]
        and row["gauge_byte_exact"]
        and row["display_byte_exact"]
        for row in partition_rows
    ):
        raise RuntimeError("P8AM partition parity failed")

    stage_sha256 = {
        "source_encoded": hashlib.sha256(
            source_encoded.tobytes()
        ).hexdigest(),
        "source_linear": hashlib.sha256(
            source_linear.tobytes()
        ).hexdigest(),
        "physical_scanner_linear": hashlib.sha256(
            physical.tobytes()
        ).hexdigest(),
        "neutral_gauged_linear": hashlib.sha256(
            gauged.tobytes()
        ).hexdigest(),
        "neutral_gauged_encoded": hashlib.sha256(
            gauged_encoded.tobytes()
        ).hexdigest(),
        "native_display_output": hashlib.sha256(
            native_output.tobytes()
        ).hexdigest(),
        "python_display_output": hashlib.sha256(
            python_output.tobytes()
        ).hexdigest(),
    }
    stable = {
        "schema": "neuro_film.u6_p8am_native_end_to_end.v1",
        "shape": [height, width, 3],
        "stage_sha256": stage_sha256,
        "context_sha256": context_sha,
        "component_dll_sha256": {
            **{
                name: build["dll_sha256"]
                for name, build in builds.items()
            },
            "context": context_build["dll_sha256"],
            "display": display_build["dll_sha256"],
        },
        "partition_rows": partition_rows,
        "all_stage_partition_rows_byte_exact": True,
        "maximum_output_error_vs_python_reference": maximum_error,
        "p999_output_error_vs_python_reference": float(
            np.quantile(difference, 0.999)
        ),
        "no_double_count_receipts": config[
            "no_double_count_receipts"
        ],
        "decision": (
            "pass the end-to-end native Standard physical, neutral-gauge "
            "and complete AO6 display-look composition; every stage is "
            "partition exact and the downstream AO6 component remains a "
            "separately identified look approximation"
        ),
        "claim_ceiling": config["claim_ceiling"],
        "production_default_changed": False,
        "next_leaf": (
            "U6.P8AN measure 12MP row-streamed end-to-end CPU latency and "
            "memory, then compare against the provisional Standard targets"
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
