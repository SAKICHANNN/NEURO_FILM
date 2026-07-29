"""Artifact-only canonical CPU consumer for the U6.P8 profile bundle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.eval.global_frontier import sha256_file
from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.physical_neutral_gauged_chain import (
    apply_gauge_to_intermediate,
)
from src.eval.physical_virtual_scan_sampling import (
    _render_physical,
    compile_virtual_scan_profile,
)
from src.eval.physical_neutral_gauged_invariance import (
    _bounded_real_source,
    render_challenger,
    render_challenger_row_tiled,
)
from src.film_physics.contracts import (
    ComponentBinding,
    FilmProfileBundle,
    scene_exposure_from_working_image,
)
from src.film_physics.display_look import (
    DISPLAY_LOOK_SCHEMA,
    build_density_source_context_inplace_packed_lab,
    build_density_source_context_row_staged,
    build_source_context_display_look,
    build_source_context_display_look_row_streamed,
    validate_display_look_payload,
)
from src.film_physics.spatial_response import required_spatial_response_halo
from src.film_physics.profile_compiler import (
    _adjacency_from_payload,
    _canonical_bytes,
    _payload_sha256,
    _spatial_from_payload,
    compile_profile_artifact,
    validate_contract as validate_p8a_contract,
    validate_profile_artifact as validate_p8a_artifact,
)
from src.roll2film.sensitometry_gauge import NeutralAxisGaugeOperator
from src.roll2film.sensitometry_print import SensitometryPrintOperator
from src.roll2film.splines import RationalQuadraticSpline
from src.preprocess.types import WorkingImage


SCHEMA = "neuro_film.u6_p8b_artifact_only_cpu_consumer_contract.v1"
ARTIFACT_SCHEMA = (
    "neuro_film.u6_p8b_artifact_only_cpu_profile_artifact.v1"
)
MAX_ARTIFACT_BYTES = 1_048_576


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _linear_srgb_to_encoded_row_staged(
    linear: np.ndarray, *, tile_rows: int
) -> np.ndarray:
    """Apply the exact pointwise OETF without a full-frame float64 input cast."""

    value = np.asarray(linear)
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or value.shape[0] == 0
        or value.shape[1] == 0
        or isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
        or not np.all(np.isfinite(value))
    ):
        raise ValueError("row-staged OETF requires finite HxWx3 input")
    encoded = np.empty(value.shape, dtype=np.float64)
    for y0 in range(0, value.shape[0], tile_rows):
        y1 = min(value.shape[0], y0 + tile_rows)
        encoded[y0:y1] = linear_srgb_to_encoded(
            value[y0:y1].astype(np.float64)
        )
    return encoded


def _encoded_srgb_to_linear_row_staged(
    encoded: np.ndarray, *, tile_rows: int
) -> np.ndarray:
    """Apply the exact pointwise EOTF without a full-frame output temporary."""

    value = np.asarray(encoded)
    if (
        value.ndim != 3
        or value.shape[-1] != 3
        or value.shape[0] == 0
        or value.shape[1] == 0
        or isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
        or not np.all(np.isfinite(value))
    ):
        raise ValueError("row-staged EOTF requires finite HxWx3 input")
    linear = np.empty(value.shape, dtype=np.float64)
    for y0 in range(0, value.shape[0], tile_rows):
        y1 = min(value.shape[0], y0 + tile_rows)
        linear[y0:y1] = encoded_srgb_to_linear(value[y0:y1])
    return linear


def _encoded_srgb_to_linear_inplace_row_staged(
    encoded: np.ndarray, *, tile_rows: int
) -> np.ndarray:
    """Replace one writable float64 encoded buffer with its exact EOTF."""

    value = np.asarray(encoded)
    if (
        value.dtype != np.float64
        or value.ndim != 3
        or value.shape[-1] != 3
        or value.shape[0] == 0
        or value.shape[1] == 0
        or not value.flags.c_contiguous
        or not value.flags.writeable
        or isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
        or not np.all(np.isfinite(value))
    ):
        raise ValueError(
            "in-place row EOTF requires writable C-contiguous float64 HxWx3"
        )
    for y0 in range(0, value.shape[0], tile_rows):
        y1 = min(value.shape[0], y0 + tile_rows)
        value[y0:y1] = encoded_srgb_to_linear(value[y0:y1])
    return value


def _refill_roundtrip_linear_inplace_row_staged(
    consumed: np.ndarray,
    original_linear: np.ndarray,
    *,
    tile_rows: int,
) -> np.ndarray:
    """Refill consumed float64 storage with the exact OETF-to-EOTF roundtrip."""

    output = np.asarray(consumed)
    source = np.asarray(original_linear)
    if (
        output.dtype != np.float64
        or output.ndim != 3
        or output.shape[-1] != 3
        or output.shape != source.shape
        or not output.flags.c_contiguous
        or not output.flags.writeable
        or isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
        or not np.all(np.isfinite(source))
    ):
        raise ValueError(
            "roundtrip refill requires writable float64 output and matching input"
        )
    for y0 in range(0, output.shape[0], tile_rows):
        y1 = min(output.shape[0], y0 + tile_rows)
        encoded = linear_srgb_to_encoded(
            source[y0:y1].astype(np.float64)
        )
        output[y0:y1] = encoded_srgb_to_linear(encoded)
    return output


def _render_physical_gauged_encoded_inplace(
    linear: np.ndarray,
    *,
    ranges: list[tuple[int, int]],
    order: str,
    halo: int,
    runtime: Any,
    gauge: NeutralAxisGaugeOperator,
) -> np.ndarray:
    """Consume linear rows into exact encoded output with one halo carry."""

    value = np.asarray(linear)
    if (
        value.dtype != np.float64
        or value.ndim != 3
        or value.shape[-1] != 3
        or not value.flags.c_contiguous
        or not value.flags.writeable
        or order not in {"forward", "reverse"}
        or isinstance(halo, bool)
        or not isinstance(halo, int)
        or halo < 0
    ):
        raise ValueError("invalid in-place physical output buffer")
    height = value.shape[0]
    saved_overlap: np.ndarray | None = None
    for y0, y1 in ranges:
        source_y0 = max(0, y0 - halo)
        source_y1 = min(height, y1 + halo)
        source_tile = np.empty_like(value[source_y0:source_y1])
        if order == "forward":
            prefix_rows = y0 - source_y0
            if prefix_rows:
                if (
                    saved_overlap is None
                    or saved_overlap.shape[0] != prefix_rows
                ):
                    raise RuntimeError("forward halo carry mismatch")
                source_tile[:prefix_rows] = saved_overlap
            source_tile[prefix_rows:] = value[y0:source_y1]
            if y1 < height:
                next_source_y0 = max(0, y1 - halo)
                saved_overlap = source_tile[
                    next_source_y0 - source_y0 : y1 - source_y0
                ].copy()
            else:
                saved_overlap = None
        else:
            suffix_rows = source_y1 - y1
            source_tile[: y1 - source_y0] = value[source_y0:y1]
            if suffix_rows:
                if (
                    saved_overlap is None
                    or saved_overlap.shape[0] != suffix_rows
                ):
                    raise RuntimeError("reverse halo carry mismatch")
                source_tile[-suffix_rows:] = saved_overlap
            if y0 > 0:
                next_source_y1 = min(height, y0 + halo)
                saved_overlap = source_tile[
                    y0 - source_y0 : next_source_y1 - source_y0
                ].copy()
            else:
                saved_overlap = None
        physical = _render_physical(source_tile, runtime)
        core = physical[y0 - source_y0 : y1 - source_y0]
        gauged = apply_gauge_to_intermediate(core, gauge)
        value[y0:y1] = linear_srgb_to_encoded(gauged)
    if saved_overlap is not None:
        raise RuntimeError("unconsumed physical halo carry")
    return value


@dataclass(frozen=True)
class CompiledProfileRuntime:
    """Minimum runtime surface consumed by the frozen challenger renderer."""

    profile: Any
    print_operator: SensitometryPrintOperator
    apply_adjacency: Callable[..., np.ndarray]
    build_source_context_colour: Callable[
        [np.ndarray], Callable[[np.ndarray], np.ndarray]
    ]


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], Any, NeutralAxisGaugeOperator]:
    if (
        config.get("schema") != SCHEMA
        or not config["execution"].get(
            "artifact_only_reconstruction_required"
        )
        or config["execution"].get(
            "experiment_config_reads_after_compile_allowed"
        )
        or config["execution"].get("post_result_retuning_allowed")
        or int(config["reference_sampling_dpi"]) != 4000
    ):
        raise ValueError("unsupported U6.P8B contract")
    decision = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    parent = _load_exact_json(
        root,
        config["parent_contract"],
        config["parent_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8B")
        or decision["artifact_canonical_sha256"]
        != config["required_parent_artifact_sha256"]
        or decision["production_default_changed"]
        or decision["calibration_claim_opened"]
    ):
        raise ValueError("U6.P8B parent decision drift")
    runtime, gauge = validate_p8a_contract(root, parent)
    return parent, runtime, gauge


def compile_standalone_profile_artifact(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    parent, runtime, _ = validate_contract(root, config)
    p8a = compile_profile_artifact(root=root, config=parent)
    if (
        _payload_sha256(p8a)
        != config["required_parent_artifact_sha256"]
    ):
        raise ValueError("U6.P8A compiled artifact drift")
    validate_p8a_artifact(p8a)
    payloads = dict(p8a["component_payloads"])
    payloads["ao6-source-context-display-look"] = (
        runtime.display_look_payload
    )
    validate_display_look_payload(
        payloads["ao6-source-context-display-look"]
    )
    components = []
    for component in FilmProfileBundle.from_dict(
        p8a["film_profile_bundle"]
    ).components:
        if component.component_id == "ao6-source-context-display-look":
            component = ComponentBinding(
                component_id=component.component_id,
                schema=DISPLAY_LOOK_SCHEMA,
                sha256=_payload_sha256(
                    payloads[component.component_id]
                ),
                input_domain=component.input_domain,
                output_domain=component.output_domain,
            )
        components.append(component)
    identity = _payload_sha256(
        {
            "components": {
                item.component_id: item.sha256 for item in components
            },
            "parent_artifact_sha256": (
                config["required_parent_artifact_sha256"]
            ),
            "reference_sampling_dpi": 4000,
        }
    )
    bundle = FilmProfileBundle(
        profile_id=f"generic-p8b-4000-{identity[:16]}",
        claim_level="generic-physical-inspired",
        stock_id="unknown",
        process_id="unknown",
        scanner_profile_id="unknown",
        evidence_manifest_sha256=config["parent_decision_sha256"],
        components=tuple(components),
    )
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "film_profile_bundle": bundle.to_dict(),
        "bundle_sha256": bundle.bundle_sha256,
        "parent_artifact_sha256": config[
            "required_parent_artifact_sha256"
        ],
        "reference_sampling_dpi": 4000,
        "preview_policy": p8a["preview_policy"],
        "component_payloads": payloads,
        "claim_ceiling": config["claim_ceiling"],
    }
    validate_standalone_profile_artifact(artifact)
    return artifact


def validate_standalone_profile_artifact(
    artifact: dict[str, Any]
) -> FilmProfileBundle:
    if set(artifact) != {
        "schema",
        "film_profile_bundle",
        "bundle_sha256",
        "parent_artifact_sha256",
        "reference_sampling_dpi",
        "preview_policy",
        "component_payloads",
        "claim_ceiling",
    } or artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("standalone profile artifact fields drift")
    bundle = FilmProfileBundle.from_dict(
        artifact["film_profile_bundle"]
    )
    if (
        bundle.bundle_sha256 != artifact["bundle_sha256"]
        or bundle.claim_level != "generic-physical-inspired"
        or {
            bundle.stock_id,
            bundle.process_id,
            bundle.scanner_profile_id,
        }
        != {"unknown"}
        or int(artifact["reference_sampling_dpi"]) != 4000
        or artifact["preview_policy"].get(
            "independent_low_resolution_equivalence_claim_allowed"
        )
    ):
        raise ValueError("standalone profile identity or claim drift")
    payloads = artifact["component_payloads"]
    if not isinstance(payloads, dict) or set(payloads) != {
        component.component_id for component in bundle.components
    }:
        raise ValueError("standalone component inventory drift")
    for component in bundle.components:
        payload = payloads[component.component_id]
        if _payload_sha256(payload) != component.sha256:
            raise ValueError("standalone component hash drift")
    physical = payloads["physical-chain-4000dpi"]
    _spatial_from_payload(physical["spatial_profile"])
    SensitometryPrintOperator.from_dict(physical["print_operator"])
    _adjacency_from_payload(physical["adjacency"])
    gauge = payloads["neutral-axis-gauge"]
    if len(gauge["inverse_neutral_splines"]) != 3:
        raise ValueError("standalone gauge drift")
    tuple(
        RationalQuadraticSpline.from_dict(item)
        for item in gauge["inverse_neutral_splines"]
    )
    validate_display_look_payload(
        payloads["ao6-source-context-display-look"]
    )
    return bundle


def reconstruct_standalone_runtime(
    artifact: dict[str, Any],
) -> tuple[CompiledProfileRuntime, NeutralAxisGaugeOperator]:
    validate_standalone_profile_artifact(artifact)
    payloads = artifact["component_payloads"]
    physical = payloads["physical-chain-4000dpi"]
    print_operator = SensitometryPrintOperator.from_dict(
        physical["print_operator"]
    )
    display_payload = payloads["ao6-source-context-display-look"]
    runtime = CompiledProfileRuntime(
        profile=_spatial_from_payload(physical["spatial_profile"]),
        print_operator=print_operator,
        apply_adjacency=_adjacency_from_payload(physical["adjacency"]),
        build_source_context_colour=partial(
            build_source_context_display_look, display_payload
        ),
    )
    gauge = NeutralAxisGaugeOperator(
        print_operator,
        tuple(
            RationalQuadraticSpline.from_dict(item)
            for item in payloads["neutral-axis-gauge"][
                "inverse_neutral_splines"
            ]
        ),
    )
    return runtime, gauge


def render_standalone_profile(
    artifact: dict[str, Any], source: np.ndarray
) -> np.ndarray:
    runtime, gauge = reconstruct_standalone_runtime(artifact)
    return render_challenger(
        source,
        runtime,
        gauge,
        sampling_dpi=int(artifact["reference_sampling_dpi"]),
    )


def serialize_standalone_profile_artifact(
    artifact: dict[str, Any],
) -> bytes:
    validate_standalone_profile_artifact(artifact)
    return _canonical_bytes(artifact)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate artifact key: {key}")
        result[key] = value
    return result


def _validate_json_tree(value: Any, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("artifact JSON nesting exceeds 64")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("artifact JSON contains a non-finite number")
    if isinstance(value, str) and any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise ValueError("artifact JSON contains a lone surrogate")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_tree(key, depth=depth + 1)
            _validate_json_tree(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_tree(item, depth=depth + 1)


def load_standalone_profile_artifact_bytes(
    raw: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    if (
        not isinstance(raw, bytes)
        or len(raw) == 0
        or len(raw) > MAX_ARTIFACT_BYTES
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("artifact byte identity or size mismatch")
    try:
        artifact = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("artifact is not strict ASCII JSON") from exc
    if not isinstance(artifact, dict):
        raise ValueError("artifact root must be an object")
    _validate_json_tree(artifact)
    validate_standalone_profile_artifact(artifact)
    if _canonical_bytes(artifact) != raw:
        raise ValueError("artifact bytes are not canonical")
    return artifact


def render_working_image(
    artifact: dict[str, Any], working: WorkingImage
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one bounded scene-linear WorkingImage and return a receipt."""

    bundle = validate_standalone_profile_artifact(artifact)
    scene = scene_exposure_from_working_image(working)
    if np.any(scene.values > 1.0):
        raise ValueError(
            "v1 profile consumer requires scene-linear samples in [0,1]; "
            "a calibrated scene-to-relative exposure map is not available"
        )
    encoded = linear_srgb_to_encoded(
        scene.values.astype(np.float64)
    )
    output = render_standalone_profile(artifact, encoded)
    input_bytes = np.ascontiguousarray(
        scene.values.astype(np.float32, copy=False)
    ).tobytes()
    output_bytes = np.ascontiguousarray(output).tobytes()
    receipt_core = {
        "schema": "neuro_film.physical_profile_render_receipt.v1",
        "profile_id": bundle.profile_id,
        "bundle_sha256": bundle.bundle_sha256,
        "artifact_sha256": _payload_sha256(artifact),
        "reference_sampling_dpi": int(
            artifact["reference_sampling_dpi"]
        ),
        "input": {
            "array_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "dtype": "float32",
            "shape": list(scene.values.shape),
            "working_space": working.working_space,
            "transfer_state": working.transfer_state,
            "source_transfer_state": working.source_transfer_state,
        },
        "output": {
            "array_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "dtype": output.dtype.name,
            "shape": list(output.shape),
            "domain": "display-encoded-rgb",
            "quantized": False,
        },
        "claim_ceiling": artifact["claim_ceiling"],
    }
    return output, {
        **receipt_core,
        "receipt_sha256": _payload_sha256(receipt_core),
    }


def render_working_image_row_streamed(
    artifact: dict[str, Any],
    working: WorkingImage,
    *,
    tile_rows: int,
    order: str = "forward",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render exact finite-support row cores before one full-frame AO6 pass."""

    bundle = validate_standalone_profile_artifact(artifact)
    scene = scene_exposure_from_working_image(working)
    if np.any(scene.values > 1.0):
        raise ValueError(
            "v1 profile consumer requires scene-linear samples in [0,1]; "
            "a calibrated scene-to-relative exposure map is not available"
        )
    encoded = linear_srgb_to_encoded(
        scene.values.astype(np.float64)
    )
    runtime, gauge = reconstruct_standalone_runtime(artifact)
    output, seams = render_challenger_row_tiled(
        encoded,
        runtime,
        gauge,
        sampling_dpi=int(artifact["reference_sampling_dpi"]),
        tile_rows=tile_rows,
        order=order,
    )
    input_bytes = np.ascontiguousarray(
        scene.values.astype(np.float32, copy=False)
    ).tobytes()
    output_bytes = np.ascontiguousarray(output).tobytes()
    receipt_core = {
        "schema": (
            "neuro_film.physical_profile_row_streamed_render_receipt.v1"
        ),
        "profile_id": bundle.profile_id,
        "bundle_sha256": bundle.bundle_sha256,
        "artifact_sha256": _payload_sha256(artifact),
        "reference_sampling_dpi": int(
            artifact["reference_sampling_dpi"]
        ),
        "execution": {
            "mode": "exact-row-streamed-physical-one-full-frame-display-look",
            "tile_rows": int(tile_rows),
            "order": order,
            "seam_rows": list(seams),
        },
        "input": {
            "array_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "dtype": "float32",
            "shape": list(scene.values.shape),
            "working_space": working.working_space,
            "transfer_state": working.transfer_state,
            "source_transfer_state": working.source_transfer_state,
        },
        "output": {
            "array_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "dtype": output.dtype.name,
            "shape": list(output.shape),
            "domain": "display-encoded-rgb",
            "quantized": False,
        },
        "claim_ceiling": artifact["claim_ceiling"],
    }
    return output, {
        **receipt_core,
        "receipt_sha256": _payload_sha256(receipt_core),
    }


def render_working_image_fully_row_streamed(
    artifact: dict[str, Any],
    working: WorkingImage,
    *,
    tile_rows: int,
    order: str = "forward",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Row-bound both the physical chain and AO6 base/residual execution."""

    bundle = validate_standalone_profile_artifact(artifact)
    scene = scene_exposure_from_working_image(working)
    if np.any(scene.values > 1.0):
        raise ValueError(
            "v1 profile consumer requires scene-linear samples in [0,1]; "
            "a calibrated scene-to-relative exposure map is not available"
        )
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
        or order not in {"forward", "reverse"}
    ):
        raise ValueError("invalid fully row-streamed partition")
    input_array_sha256 = _array_sha256(scene.values)
    input_shape = list(scene.values.shape)
    encoded = _linear_srgb_to_encoded_row_staged(
        scene.values,
        tile_rows=tile_rows,
    )
    del scene
    source_context = build_density_source_context_inplace_packed_lab(
        artifact["component_payloads"][
            "ao6-source-context-display-look"
        ],
        encoded,
        tile_rows=tile_rows,
    )
    runtime, gauge = reconstruct_standalone_runtime(artifact)
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile,
            sampling_dpi=int(artifact["reference_sampling_dpi"]),
        ),
    )
    halo = required_spatial_response_halo(compiled.profile)
    linear = _refill_roundtrip_linear_inplace_row_staged(
        encoded,
        working.pixels,
        tile_rows=tile_rows,
    )
    del encoded
    ranges = [
        (y0, min(linear.shape[0], y0 + tile_rows))
        for y0 in range(0, linear.shape[0], tile_rows)
    ]
    if order == "reverse":
        ranges.reverse()
    seams = sorted(y0 for y0, _ in ranges if 0 < y0 < linear.shape[0])
    gauged_encoded = _render_physical_gauged_encoded_inplace(
        linear,
        ranges=ranges,
        order=order,
        halo=halo,
        runtime=compiled,
        gauge=gauge,
    )
    display = build_source_context_display_look_row_streamed(
        artifact["component_payloads"][
            "ao6-source-context-display-look"
        ],
        gauged_encoded,
        tile_rows=tile_rows,
        reuse_input_buffer=True,
        source_context=source_context,
    )
    output = display(gauged_encoded)
    receipt_core = {
        "schema": (
            "neuro_film.physical_profile_fully_row_streamed_render_receipt.v1"
        ),
        "profile_id": bundle.profile_id,
        "bundle_sha256": bundle.bundle_sha256,
        "artifact_sha256": _payload_sha256(artifact),
        "reference_sampling_dpi": int(
            artifact["reference_sampling_dpi"]
        ),
        "execution": {
            "mode": "exact-row-streamed-physical-and-display-look",
            "tile_rows": int(tile_rows),
            "order": order,
            "seam_rows": sorted(seams),
        },
        "input": {
            "array_sha256": input_array_sha256,
            "dtype": "float32",
            "shape": input_shape,
            "working_space": working.working_space,
            "transfer_state": working.transfer_state,
            "source_transfer_state": working.source_transfer_state,
        },
        "output": {
            "array_sha256": _array_sha256(output),
            "dtype": output.dtype.name,
            "shape": list(output.shape),
            "domain": "display-encoded-rgb",
            "quantized": False,
        },
        "claim_ceiling": artifact["claim_ceiling"],
    }
    return output, {
        **receipt_core,
        "receipt_sha256": _payload_sha256(receipt_core),
    }


def evaluate_standalone_profile(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    _, reference_runtime, reference_gauge = validate_contract(root, config)
    first = compile_standalone_profile_artifact(root=root, config=config)
    second = compile_standalone_profile_artifact(root=root, config=config)
    artifact_exact = _canonical_bytes(first) == _canonical_bytes(second)
    replay = config["replay"]
    sources = {
        "synthetic": np.random.default_rng(
            int(replay["synthetic_seed"])
        ).random(
            tuple(int(value) for value in replay["synthetic_shape"])
            + (3,)
        )
    }
    for sample_id in replay["real_ids"]:
        sources[sample_id] = _bounded_real_source(
            root,
            reference_runtime,
            sample_id,
            max_long_edge=int(replay["real_max_long_edge"]),
        )
    rows = []
    replay_exact = True
    for sample_id, source in sources.items():
        expected = render_challenger(
            source,
            reference_runtime,
            reference_gauge,
            sampling_dpi=4000,
        )
        actual = render_standalone_profile(first, source)
        exact = np.array_equal(expected, actual)
        replay_exact &= exact
        rows.append(
            {
                "sample_id": sample_id,
                "float_exact": exact,
                "maximum_absolute_error": float(
                    np.max(np.abs(expected - actual))
                ),
                "expected_float64_sha256": hashlib.sha256(
                    np.ascontiguousarray(expected).tobytes()
                ).hexdigest(),
                "actual_float64_sha256": hashlib.sha256(
                    np.ascontiguousarray(actual).tobytes()
                ).hexdigest(),
            }
        )
    core = {
        "schema": (
            "neuro_film.u6_p8b_artifact_only_cpu_consumer_report.v1"
        ),
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "artifact": first,
        "artifact_canonical_sha256": _payload_sha256(first),
        "artifact_exact": artifact_exact,
        "bundle_roundtrip_exact": (
            FilmProfileBundle.from_dict(
                first["film_profile_bundle"]
            ).to_dict()
            == first["film_profile_bundle"]
        ),
        "artifact_only_reconstruction": True,
        "replay_rows": rows,
        "replay_exact": replay_exact,
        "decision": (
            "pass" if artifact_exact and replay_exact else "fail"
        ),
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(core)
        ).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "CompiledProfileRuntime",
    "compile_standalone_profile_artifact",
    "evaluate_standalone_profile",
    "load_standalone_profile_artifact_bytes",
    "reconstruct_standalone_runtime",
    "render_standalone_profile",
    "render_working_image",
    "render_working_image_row_streamed",
    "render_working_image_fully_row_streamed",
    "serialize_standalone_profile_artifact",
    "validate_contract",
    "validate_standalone_profile_artifact",
    "write_report",
]
