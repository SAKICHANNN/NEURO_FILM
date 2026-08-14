"""Automatic U6.P7H value evaluation for fixed P4HU and AO6 arms."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.layer_gamma_photographic_development import (
    _high_frequency_chroma_p999,
)
from src.eval.native_density_photographic_integration import (
    build_native_density_stage_runtime,
)
from src.eval.native_msvc import sha256_file
from src.eval.neutral_base_photographic_ablation import (
    _new_boundary_fraction,
)
from src.eval.physical_spatial_photographic_stress import (
    _flat_region_p99,
    _isolated_excursions,
)
from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.display_look import (
    build_source_context_display_look_stages,
)
from src.film_physics.scanner import apply_scanner_safe_residual
from src.film_physics.spatial_response import (
    SpatialResponseProfile,
    apply_scanner_mtf,
)
from src.preprocess import save_srgb16_png
from src.preprocess.output_encode import srgb_icc_profile

CONTRACT_SCHEMA = "neuro-film.u6-p7h-p4hu-ao6-value-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p7h-p4hu-ao6-value-result.v1"
ARMS = (
    "fixed_ao6_colour_only_t15_c35",
    "matched_scanner_only_ao6_t15_c35",
    "p4hu_scanner_physical_only_diagnostic",
    "p4hu_scanner_then_fixed_ao6_t15_c35",
)
CURRENT_AO6, MATCHED_AO6, PHYSICAL_ONLY, COMBINED = ARMS
INCREMENTAL_REFERENCE_ARM = MATCHED_AO6
_SUPPORTED_TRANSFORMS = {"rotate_180"}
_P4HU_SCHEMA = (
    "neuro-film.u6-p4hu-native-spatial-density-photographic-integration-contract.v1"
)
_FLOAT_GATES = {
    "minimum_per_row_physical_residual_rms": (0.0, None, False),
    "minimum_population_p95_combined_vs_matched_scanner_ao6_abs": (
        0.0,
        None,
        False,
    ),
    "maximum_population_p99_combined_vs_matched_scanner_ao6_abs": (
        0.0,
        1.0,
        True,
    ),
    "maximum_flat_region_p99_combined_vs_matched_scanner_ao6_abs": (
        0.0,
        1.0,
        True,
    ),
    "maximum_high_frequency_chroma_p999": (0.0, 1.0, True),
    "isolated_excursion_threshold": (0.0, 1.0, False),
    "maximum_new_boundary_fraction_vs_matched_scanner_ao6": (
        0.0,
        1.0,
        True,
    ),
    "maximum_output_code_boundary_fraction": (0.0, 1.0, True),
}
_INTEGER_GATES = (
    "isolated_support_radius_pixels",
    "minimum_isolated_support_count",
    "maximum_isolated_excursion_count",
)
_DIAGNOSTIC_KEYS = (
    "receipt_ids",
    "bounded_residual_rms",
    "minimum_target_sigma_d",
    "maximum_target_sigma_d",
    "support_degenerate_fraction",
    "minimum_developed_density",
    "limited_fraction",
    "hard_clipping_used",
    "rank_bins",
    "pass_count",
    "canonical_row_block_height",
    "peak_live_temporary_bytes",
    "full_frame_intermediate_count_excluding_input_output",
)


class DensityStageRuntime(Protocol):
    """Minimal native-stage surface accepted by the evaluator."""

    native_toolchains: dict[str, Any]

    def apply_source(
        self,
        source: np.ndarray,
        *,
        seeds: tuple[int, int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]: ...


def load_contract(path: Path) -> dict[str, Any]:
    """Load one P7H evaluator contract without resolving its bindings."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported U6.P7H evaluator contract")
    return payload


def _canonical_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _root_path(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or any(part == ".." for part in value.parts):
        raise ValueError(f"{label} must be repository-relative")
    # Deliberately do not resolve here: project-owned data/output directories
    # are repository-relative NTFS junctions whose physical P: target lies
    # outside the checkout. Lexical containment preserves that contract.
    return root / value


def _load_bound_json(
    root: Path,
    binding: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    path = _root_path(root, binding["path"], label)
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"{label} identity drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _require_parent_decision(
    root: Path,
    binding: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    payload = _load_bound_json(root, binding, label)
    expected_schema = binding.get("required_schema")
    if expected_schema is not None and payload.get("schema") != expected_schema:
        raise ValueError(f"{label} schema drift")
    actual = payload.get("decision")
    if actual is None and isinstance(payload.get("result"), dict):
        actual = payload["result"].get("decision")
    if actual != binding["required_decision"]:
        raise ValueError(f"{label} decision drift")
    return payload


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unsupported U6.P7H evaluator contract")
    if tuple(contract["comparison"]["arms"]) != ARMS:
        raise ValueError("U6.P7H arm identity or order drift")
    candidate = contract["candidate"]
    if (
        tuple(float(value) for value in candidate["scanner_mtf_sigma_pixels_rgb"])
        != (0.7, 0.7, 0.7)
        or float(candidate["gaussian_truncate"]) != 3.0
        or candidate.get("cohort_fitting_allowed") is not False
        or candidate.get("hard_clipping_allowed") is not False
        or candidate.get("posthoc_limiting_allowed") is not False
        or candidate.get("ao6_source_context")
        != "original-encoded-source-only"
        or candidate.get("physical_only_promotion_eligible") is not False
    ):
        raise ValueError("U6.P7H fixed candidate policy drift")
    seeds = candidate["layer_field_seeds"]
    if len(seeds) != 3 or any(type(seed) is not int for seed in seeds):
        raise ValueError("U6.P7H requires exactly three integer field seeds")
    stride = candidate["field_seed_stride_per_source"]
    if type(stride) is not int or stride <= 0:
        raise ValueError("U6.P7H field-seed stride must be a positive integer")
    gates = contract["automatic_gates"]
    if (
        gates.get("require_all_inputs_and_outputs_finite") is not True
        or gates.get("require_no_hard_clipping_or_posthoc_limiting") is not True
    ):
        raise ValueError("U6.P7H automatic safety policy drift")
    for key, (minimum, maximum, inclusive_minimum) in _FLOAT_GATES.items():
        value = gates.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (
                float(value) < minimum
                if inclusive_minimum
                else float(value) <= minimum
            )
            or (maximum is not None and float(value) > maximum)
        ):
            raise ValueError(f"invalid U6.P7H automatic gate: {key}")
    for key in _INTEGER_GATES:
        value = gates.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid U6.P7H automatic gate: {key}")
    if (
        gates["isolated_support_radius_pixels"] <= 0
        or gates["minimum_isolated_support_count"] <= 0
        or gates["minimum_isolated_support_count"]
        > (2 * gates["isolated_support_radius_pixels"] + 1) ** 2
    ):
        raise ValueError("invalid U6.P7H isolated-support gate geometry")
    transforms = contract["source"].get("transforms", {})
    if not isinstance(transforms, dict) or any(
        value not in _SUPPORTED_TRANSFORMS for value in transforms.values()
    ):
        raise ValueError("unsupported U6.P7H source transform")


def _load_parents(
    root: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parents = contract["parents"]
    p4hu_contract = _load_bound_json(root, parents["p4hu_contract"], "P4HU contract")
    _require_parent_decision(root, parents["p4hu_evidence"], "P4HU evidence")
    _require_parent_decision(root, parents["p4hv_evidence"], "P4HV evidence")
    p4hu_candidate = p4hu_contract["candidate"]
    if (
        p4hu_contract.get("schema") != _P4HU_SCHEMA
        or p4hu_candidate.get("native_backend") != "msvc-x64-c11"
        or p4hu_candidate.get("spatial_backend")
        != "native-thomas-field-f32-v1"
        or p4hu_candidate.get("gamma_backend") != "fast-hybrid-v1"
        or p4hu_candidate.get("cohort_refit_allowed") is not False
        or p4hu_candidate.get("hard_clipping_allowed") is not False
    ):
        raise ValueError("P4HU native-source restriction drift")
    profile_binding = p4hu_contract["parents"]["profile"]
    profile = _load_bound_json(root, profile_binding, "P4HU profile")
    if profile.get("bundle_sha256") != profile_binding["bundle_sha256"]:
        raise ValueError("P4HU profile bundle drift")
    execution = profile.get("execution")
    if not isinstance(execution, dict):
        raise TypeError("P4HU profile execution contract is missing")
    p7h_candidate = contract["candidate"]
    profile_seeds = execution.get("layer_field_seeds")
    if (
        not isinstance(profile_seeds, list)
        or len(profile_seeds) != 3
        or any(type(seed) is not int for seed in profile_seeds)
        or profile_seeds != p7h_candidate["layer_field_seeds"]
        or execution.get("field_seed_stride_per_source")
        != p7h_candidate["field_seed_stride_per_source"]
        or any(seed < 0 or seed > 0xFFFFFFFFFFFFFFFF for seed in profile_seeds)
    ):
        raise ValueError("P7H field realization drift from bound P4HU profile")
    ao6_document = _load_bound_json(root, parents["ao6_artifact"], "AO6 artifact")
    expected_schema = parents["ao6_artifact"].get("required_schema")
    if expected_schema is not None and ao6_document.get("schema") != expected_schema:
        raise ValueError("AO6 artifact report schema drift")
    artifact = ao6_document.get("artifact", ao6_document)
    expected_artifact_schema = parents["ao6_artifact"].get(
        "required_artifact_schema"
    )
    if (
        expected_artifact_schema is not None
        and artifact.get("schema") != expected_artifact_schema
    ):
        raise ValueError("AO6 artifact schema drift")
    if artifact.get("bundle_sha256") != parents["ao6_artifact"][
        "required_bundle_sha256"
    ]:
        raise ValueError("AO6 artifact bundle drift")
    try:
        ao6_payload = artifact["component_payloads"][
            "ao6-source-context-display-look"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("AO6 artifact lacks the source-context display look") from exc
    return p4hu_contract, profile, ao6_payload


def _load_manifest(
    root: Path,
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    path = _root_path(root, source["manifest"], "source manifest")
    digest = sha256_file(path)
    if digest != source["manifest_sha256"]:
        raise ValueError("source manifest identity drift")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(manifest) != int(
        source["expected_manifest_rows"]
    ):
        raise ValueError("source manifest row-count drift")
    by_id: dict[str, dict[str, Any]] = {}
    for row in manifest:
        source_id = row.get("id")
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in by_id
            or any(
                not (character.isalnum() or character in {"-", "_"})
                for character in source_id
            )
        ):
            raise ValueError("source manifest IDs must be unique strings")
        by_id[source_id] = row
    included_ids = source["included_ids"]
    if (
        not isinstance(included_ids, list)
        or len(included_ids) != int(source["expected_evaluation_rows"])
        or len(set(included_ids)) != len(included_ids)
        or any(source_id not in by_id for source_id in included_ids)
    ):
        raise ValueError("source included-ID contract drift")
    if set(source.get("transforms", {})) - set(included_ids):
        raise ValueError("source transform names an excluded ID")
    selected = [by_id[source_id] for source_id in included_ids]
    required_fields = (
        "id",
        "source_id",
        "make",
        "decoded_path",
        "decoded_sha256",
        "width",
        "height",
        "allowed_use",
        "rights_scope",
        "decoded_color_state",
    )
    for row in selected:
        if any(key not in row for key in required_fields):
            raise ValueError(f"incomplete source manifest row: {row.get('id')}")
        if (
            row["allowed_use"] != source["required_allowed_use"]
            or row["rights_scope"] != source["required_rights_scope"]
            or row["decoded_color_state"] != source["required_color_state"]
        ):
            raise ValueError(f"source eligibility drift: {row['id']}")
    makes = {row["make"] for row in selected}
    if len(makes) != int(source["expected_camera_makes"]):
        raise ValueError("source camera-make count drift")
    return selected, digest


def _load_rgb8_source(
    root: Path,
    row: dict[str, Any],
    transform: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = _root_path(root, row["decoded_path"], f"decoded source {row['id']}")
    digest = sha256_file(path)
    if digest != row["decoded_sha256"]:
        raise ValueError(f"decoded source hash drift: {row['id']}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"decoded source must be RGB8 PNG: {row['id']}")
        rgb8 = np.asarray(image)
        if rgb8.dtype != np.uint8:
            raise ValueError(f"decoded source must be true RGB8: {row['id']}")
        rgb8 = rgb8.copy()
    if rgb8.shape != (int(row["height"]), int(row["width"]), 3):
        raise ValueError(f"decoded source dimension drift: {row['id']}")
    if transform == "rotate_180":
        rgb8 = np.ascontiguousarray(rgb8[::-1, ::-1])
    elif transform is not None:
        raise ValueError(f"unsupported transform for {row['id']}")
    encoded = np.ascontiguousarray(rgb8, dtype=np.float32) / np.float32(255.0)
    return encoded, {
        "decoded_png_sha256": digest,
        "transform": transform or "identity",
        "post_transform_rgb8_sha256": _array_sha256(rgb8),
        "post_transform_encoded_array_sha256": _array_sha256(encoded),
        "post_transform_shape": list(encoded.shape),
    }


def _scanner_profile(candidate: dict[str, Any]) -> SpatialResponseProfile:
    scanner_sigma = tuple(
        float(value) for value in candidate["scanner_mtf_sigma_pixels_rgb"]
    )
    return SpatialResponseProfile(
        pixel_pitch_um=1.0,
        forward_scatter_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_sigma_um_rgb=(0.0, 0.0, 0.0),
        development_adjacency_gain_rgb=(0.0, 0.0, 0.0),
        dye_diffusion_sigma_um_rgb=(0.0, 0.0, 0.0),
        scanner_mtf_sigma_um_rgb=scanner_sigma,
        gaussian_truncate=float(candidate["gaussian_truncate"]),
    )


def _display(values: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    output = np.ascontiguousarray(values, dtype=np.float32)
    if (
        output.shape != shape
        or not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
    ):
        raise RuntimeError(f"{label} left finite display-sRGB")
    return output


def _extract_png_icc(payload: bytes) -> bytes:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("output is not a PNG")
    offset = 8
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"iCCP":
            separator = chunk.find(b"\x00")
            if separator < 1 or chunk[separator + 1 : separator + 2] != b"\x00":
                raise RuntimeError("invalid PNG iCCP chunk")
            return zlib.decompress(chunk[separator + 2 :])
        if chunk_type == b"IEND":
            break
    raise RuntimeError("output PNG lacks an ICC profile")


def _write_verified_png16(
    values: np.ndarray,
    *,
    output_dir: Path,
    arm_id: str,
    source_id: str,
) -> dict[str, Any]:
    path = output_dir / "renders" / arm_id / f"{source_id}.png"
    if path.exists():
        raise FileExistsError(f"U6.P7H render is create-only: {path}")
    save_srgb16_png(values, path)
    expected = np.rint(values * np.float32(65535.0)).astype(np.uint16)
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("OpenCV could not reopen U6.P7H PNG16 output")
    decoded = np.ascontiguousarray(decoded[..., ::-1])
    if (
        decoded.dtype != np.uint16
        or decoded.shape != expected.shape
        or not np.array_equal(decoded, expected)
    ):
        raise RuntimeError("U6.P7H PNG16 exact readback failed")
    payload = path.read_bytes()
    icc = _extract_png_icc(payload)
    expected_icc = srgb_icc_profile()
    if icc != expected_icc:
        raise RuntimeError("U6.P7H PNG16 ICC identity drift")
    boundary = np.any((expected == 0) | (expected == 65535), axis=-1)
    return {
        "arm_id": arm_id,
        "output_path": path.relative_to(output_dir).as_posix(),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_encoded_array_sha256": _array_sha256(values),
        "output_rgb16_array_sha256": _array_sha256(expected),
        "output_code_boundary_fraction": float(np.mean(boundary)),
        "output_icc_profile_sha256": hashlib.sha256(icc).hexdigest(),
        "exact_uint16_readback": True,
    }


def _stable_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in _DIAGNOSTIC_KEYS if key not in diagnostics]
    if missing:
        raise RuntimeError(f"P4HU diagnostic receipt is incomplete: {missing}")
    stable: dict[str, Any] = {}
    for key in _DIAGNOSTIC_KEYS:
        value = diagnostics[key]
        if isinstance(value, np.generic):
            value = value.item()
        stable[key] = value
    return stable


def _evaluate_source(
    *,
    root: Path,
    output_dir: Path,
    row: dict[str, Any],
    transform: str | None,
    seeds: tuple[int, int, int],
    runtime: DensityStageRuntime,
    ao6_payload: dict[str, Any],
    scanner: SpatialResponseProfile,
    gates: dict[str, Any],
    arm_ids: tuple[str, str, str, str] = ARMS,
    physical_after_ao6_base: bool = False,
    scanner_safe_residual: bool = False,
) -> dict[str, Any]:
    if physical_after_ao6_base and scanner_safe_residual:
        raise ValueError("scanner-safe residual is unsupported for base-first execution")
    current_ao6, matched_ao6, physical_only, combined = arm_ids
    encoded, source_receipt = _load_rgb8_source(root, row, transform)
    linear = np.ascontiguousarray(
        encoded_srgb_to_linear(encoded.astype(np.float64)), dtype=np.float32
    )
    scanner_source = np.ascontiguousarray(
        apply_scanner_mtf(linear, scanner), dtype=np.float32
    )
    encoded_scanner_source = _display(
        linear_srgb_to_encoded(scanner_source.astype(np.float64)),
        encoded.shape,
        "matched scanner source",
    )

    # This is deliberately the sole context build. Every AO6 arm shares the
    # original (possibly objective-transform-corrected) source statistics.
    apply_base, apply_residual = build_source_context_display_look_stages(
        ao6_payload, encoded
    )

    def apply_ao6(values: np.ndarray, label: str) -> np.ndarray:
        return _display(
            apply_residual(apply_base(values)), encoded.shape, label
        )
    scanner_safe_receipt = None
    if physical_after_ao6_base:
        current_base = _display(apply_base(encoded), encoded.shape, "current AO6 base")
        matched_base = _display(
            apply_base(encoded_scanner_source), encoded.shape, "matched AO6 base"
        )
        physical_input = np.ascontiguousarray(
            encoded_srgb_to_linear(matched_base.astype(np.float64)), dtype=np.float32
        )
        physical, diagnostics = runtime.apply_source(physical_input, seeds=seeds)
        physical = _display(physical, linear.shape, "base-first physical source")
        physical = np.ascontiguousarray(
            apply_scanner_mtf(physical, scanner), dtype=np.float32
        )
        encoded_physical = _display(
            linear_srgb_to_encoded(physical.astype(np.float64)),
            encoded.shape,
            "base-first physical output",
        )
        arms = {
            current_ao6: _display(
                apply_residual(current_base), encoded.shape, current_ao6
            ),
            matched_ao6: _display(
                apply_residual(matched_base), encoded.shape, matched_ao6
            ),
            physical_only: encoded_physical,
            combined: _display(
                apply_residual(encoded_physical), encoded.shape, combined
            ),
        }
    else:
        physical, diagnostics = runtime.apply_source(linear, seeds=seeds)
        physical = _display(physical, linear.shape, "P4HU physical source")
        scanner_physical = np.ascontiguousarray(
            apply_scanner_mtf(physical, scanner), dtype=np.float32
        )
        if scanner_safe_residual:
            scanner_physical, receipt = apply_scanner_safe_residual(
                scanner_source.astype(np.float64),
                scanner_physical.astype(np.float64),
            )
            scanner_safe_receipt = {
                "runtime_id": receipt.runtime_id,
                "limited_pixel_fraction": receipt.limited_pixel_fraction,
                "median_scale": receipt.median_scale,
                "minimum_scale": receipt.minimum_scale,
                "maximum_collinearity_error": receipt.maximum_collinearity_error,
            }
        encoded_scanner_physical = _display(
            linear_srgb_to_encoded(scanner_physical.astype(np.float64)),
            encoded.shape,
            "physical-only scanner output",
        )
        arms = {
            current_ao6: apply_ao6(encoded, current_ao6),
            matched_ao6: apply_ao6(encoded_scanner_source, matched_ao6),
            physical_only: encoded_scanner_physical,
            combined: apply_ao6(encoded_scanner_physical, combined),
        }
    outputs = [
        _write_verified_png16(
            arms[arm_id],
            output_dir=output_dir,
            arm_id=arm_id,
            source_id=row["id"],
        )
        for arm_id in arm_ids
    ]
    incremental = arms[combined] - arms[matched_ao6]
    product_delta = arms[combined] - arms[current_ao6]
    scanner_control_delta = arms[matched_ao6] - arms[current_ao6]
    stable_diagnostics = _stable_diagnostics(diagnostics)
    result = {
        "source_id": row["id"],
        "source_manifest_id": row["source_id"],
        "make": row["make"],
        "shape": list(encoded.shape),
        "source": source_receipt,
        "ao6_source_context_encoded_sha256": _array_sha256(encoded),
        "ao6_source_context_policy": "original-encoded-source-only",
        "layer_field_seeds": list(seeds),
        "native_diagnostics": stable_diagnostics,
        "physical_residual_rms": float(
            stable_diagnostics["bounded_residual_rms"]
        ),
        "combined_vs_matched_scanner_ao6_p95_abs": float(
            np.quantile(np.abs(incremental), 0.95)
        ),
        "combined_vs_matched_scanner_ao6_p99_abs": float(
            np.quantile(np.abs(incremental), 0.99)
        ),
        "flat_region_p99_combined_vs_matched_scanner_ao6_abs": (
            _flat_region_p99(linear, incremental)
        ),
        "high_frequency_chroma_p999": _high_frequency_chroma_p999(
            incremental
        ),
        "isolated_excursion_count": _isolated_excursions(
            incremental,
            threshold=float(gates["isolated_excursion_threshold"]),
            radius=int(gates["isolated_support_radius_pixels"]),
            minimum_support=int(gates["minimum_isolated_support_count"]),
        ),
        "new_boundary_fraction_vs_matched_scanner_ao6": (
            _new_boundary_fraction(arms[matched_ao6], arms[combined])
        ),
        "combined_vs_current_ao6_p95_abs": float(
            np.quantile(np.abs(product_delta), 0.95)
        ),
        "combined_vs_current_ao6_p99_abs": float(
            np.quantile(np.abs(product_delta), 0.99)
        ),
        "new_boundary_fraction_vs_current_ao6": _new_boundary_fraction(
            arms[current_ao6], arms[combined]
        ),
        "matched_scanner_vs_current_ao6_p95_abs": float(
            np.quantile(np.abs(scanner_control_delta), 0.95)
        ),
        "matched_scanner_vs_current_ao6_p99_abs": float(
            np.quantile(np.abs(scanner_control_delta), 0.99)
        ),
        "outputs": outputs,
    }
    if scanner_safe_receipt is not None:
        result["scanner_safe_residual"] = scanner_safe_receipt
    return result


def evaluate_source_arms(
    *,
    root: Path,
    output_dir: Path,
    row: dict[str, Any],
    transform: str | None,
    seeds: tuple[int, int, int],
    runtime: DensityStageRuntime,
    ao6_payload: dict[str, Any],
    scanner: SpatialResponseProfile,
    gates: dict[str, Any],
    arm_ids: tuple[str, str, str, str] = ARMS,
) -> dict[str, Any]:
    """Evaluate the frozen P7H four-arm surface for one bound structure runtime."""

    return _evaluate_source(
        root=root,
        output_dir=output_dir,
        row=row,
        transform=transform,
        seeds=seeds,
        runtime=runtime,
        ao6_payload=ao6_payload,
        scanner=scanner,
        gates=gates,
        arm_ids=arm_ids,
    )


def evaluate_scanner_safe_source_arms(
    *,
    root: Path,
    output_dir: Path,
    row: dict[str, Any],
    transform: str | None,
    seeds: tuple[int, int, int],
    runtime: DensityStageRuntime,
    ao6_payload: dict[str, Any],
    scanner: SpatialResponseProfile,
    gates: dict[str, Any],
    arm_ids: tuple[str, str, str, str],
) -> dict[str, Any]:
    """Evaluate P4HU after the canonical scanner-safe residual executor."""

    return _evaluate_source(
        root=root,
        output_dir=output_dir,
        row=row,
        transform=transform,
        seeds=seeds,
        runtime=runtime,
        ao6_payload=ao6_payload,
        scanner=scanner,
        gates=gates,
        arm_ids=arm_ids,
        scanner_safe_residual=True,
    )


def evaluate_base_first_source_arms(
    *,
    root: Path,
    output_dir: Path,
    row: dict[str, Any],
    transform: str | None,
    seeds: tuple[int, int, int],
    runtime: DensityStageRuntime,
    ao6_payload: dict[str, Any],
    scanner: SpatialResponseProfile,
    gates: dict[str, Any],
    arm_ids: tuple[str, str, str, str] = ARMS,
) -> dict[str, Any]:
    """Evaluate physical structure after the fixed AO6 base and before residual."""

    return _evaluate_source(
        root=root,
        output_dir=output_dir,
        row=row,
        transform=transform,
        seeds=seeds,
        runtime=runtime,
        ao6_payload=ao6_payload,
        scanner=scanner,
        gates=gates,
        arm_ids=arm_ids,
        physical_after_ao6_base=True,
    )


def aggregate_arm_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows produced by :func:`evaluate_source_arms`."""

    return _aggregate(rows)


def automatic_arm_checks(
    aggregates: dict[str, Any],
    gates: dict[str, Any],
    *,
    expected_rows: int,
    expected_arm_count: int = len(ARMS),
) -> dict[str, bool]:
    """Apply the frozen P7H automatic gates to a compatible four-arm result."""

    return _automatic_checks(
        aggregates,
        gates,
        expected_rows=expected_rows,
        expected_arm_count=expected_arm_count,
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_rows = [output for row in rows for output in row["outputs"]]
    return {
        "source_count": len(rows),
        "output_count": len(output_rows),
        "minimum_physical_residual_rms": min(
            row["physical_residual_rms"] for row in rows
        ),
        "population_p95_combined_vs_matched_scanner_ao6_abs": float(
            np.quantile(
                [
                    row["combined_vs_matched_scanner_ao6_p95_abs"]
                    for row in rows
                ],
                0.95,
            )
        ),
        "population_p99_combined_vs_matched_scanner_ao6_abs": float(
            np.quantile(
                [
                    row["combined_vs_matched_scanner_ao6_p99_abs"]
                    for row in rows
                ],
                0.99,
            )
        ),
        "maximum_flat_region_p99_combined_vs_matched_scanner_ao6_abs": max(
            row["flat_region_p99_combined_vs_matched_scanner_ao6_abs"]
            for row in rows
        ),
        "maximum_high_frequency_chroma_p999": max(
            row["high_frequency_chroma_p999"] for row in rows
        ),
        "total_isolated_excursion_count": sum(
            row["isolated_excursion_count"] for row in rows
        ),
        "maximum_new_boundary_fraction_vs_matched_scanner_ao6": max(
            row["new_boundary_fraction_vs_matched_scanner_ao6"]
            for row in rows
        ),
        "maximum_new_boundary_fraction_vs_current_ao6": max(
            row["new_boundary_fraction_vs_current_ao6"] for row in rows
        ),
        "population_p95_combined_vs_current_ao6_abs": float(
            np.quantile(
                [row["combined_vs_current_ao6_p95_abs"] for row in rows],
                0.95,
            )
        ),
        "population_p99_combined_vs_current_ao6_abs": float(
            np.quantile(
                [row["combined_vs_current_ao6_p99_abs"] for row in rows],
                0.99,
            )
        ),
        "population_p95_matched_scanner_vs_current_ao6_abs": float(
            np.quantile(
                [
                    row["matched_scanner_vs_current_ao6_p95_abs"]
                    for row in rows
                ],
                0.95,
            )
        ),
        "maximum_output_code_boundary_fraction": max(
            output["output_code_boundary_fraction"] for output in output_rows
        ),
        "all_png16_exact_readback": all(
            output["exact_uint16_readback"] for output in output_rows
        ),
        "maximum_limited_fraction": max(
            float(row["native_diagnostics"]["limited_fraction"])
            for row in rows
        ),
        "hard_clipping_used": any(
            bool(row["native_diagnostics"]["hard_clipping_used"])
            for row in rows
        ),
    }


def _automatic_checks(
    aggregates: dict[str, Any],
    gates: dict[str, Any],
    *,
    expected_rows: int,
    expected_arm_count: int = len(ARMS),
) -> dict[str, bool]:
    return {
        "complete_inventory": (
            aggregates["source_count"] == expected_rows
            and aggregates["output_count"] == expected_rows * expected_arm_count
        ),
        "physical_residual": aggregates["minimum_physical_residual_rms"]
        >= float(gates["minimum_per_row_physical_residual_rms"]),
        "nontrivial_population": (
            aggregates[
                "population_p95_combined_vs_matched_scanner_ao6_abs"
            ]
            >= float(
                gates[
                    "minimum_population_p95_combined_vs_matched_scanner_ao6_abs"
                ]
            )
        ),
        "population_tail": (
            aggregates[
                "population_p99_combined_vs_matched_scanner_ao6_abs"
            ]
            <= float(
                gates[
                    "maximum_population_p99_combined_vs_matched_scanner_ao6_abs"
                ]
            )
        ),
        "flat_regions": (
            aggregates[
                "maximum_flat_region_p99_combined_vs_matched_scanner_ao6_abs"
            ]
            <= float(
                gates[
                    "maximum_flat_region_p99_combined_vs_matched_scanner_ao6_abs"
                ]
            )
        ),
        "high_frequency_chroma": (
            aggregates["maximum_high_frequency_chroma_p999"]
            <= float(gates["maximum_high_frequency_chroma_p999"])
        ),
        "isolated_excursions": aggregates["total_isolated_excursion_count"]
        <= int(gates["maximum_isolated_excursion_count"]),
        "new_boundaries": (
            aggregates[
                "maximum_new_boundary_fraction_vs_matched_scanner_ao6"
            ]
            <= float(
                gates[
                    "maximum_new_boundary_fraction_vs_matched_scanner_ao6"
                ]
            )
        ),
        "output_boundaries": aggregates[
            "maximum_output_code_boundary_fraction"
        ]
        <= float(gates["maximum_output_code_boundary_fraction"]),
        "png16_exact_readback": aggregates["all_png16_exact_readback"],
        "finite": True,
        "no_clipping_or_limiting": (
            not aggregates["hard_clipping_used"]
            and aggregates["maximum_limited_fraction"] == 0.0
        ),
        "physical_only_diagnostic": True,
    }


def evaluate(
    contract: dict[str, Any],
    *,
    root: Path,
    output_dir: Path,
    build_dir: Path,
) -> dict[str, Any]:
    """Render and automatically gate the four fixed P7H arms.

    The caller owns process monitoring and report serialization. This function
    writes only create-only full-resolution arm PNGs and returns stable science
    content that is independent of the caller's run-directory name.
    """

    _validate_contract(contract)
    p4hu_contract, profile_payload, ao6_payload = _load_parents(root, contract)
    selected, manifest_sha256 = _load_manifest(root, contract["source"])
    components = reconstruct_bounded_photographic_profile(profile_payload)
    native_stage_runtime = build_native_density_stage_runtime(
        root=root,
        build_dir=build_dir,
        candidate=p4hu_contract["candidate"],
        profile_payload=profile_payload,
        components=components,
    )
    candidate = contract["candidate"]
    scanner = _scanner_profile(candidate)
    gates = contract["automatic_gates"]
    base_seeds = tuple(int(seed) for seed in candidate["layer_field_seeds"])
    stride = int(candidate["field_seed_stride_per_source"])
    transforms = contract["source"].get("transforms", {})
    rows = []
    for index, source_row in enumerate(selected):
        seeds = tuple(seed + index * stride for seed in base_seeds)
        rows.append(
            _evaluate_source(
                root=root,
                output_dir=output_dir,
                row=source_row,
                transform=transforms.get(source_row["id"]),
                seeds=seeds,
                runtime=native_stage_runtime,
                ao6_payload=ao6_payload,
                scanner=scanner,
                gates=gates,
            )
        )
    aggregates = _aggregate(rows)
    checks = _automatic_checks(
        aggregates,
        gates,
        expected_rows=int(contract["source"]["expected_evaluation_rows"]),
    )
    automatic_pass = all(checks.values())
    core = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _canonical_sha256(contract),
        "manifest_sha256": manifest_sha256,
        "profile_bundle_sha256": profile_payload["bundle_sha256"],
        "ao6_payload_sha256": _canonical_sha256(ao6_payload),
        "native_toolchains": native_stage_runtime.native_toolchains,
        "scanner_profile": {
            "scanner_mtf_sigma_pixels_rgb": [0.7, 0.7, 0.7],
            "gaussian_truncate": 3.0,
        },
        "arms": list(ARMS),
        "arm_roles": {
            CURRENT_AO6: "current_incumbent",
            MATCHED_AO6: "incremental_safety_reference",
            PHYSICAL_ONLY: "diagnostic_only_never_promotion_eligible",
            COMBINED: "promotion_challenger",
        },
        "incremental_gate_reference_arm": INCREMENTAL_REFERENCE_ARM,
        "rows": rows,
        "aggregates": aggregates,
        "gates": checks,
        "automatic_pass": automatic_pass,
        "blind_review_allowed": automatic_pass,
        "decision": (
            contract["decision_if_pass"]
            if automatic_pass
            else contract["decision_if_fail"]
        ),
        "physical_only_promotion_eligible": False,
        "production_default_changed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


__all__ = [
    "ARMS",
    "COMBINED",
    "CONTRACT_SCHEMA",
    "CURRENT_AO6",
    "MATCHED_AO6",
    "PHYSICAL_ONLY",
    "RESULT_SCHEMA",
    "aggregate_arm_rows",
    "automatic_arm_checks",
    "evaluate",
    "evaluate_base_first_source_arms",
    "evaluate_scanner_safe_source_arms",
    "evaluate_source_arms",
    "load_contract",
]
