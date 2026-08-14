"""U1.4C4 native ProPhoto ingress plus unchanged C3 residual confirmation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from scripts.build_fivek_freeze_pack import (
    D50_TO_D65_BRADFORD,
    PROPHOTO_TO_XYZ_D50,
    XYZ_D65_TO_SRGB,
    prophoto_decode,
    resize_float,
)
from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.rec2020_safe_lab import apply_rec2020_safe_lab
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.rec2020_source_anchored_interior import (
    _array_sha256,
    _median_lab_delta_e76,
    _new_boundary_fraction,
    _save_and_verify,
    _style_kwargs,
    source_anchored_interior_residual,
)
from src.preprocess.color_management import convert_linear_rgb
from src.preprocess.types import SourceProfile, WorkingImage

SCHEMA = "neuro_film.u1-4c4-native-prophoto-rec2020-confirmation-contract.v1"
REPORT_SCHEMA = "neuro_film.u1-4c4-native-prophoto-rec2020-confirmation-report.v1"
EXPERIMENT_ID = "U1.4C4"
CONTRACT_SHA256 = "6e3caf8e65a71c798e97dff26e3af21c56193409decdb1f5c48b7b794733d607"


class NativeProPhotoError(RuntimeError):
    """Raised when the frozen C4 contract or execution invariant fails."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise NativeProPhotoError("C4 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise NativeProPhotoError("C4 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise NativeProPhotoError("C4 contract structure drift")
    _relative_path(payload["source"]["manifest"])
    for parent in payload["parents"].values():
        _relative_path(parent["path"])
    return payload


def luminance_axis_interior_compress(
    source: np.ndarray,
    *,
    luminance_weights: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map extended Rec.2020 to gamut along its luminance-neutral RGB axis."""

    if (
        not isinstance(source, np.ndarray)
        or source.dtype != np.float32
        or source.ndim != 3
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or not isinstance(luminance_weights, np.ndarray)
        or luminance_weights.shape != (3,)
        or not np.isfinite(luminance_weights).all()
        or np.any(luminance_weights <= 0.0)
        or not np.isclose(float(np.sum(luminance_weights)), 1.0, atol=1e-12)
        or not np.isfinite(margin)
        or margin <= 0.0
        or margin >= 0.5
    ):
        raise NativeProPhotoError("C4 source, luminance weights, or margin is invalid")
    frozen = source.copy()
    source64 = source.astype(np.float64)
    weights64 = luminance_weights.astype(np.float64)
    luminance = np.matmul(source64, weights64)
    if float(np.min(luminance)) < -2e-6 or float(np.max(luminance)) > 1.0 + 2e-6:
        raise NativeProPhotoError("C4 source luminance is outside the display interval")
    neutral = np.repeat(luminance[..., None], 3, axis=-1)
    residual = source64 - neutral
    lower = np.minimum(neutral, margin)
    upper = np.maximum(neutral, 1.0 - margin)
    scale = np.ones(source.shape[:2], dtype=np.float64)
    out_of_gamut = np.any((source64 < 0.0) | (source64 > 1.0), axis=-1)
    for channel in range(3):
        value = residual[..., channel]
        positive = out_of_gamut & (value > 0.0) & (
            neutral[..., channel] + value > upper[..., channel]
        )
        negative = out_of_gamut & (value < 0.0) & (
            neutral[..., channel] + value < lower[..., channel]
        )
        bound = np.ones_like(value)
        bound[positive] = (
            upper[..., channel][positive] - neutral[..., channel][positive]
        ) / value[positive]
        bound[negative] = (
            lower[..., channel][negative] - neutral[..., channel][negative]
        ) / value[negative]
        scale = np.minimum(scale, bound)
    scale = np.minimum(np.maximum(scale, 0.0), 1.0)
    mapped64 = np.where(
        out_of_gamut[..., None],
        neutral + scale[..., None] * residual,
        source64,
    )
    mapped = np.asarray(mapped64, dtype=np.float32)
    mapped_luminance = np.matmul(mapped.astype(np.float64), weights64)
    if (
        not np.array_equal(source, frozen)
        or not np.isfinite(mapped).all()
        or float(np.min(mapped)) < 0.0
        or float(np.max(mapped)) > 1.0
        or not np.array_equal(mapped[~out_of_gamut], source[~out_of_gamut])
        or float(np.max(np.abs(mapped_luminance - luminance))) > 2e-6
    ):
        raise NativeProPhotoError("C4 analytical ingress invariant failed")
    return mapped, np.asarray(scale, dtype=np.float32), out_of_gamut


def _working(pixels: np.ndarray, source_path: Path) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space="linear_rec2020",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile(
            "embedded_icc",
            "ProPhoto RGB decoded, Bradford-adapted, then analytically mapped to Rec.2020",
        ),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=source_path,
    )


def _validate_inputs(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    for parent in config["parents"].values():
        path = root / _relative_path(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise NativeProPhotoError("C4 parent identity drift")
    c3_result = json.loads(
        (root / _relative_path(config["parents"]["c3_result"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if c3_result.get("automatic_pass") is not True:
        raise NativeProPhotoError("C4 requires the passing C3 result")
    source = config["source"]
    manifest_path = root / _relative_path(source["manifest"])
    if not manifest_path.is_file() or hash_file(manifest_path) != source["manifest_sha256"]:
        raise NativeProPhotoError("C4 source manifest identity drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows")
    if (
        manifest.get("schema")
        != "neuro_film.u1-4c4-native-prophoto-source-manifest.v1"
        or not isinstance(rows, list)
        or len(rows) != int(source["expected_rows"])
        or len({row["id"] for row in rows}) != len(rows)
        or len({(row["make"], row["model"]) for row in rows})
        != int(source["expected_camera_models"])
        or [row["selection_rank"] for row in rows] != list(range(len(rows)))
        or manifest.get("embedded_icc_sha256")
        != source["expected_embedded_icc_sha256"]
        or manifest.get("allowed_use") != source["required_allowed_use"]
        or manifest.get("rights_scope") != source["required_rights_scope"]
    ):
        raise NativeProPhotoError("C4 source inventory drift")
    return [dict(row) for row in rows]


def _load_native_source(
    row: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = root / _relative_path(row["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(row["bytes"])
        or hash_file(path) != row["sha256"]
    ):
        raise NativeProPhotoError("C4 source file identity drift")
    with tifffile.TiffFile(path) as document:
        if len(document.pages) != 1:
            raise NativeProPhotoError("C4 source TIFF must contain one page")
        page = document.pages[0]
        tags = {tag.name: tag.value for tag in page.tags.values()}
        profile = tags.get("InterColorProfile")
        if (
            page.dtype != np.dtype(np.uint16)
            or tuple(page.shape) != (int(row["height"]), int(row["width"]), 3)
            or not isinstance(profile, bytes)
            or hashlib.sha256(profile).hexdigest()
            != config["source"]["expected_embedded_icc_sha256"]
            or str(tags.get("Make")) != row["make"]
            or str(tags.get("Model")) != row["model"]
        ):
            raise NativeProPhotoError("C4 source TIFF structure or profile drift")
    data = tifffile.imread(path)
    encoded = resize_float(
        data.astype(np.float32) / np.float32(65535.0),
        int(config["source"]["maximum_evaluation_side"]),
    )
    prophoto_linear = prophoto_decode(encoded)
    xyz_d50 = prophoto_linear @ PROPHOTO_TO_XYZ_D50.T
    xyz_d65 = xyz_d50 @ D50_TO_D65_BRADFORD.T
    linear_srgb = xyz_d65 @ XYZ_D65_TO_SRGB.T
    rec2020_extended = convert_linear_rgb(
        np.asarray(linear_srgb, dtype=np.float32),
        source_space="linear_srgb",
        destination_space="linear_rec2020",
    )
    weights = np.asarray(config["ingress"]["rec2020_luminance_weights"], dtype=np.float64)
    mapped, ingress_scale, out_of_gamut = luminance_axis_interior_compress(
        rec2020_extended,
        luminance_weights=weights,
        margin=2.0 / 65535.0,
    )
    source_luminance = np.matmul(rec2020_extended.astype(np.float64), weights)
    mapped_luminance = np.matmul(mapped.astype(np.float64), weights)
    facts = {
        "native_srgb_out_of_gamut_fraction": float(
            np.mean(np.any((linear_srgb < 0.0) | (linear_srgb > 1.0), axis=-1))
        ),
        "precompression_rec2020_out_of_gamut_fraction": float(np.mean(out_of_gamut)),
        "ingress_scale_median_on_out_of_gamut": float(np.median(ingress_scale[out_of_gamut])),
        "ingress_scale_minimum": float(np.min(ingress_scale)),
        "maximum_luminance_preservation_absolute_error": float(
            np.max(np.abs(mapped_luminance - source_luminance))
        ),
        "mapped_source_array_sha256": _array_sha256(mapped),
        "mapped_source_boundary_fraction": float(
            np.mean(np.any((mapped <= 1.0 / 65536.0) | (mapped >= 1.0 - 1.0 / 65536.0), axis=-1))
        ),
        "shape": list(mapped.shape),
    }
    return mapped, rec2020_extended, facts


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    source_rows = _validate_inputs(config, root)
    if output_dir.exists():
        raise NativeProPhotoError("C4 output directory must be create-only")
    output_dir.mkdir(parents=True)
    source_facts: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        mapped, extended, facts = _load_native_source(source_row, config, root)
        facts.update(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "model": source_row["model"],
                "source_sha256": source_row["sha256"],
                "extended_rec2020_array_sha256": _array_sha256(extended),
            }
        )
        source_facts.append(facts)
        source_path = root / _relative_path(source_row["path"])
        working = _working(mapped, source_path)
        source_lab = linear_rgb_to_lab(mapped, working_space="linear_rec2020")
        for style in config["render"]["styles"]:
            for mode in config["render"]["gamut_modes"]:
                candidate = apply_rec2020_safe_lab(
                    working, **_style_kwargs(style, mode, root)
                ).pixels
                guarded, scale = source_anchored_interior_residual(
                    mapped,
                    candidate,
                    margin=2.0 / 65535.0,
                )
                candidate_lab = linear_rgb_to_lab(
                    candidate, working_space="linear_rec2020"
                )
                guarded_lab = linear_rgb_to_lab(
                    guarded, working_space="linear_rec2020"
                )
                baseline_style = _median_lab_delta_e76(source_lab, candidate_lab)
                guarded_style = _median_lab_delta_e76(source_lab, guarded_lab)
                style_retention = (
                    guarded_style / baseline_style if baseline_style > 1e-12 else 1.0
                )
                relative_path = Path("renders") / style / mode / f"{source_row['id']}.png"
                output_sha = _save_and_verify(
                    _working(guarded, source_path), output_dir / relative_path
                )
                render_rows.append(
                    {
                        "id": source_row["id"],
                        "make": source_row["make"],
                        "model": source_row["model"],
                        "style": style,
                        "gamut_mode": mode,
                        "source_sha256": source_row["sha256"],
                        "shape": list(mapped.shape),
                        "mapped_source_array_sha256": facts["mapped_source_array_sha256"],
                        "candidate_array_sha256": _array_sha256(candidate),
                        "guarded_array_sha256": _array_sha256(guarded),
                        "output_path": relative_path.as_posix(),
                        "output_sha256": output_sha,
                        "baseline_style_delta_e76": baseline_style,
                        "guarded_style_delta_e76": guarded_style,
                        "style_retention_ratio": style_retention,
                        "residual_scale_median": float(np.median(scale)),
                        "fraction_residual_scale_below_0p5": float(np.mean(scale < 0.5)),
                        "new_rgb16_boundary_fraction_vs_mapped_source": _new_boundary_fraction(
                            mapped, guarded
                        ),
                        "p999_gradient_ratio_vs_mapped_source": _gradient_p999_ratio(
                            mapped, guarded
                        ),
                        "adjacent_lstar_sign_inversion_fraction": _gradient_inversion_fraction(
                            source_lab[..., 0], guarded_lab[..., 0], epsilon=0.02
                        ),
                        "output_minimum": float(np.min(guarded)),
                        "output_maximum": float(np.max(guarded)),
                    }
                )
    metrics = {
        "source_count": len(source_facts),
        "camera_model_count": len({(row["make"], row["model"]) for row in source_facts}),
        "render_count": len(render_rows),
        "minimum_native_srgb_out_of_gamut_fraction": min(
            row["native_srgb_out_of_gamut_fraction"] for row in source_facts
        ),
        "minimum_precompression_rec2020_out_of_gamut_fraction": min(
            row["precompression_rec2020_out_of_gamut_fraction"] for row in source_facts
        ),
        "maximum_luminance_preservation_absolute_error": max(
            row["maximum_luminance_preservation_absolute_error"] for row in source_facts
        ),
        "maximum_new_rgb16_boundary_fraction_vs_mapped_source": max(
            row["new_rgb16_boundary_fraction_vs_mapped_source"] for row in render_rows
        ),
        "population_median_style_retention_ratio": float(
            np.median([row["style_retention_ratio"] for row in render_rows])
        ),
        "worst_render_style_retention_ratio": min(
            row["style_retention_ratio"] for row in render_rows
        ),
        "population_median_residual_scale": float(
            np.median([row["residual_scale_median"] for row in render_rows])
        ),
        "population_median_fraction_scale_below_0p5": float(
            np.median([row["fraction_residual_scale_below_0p5"] for row in render_rows])
        ),
        "maximum_p999_gradient_ratio_vs_mapped_source": max(
            row["p999_gradient_ratio_vs_mapped_source"] for row in render_rows
        ),
        "maximum_adjacent_lstar_sign_inversion_fraction": max(
            row["adjacent_lstar_sign_inversion_fraction"] for row in render_rows
        ),
        "output_minimum": min(row["output_minimum"] for row in render_rows),
        "output_maximum": max(row["output_maximum"] for row in render_rows),
    }
    gates = config["automatic_gates"]
    checks = {
        "inventory": metrics["source_count"] == int(config["source"]["expected_rows"])
        and metrics["camera_model_count"] == int(config["source"]["expected_camera_models"])
        and metrics["render_count"]
        == int(config["source"]["expected_rows"])
        * len(config["render"]["styles"])
        * len(config["render"]["gamut_modes"]),
        "native_wide_gamut_support": metrics["minimum_native_srgb_out_of_gamut_fraction"]
        >= float(gates["minimum_native_srgb_out_of_gamut_fraction_per_source"]),
        "rec2020_compression_stress": metrics[
            "minimum_precompression_rec2020_out_of_gamut_fraction"
        ]
        >= float(gates["minimum_precompression_rec2020_out_of_gamut_fraction_per_source"]),
        "luminance_preservation": metrics["maximum_luminance_preservation_absolute_error"]
        <= float(gates["maximum_luminance_preservation_absolute_error"]),
        "finite_in_gamut_outputs": bool(
            np.isfinite(metrics["output_minimum"])
            and np.isfinite(metrics["output_maximum"])
            and metrics["output_minimum"] >= 0.0
            and metrics["output_maximum"] <= 1.0
        ),
        "new_rgb16_boundary": metrics[
            "maximum_new_rgb16_boundary_fraction_vs_mapped_source"
        ]
        <= float(gates["maximum_new_rgb16_boundary_fraction_vs_mapped_source"]),
        "population_style_retention": metrics["population_median_style_retention_ratio"]
        >= float(gates["minimum_population_median_style_retention_ratio"]),
        "worst_style_retention": metrics["worst_render_style_retention_ratio"]
        >= float(gates["minimum_worst_render_style_retention_ratio"]),
        "residual_scale": metrics["population_median_residual_scale"]
        >= float(gates["minimum_population_median_residual_scale"]),
        "low_scale_fraction": metrics["population_median_fraction_scale_below_0p5"]
        <= float(gates["maximum_population_median_fraction_scale_below_0p5"]),
        "gradient": metrics["maximum_p999_gradient_ratio_vs_mapped_source"]
        <= float(gates["maximum_p999_gradient_ratio_vs_mapped_source"]),
        "gradient_order": metrics["maximum_adjacent_lstar_sign_inversion_fraction"]
        <= float(gates["maximum_adjacent_lstar_sign_inversion_fraction"]),
    }
    automatic_pass = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "source_manifest_sha256": config["source"]["manifest_sha256"],
        "ingress_mapping_id": config["ingress"]["mapping_id"],
        "c3_mechanism_id": config["render"]["c3_mechanism_id"],
        "sources": source_facts,
        "rows": render_rows,
        "metrics": metrics,
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "automatic_pass": automatic_pass,
        "product_ingress_opened": False,
        "decision": config["branches"]["pass" if automatic_pass else "fail"],
        "production_default_changed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "NativeProPhotoError",
    "evaluate",
    "load_contract",
    "luminance_axis_interior_compress",
    "write_report",
]
