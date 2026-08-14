"""U1.4C3 source-anchored analytical Rec.2020 residual execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.rec2020_safe_lab import apply_rec2020_safe_lab
from src.color_engine.srgb_transfer import encoded_srgb_to_linear
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.preprocess.color_management import (
    convert_linear_rgb,
    linear_rec2020_to_rec2020,
)
from src.preprocess.output_encode import save_rec2020_16_png
from src.preprocess.types import SourceProfile, WorkingImage

SCHEMA = "neuro_film.u1-4c3-rec2020-source-anchored-interior-contract.v1"
REPORT_SCHEMA = "neuro_film.u1-4c3-rec2020-source-anchored-interior-report.v1"
EXPERIMENT_ID = "U1.4C3"
CONTRACT_SHA256 = "9b4996b8aa383081775cb000c079ef35390a5f38022a9f7f685199aa5218527b"
RGB16_BOUNDARY_EPSILON = 1.0 / 65536.0


class Rec2020InteriorError(RuntimeError):
    """Raised when the frozen C3 contract or execution invariant fails."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise Rec2020InteriorError("C3 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise Rec2020InteriorError("C3 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise Rec2020InteriorError("C3 contract structure drift")
    _relative_path(payload["source"]["manifest"])
    for parent in (payload["parents"]["c2_result"], payload["parents"]["c2_contract"]):
        _relative_path(parent["path"])
    return payload


def source_anchored_interior_residual(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each RGB residual to the source-anchored RGB16-safe interval."""

    if (
        not isinstance(source, np.ndarray)
        or not isinstance(candidate, np.ndarray)
        or source.dtype != np.float32
        or candidate.dtype != np.float32
        or source.ndim != 3
        or source.shape[-1] != 3
        or candidate.shape != source.shape
        or not np.isfinite(source).all()
        or not np.isfinite(candidate).all()
        or float(np.min(source)) < 0.0
        or float(np.max(source)) > 1.0
        or float(np.min(candidate)) < 0.0
        or float(np.max(candidate)) > 1.0
        or not np.isfinite(margin)
        or margin <= 0.0
        or margin >= 0.5
    ):
        raise Rec2020InteriorError("C3 source, candidate, or margin is invalid")
    frozen_source = source.copy()
    frozen_candidate = candidate.copy()
    source64 = source.astype(np.float64)
    residual = candidate.astype(np.float64) - source64
    lower = np.minimum(source64, margin)
    upper = np.maximum(source64, 1.0 - margin)
    scale = np.ones(source.shape[:2], dtype=np.float64)
    for channel in range(3):
        value = residual[..., channel]
        positive = (value > 0.0) & (source64[..., channel] + value > upper[..., channel])
        negative = (value < 0.0) & (source64[..., channel] + value < lower[..., channel])
        bound = np.ones_like(value)
        bound[positive] = (
            upper[..., channel][positive] - source64[..., channel][positive]
        ) / value[positive]
        bound[negative] = (
            lower[..., channel][negative] - source64[..., channel][negative]
        ) / value[negative]
        scale = np.minimum(scale, bound)
    scale = np.minimum(np.maximum(scale, 0.0), 1.0)
    output = np.asarray(source64 + scale[..., None] * residual, dtype=np.float32)
    if (
        not np.array_equal(source, frozen_source)
        or not np.array_equal(candidate, frozen_candidate)
        or not np.isfinite(output).all()
        or float(np.min(output)) < 0.0
        or float(np.max(output)) > 1.0
        or np.any(output < lower.astype(np.float32) - np.float32(2e-7))
        or np.any(output > upper.astype(np.float32) + np.float32(2e-7))
    ):
        raise Rec2020InteriorError("C3 analytical execution invariant failed")
    return output, np.asarray(scale, dtype=np.float32)


def _working(pixels: np.ndarray, source_path: Path) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(pixels, dtype=np.float32),
        working_space="linear_rec2020",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile("assumed_srgb", "sRGB display proxy converted to BT.2020"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=8,
        source_path=source_path,
    )


def _style_kwargs(style: str, mode: str, root: Path) -> dict[str, Any]:
    stats_payload = json.loads((root / "configs/film_color_stats.json").read_text(encoding="utf-8"))
    guards_payload = json.loads((root / "configs/color_guardrails.json").read_text(encoding="utf-8"))
    stats = stats_payload["styles"][style]
    guardrails = dict(guards_payload["defaults"])
    guardrails.update(guards_payload["styles"].get(style, {}))
    return {
        "destination_mean": np.asarray(stats["mean"], dtype=np.float32),
        "destination_std": np.asarray(stats["std"], dtype=np.float32),
        "style": style,
        "strength": 0.35,
        "luma_strength": 0.02,
        "gamut_mode": mode,
        "tone_rolloff": 0.04,
        "shadow_floor_l": 1.0,
        "highlight_ceiling_l": 99.0,
        "preserve_luma_detail_strength": 0.90,
        "chroma_curve_strength": 0.45,
        "neutral_protect": guardrails["neutral_protect"],
        "skin_protect": guardrails["skin_protect"],
        "max_chroma_gain": guardrails.get("max_chroma_gain"),
        "max_chroma_boost": guardrails.get("max_chroma_boost"),
        "max_chroma_absolute": guardrails.get("max_chroma_absolute"),
    }


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _median_lab_delta_e76(left_lab: np.ndarray, right_lab: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(left_lab - right_lab, axis=-1)))


def _new_boundary_fraction(source: np.ndarray, output: np.ndarray) -> float:
    source_boundary = np.any(
        (source <= RGB16_BOUNDARY_EPSILON) | (source >= 1.0 - RGB16_BOUNDARY_EPSILON),
        axis=-1,
    )
    output_boundary = np.any(
        (output <= RGB16_BOUNDARY_EPSILON) | (output >= 1.0 - RGB16_BOUNDARY_EPSILON),
        axis=-1,
    )
    return float(np.mean(output_boundary & ~source_boundary))


def _save_and_verify(working: WorkingImage, output: Path) -> str:
    save_rec2020_16_png(working, output)
    decoded = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    expected = np.rint(linear_rec2020_to_rec2020(working.pixels) * 65535.0).astype(np.uint16)
    if decoded is None or decoded.dtype != np.uint16 or decoded.shape != expected.shape:
        raise Rec2020InteriorError("C3 RGB16 PNG readback structure drift")
    if not np.array_equal(decoded[..., ::-1], expected):
        raise Rec2020InteriorError("C3 RGB16 PNG samples differ from encoded array")
    return hash_file(output)


def _validate_inputs(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    for parent in (config["parents"]["c2_result"], config["parents"]["c2_contract"]):
        path = root / _relative_path(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise Rec2020InteriorError("C3 parent identity drift")
    for parent in config["parents"]["operator_files"]:
        path = root / _relative_path(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise Rec2020InteriorError("C3 operator identity drift")
    source = config["source"]
    manifest_path = root / _relative_path(source["manifest"])
    if not manifest_path.is_file() or hash_file(manifest_path) != source["manifest_sha256"]:
        raise Rec2020InteriorError("C3 source manifest identity drift")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(rows, list)
        or len(rows) != int(source["expected_rows"])
        or len({row["id"] for row in rows}) != len(rows)
        or len({row["make"] for row in rows}) != int(source["expected_camera_makes"])
    ):
        raise Rec2020InteriorError("C3 source inventory drift")
    for row in rows:
        if (
            row.get("allowed_use") != source["required_allowed_use"]
            or row.get("rights_scope") != source["required_rights_scope"]
            or row.get("decoded_color_state") != source["required_color_state"]
        ):
            raise Rec2020InteriorError("C3 source policy drift")
    return [dict(row) for row in rows]


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    rows = _validate_inputs(config, root)
    if output_dir.exists():
        raise Rec2020InteriorError("C3 output directory must be create-only")
    output_dir.mkdir(parents=True)
    margin = float(config["mechanism"]["rgb16_interior_margin"])
    render_rows: list[dict[str, Any]] = []
    for source_row in rows:
        source_path = root / _relative_path(source_row["decoded_path"])
        if not source_path.is_file() or hash_file(source_path) != source_row["decoded_sha256"]:
            raise Rec2020InteriorError("C3 decoded source identity drift")
        with Image.open(source_path) as image:
            image.load()
            encoded = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
        source_linear_srgb = encoded_srgb_to_linear(np.asarray(encoded, dtype=np.float32))
        source = convert_linear_rgb(
            source_linear_srgb,
            source_space="linear_srgb",
            destination_space="linear_rec2020",
        )
        if float(np.min(source)) < 0.0 or float(np.max(source)) > 1.0:
            raise Rec2020InteriorError("C3 converted source is outside Rec.2020")
        working = _working(source, source_path)
        source_lab = linear_rgb_to_lab(source, working_space="linear_rec2020")
        for style in config["render"]["styles"]:
            for mode in config["render"]["gamut_modes"]:
                candidate = apply_rec2020_safe_lab(
                    working, **_style_kwargs(style, mode, root)
                ).pixels
                guarded, scale = source_anchored_interior_residual(
                    source, candidate, margin=margin
                )
                candidate_lab = linear_rgb_to_lab(
                    candidate, working_space="linear_rec2020"
                )
                guarded_lab = linear_rgb_to_lab(guarded, working_space="linear_rec2020")
                baseline_style = _median_lab_delta_e76(source_lab, candidate_lab)
                guarded_style = _median_lab_delta_e76(source_lab, guarded_lab)
                style_retention = (
                    guarded_style / baseline_style if baseline_style > 1e-12 else 1.0
                )
                relative_path = Path("renders") / style / mode / f"{source_row['id']}.png"
                output_path = output_dir / relative_path
                output_sha = _save_and_verify(_working(guarded, source_path), output_path)
                render_rows.append(
                    {
                        "id": source_row["id"],
                        "make": source_row["make"],
                        "style": style,
                        "gamut_mode": mode,
                        "source_sha256": source_row["decoded_sha256"],
                        "shape": list(source.shape),
                        "source_array_sha256": _array_sha256(source),
                        "candidate_array_sha256": _array_sha256(candidate),
                        "guarded_array_sha256": _array_sha256(guarded),
                        "output_path": relative_path.as_posix(),
                        "output_sha256": output_sha,
                        "baseline_style_delta_e76": baseline_style,
                        "guarded_style_delta_e76": guarded_style,
                        "style_retention_ratio": style_retention,
                        "residual_scale_median": float(np.median(scale)),
                        "fraction_residual_scale_below_0p5": float(np.mean(scale < 0.5)),
                        "new_rgb16_boundary_fraction_vs_source": _new_boundary_fraction(
                            source, guarded
                        ),
                        "p999_gradient_ratio_vs_source": _gradient_p999_ratio(source, guarded),
                        "adjacent_lstar_sign_inversion_fraction": _gradient_inversion_fraction(
                            source_lab[..., 0], guarded_lab[..., 0], epsilon=0.02
                        ),
                        "output_minimum": float(np.min(guarded)),
                        "output_maximum": float(np.max(guarded)),
                    }
                )
    metrics = {
        "render_count": len(render_rows),
        "source_count": len({row["id"] for row in render_rows}),
        "camera_make_count": len({row["make"] for row in render_rows}),
        "maximum_new_rgb16_boundary_fraction_vs_source": max(
            row["new_rgb16_boundary_fraction_vs_source"] for row in render_rows
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
        "maximum_p999_gradient_ratio_vs_source": max(
            row["p999_gradient_ratio_vs_source"] for row in render_rows
        ),
        "maximum_adjacent_lstar_sign_inversion_fraction": max(
            row["adjacent_lstar_sign_inversion_fraction"] for row in render_rows
        ),
        "output_minimum": min(row["output_minimum"] for row in render_rows),
        "output_maximum": max(row["output_maximum"] for row in render_rows),
    }
    gates = config["automatic_gates"]
    checks = {
        "inventory": metrics["render_count"]
        == len(config["render"]["styles"])
        * len(config["render"]["gamut_modes"])
        * int(config["source"]["expected_rows"]),
        "finite_in_gamut_outputs": bool(
            np.isfinite(metrics["output_minimum"])
            and np.isfinite(metrics["output_maximum"])
            and metrics["output_minimum"] >= 0.0
            and metrics["output_maximum"] <= 1.0
        ),
        "new_rgb16_boundary": metrics["maximum_new_rgb16_boundary_fraction_vs_source"]
        <= float(gates["maximum_new_rgb16_boundary_fraction_vs_source"]),
        "population_style_retention": metrics["population_median_style_retention_ratio"]
        >= float(gates["minimum_population_median_style_retention_ratio"]),
        "worst_style_retention": metrics["worst_render_style_retention_ratio"]
        >= float(gates["minimum_worst_render_style_retention_ratio"]),
        "residual_scale": metrics["population_median_residual_scale"]
        >= float(gates["minimum_population_median_residual_scale"]),
        "low_scale_fraction": metrics["population_median_fraction_scale_below_0p5"]
        <= float(gates["maximum_population_median_fraction_scale_below_0p5"]),
        "gradient": metrics["maximum_p999_gradient_ratio_vs_source"]
        <= float(gates["maximum_p999_gradient_ratio_vs_source"]),
        "gradient_order": metrics["maximum_adjacent_lstar_sign_inversion_fraction"]
        <= float(gates["maximum_adjacent_lstar_sign_inversion_fraction"]),
    }
    automatic_pass = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "source_manifest_sha256": config["source"]["manifest_sha256"],
        "mechanism_id": config["mechanism"]["id"],
        "rows": render_rows,
        "metrics": metrics,
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "automatic_pass": automatic_pass,
        "independent_confirmation_opened": automatic_pass,
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
    "Rec2020InteriorError",
    "evaluate",
    "load_contract",
    "source_anchored_interior_residual",
    "write_report",
]
