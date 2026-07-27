from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, sobel
from skimage.color import rgb2lab

from src.real_film.gold_matrix_transplant import style_and_basic_residual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "u5-r2ad1-external-render-manifest-v1":
        raise ValueError("unexpected AD1 external manifest schema")
    return value


def _delta_e76(left: np.ndarray, right: np.ndarray) -> float:
    left_lab = rgb2lab(np.clip(np.asarray(left, dtype=np.float64), 0.0, 1.0))
    right_lab = rgb2lab(np.clip(np.asarray(right, dtype=np.float64), 0.0, 1.0))
    return float(np.mean(np.linalg.norm(left_lab - right_lab, axis=-1)))


def matched_local_contrast(
    spatial_off: np.ndarray,
    spatial_on: np.ndarray,
    control: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    off = np.asarray(spatial_off, dtype=np.float64)
    on = np.asarray(spatial_on, dtype=np.float64)
    if off.shape != on.shape or off.ndim != 3 or off.shape[-1] != 3:
        raise ValueError("paired RGB arrays must have the same HxWx3 shape")
    target = on - off
    baseline_energy = float(np.sum(target * target))
    alpha_min, alpha_max = map(float, control["alpha_bounds"])
    best: tuple[float, float, float, np.ndarray] | None = None
    for sigma in control["gaussian_sigma_pixels"]:
        blurred = gaussian_filter(off, sigma=(float(sigma), float(sigma), 0.0), mode="reflect")
        detail = off - blurred
        denominator = float(np.sum(detail * detail))
        alpha = 0.0 if denominator == 0.0 else float(np.sum(target * detail) / denominator)
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        candidate = off + alpha * detail
        error = float(np.sum((on - candidate) ** 2))
        row = (error, float(sigma), alpha, candidate)
        if best is None or row[0] < best[0]:
            best = row
    assert best is not None
    error, sigma, alpha, candidate = best
    explained = (
        0.0
        if baseline_energy == 0.0
        else float(np.clip(1.0 - error / baseline_energy, 0.0, 1.0))
    )
    return candidate, {
        "sigma_pixels": sigma,
        "alpha": alpha,
        "rgb_sse": error,
        "baseline_rgb_sse": baseline_energy,
        "explained_rgb_energy_fraction": explained,
    }


def _luma(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])


def _gradient_p99(values: np.ndarray) -> float:
    luma = _luma(values)
    magnitude = np.hypot(
        sobel(luma, axis=0, mode="reflect"),
        sobel(luma, axis=1, mode="reflect"),
    )
    return float(np.percentile(magnitude, 99.0))


def _chroma_hf_p99(values: np.ndarray) -> float:
    rgb = np.asarray(values, dtype=np.float64)
    opponent = np.stack(
        (rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]),
        axis=-1,
    )
    low = gaussian_filter(opponent, sigma=(1.0, 1.0, 0.0), mode="reflect")
    return float(np.percentile(np.linalg.norm(opponent - low, axis=-1), 99.0))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 1.0 if numerator <= 1e-12 else float("inf")
    return float(numerator / denominator)


def new_isolated_red_speckle_fraction(
    spatial_off: np.ndarray,
    spatial_on: np.ndarray,
) -> float:
    def mask(values: np.ndarray) -> np.ndarray:
        rgb = np.asarray(values, dtype=np.float64)
        dominance = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])
        local = median_filter(dominance, size=3, mode="reflect")
        return (dominance > 0.15) & ((dominance - local) > 0.08)

    return float(np.mean(mask(spatial_on) & ~mask(spatial_off)))


def paired_metrics(
    source: np.ndarray,
    spatial_off: np.ndarray,
    spatial_on: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    off = np.asarray(spatial_off, dtype=np.float64)
    on = np.asarray(spatial_on, dtype=np.float64)
    if source.shape != off.shape or off.shape != on.shape:
        raise ValueError("source/off/on shape mismatch")
    nuisance, nuisance_fit = matched_local_contrast(
        off, on, config["matched_local_contrast_control"]
    )
    epsilon = 0.5 / 255.0
    new_clip = ((on <= epsilon) & (off > epsilon)) | (
        (on >= 1.0 - epsilon) & (off < 1.0 - epsilon)
    )
    style, non_basic = style_and_basic_residual(
        source.reshape(-1, 3), on.reshape(-1, 3)
    )
    return {
        "shape": list(on.shape),
        "finite": bool(np.all(np.isfinite(on))),
        "bounded_0_1": bool(np.min(on) >= 0.0 and np.max(on) <= 1.0),
        "spatial_effect_delta_e76": _delta_e76(off, on),
        "residual_after_local_contrast_match_delta_e76": _delta_e76(nuisance, on),
        "local_contrast_fit": nuisance_fit,
        "new_hard_clipping_fraction_vs_spatial_off": float(np.mean(new_clip)),
        "luma_gradient_p99_off": _gradient_p99(off),
        "luma_gradient_p99_on": _gradient_p99(on),
        "luma_gradient_amplification_p99_ratio": _safe_ratio(
            _gradient_p99(on), _gradient_p99(off)
        ),
        "chroma_high_frequency_p99_off": _chroma_hf_p99(off),
        "chroma_high_frequency_p99_on": _chroma_hf_p99(on),
        "chroma_high_frequency_p99_ratio": _safe_ratio(
            _chroma_hf_p99(on), _chroma_hf_p99(off)
        ),
        "new_isolated_red_speckle_fraction": new_isolated_red_speckle_fraction(
            off, on
        ),
        "style_delta_e76_vs_input": style,
        "non_basic_residual_delta_e76_vs_input": non_basic,
    }


def evaluate_manifests(
    manifest_a: dict[str, Any],
    manifest_b: dict[str, Any],
    config: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    for manifest in (manifest_a, manifest_b):
        if manifest["external_revision"] != config["external_source"]["revision"]:
            raise ValueError("external revision mismatch")
        if manifest["python_version"] != config["runtime"]["python_version"]:
            raise ValueError("external Python version mismatch")
        if manifest["package_versions"] != config["runtime"]["package_versions"]:
            raise ValueError("external package version mismatch")
        if manifest["input_manifest_sha256"] != config["inputs"]["frozen_set_sha256"]:
            raise ValueError("input manifest mismatch")
    rows_a = {row["sample_id"]: row for row in manifest_a["records"]}
    rows_b = {row["sample_id"]: row for row in manifest_b["records"]}
    if set(rows_a) != set(rows_b) or len(rows_a) != config["inputs"]["expected_samples"]:
        raise ValueError("manifest sample population mismatch")

    records: list[dict[str, Any]] = []
    exact_repeat = True
    off_replay = True
    for sample_id in sorted(rows_a):
        first = rows_a[sample_id]
        second = rows_b[sample_id]
        for key in (
            "source_sha256",
            "spatial_off_npy_sha256",
            "spatial_on_npy_sha256",
            "spatial_off_png_sha256",
            "spatial_on_png_sha256",
        ):
            exact_repeat &= first[key] == second[key]
        off_replay &= (
            first["spatial_off_png_sha256"]
            == config["rf2_c0_spatial_off_png_sha256"][sample_id]
        )
        source_path = root / first["source_path"]
        if sha256_file(source_path) != first["source_sha256"]:
            raise ValueError(f"source hash mismatch: {sample_id}")
        source = np.load(root / first["source_npy_path"], allow_pickle=False)
        off = np.load(root / first["spatial_off_npy_path"], allow_pickle=False)
        on = np.load(root / first["spatial_on_npy_path"], allow_pickle=False)
        metrics = paired_metrics(source, off, on, config)
        records.append(
            {
                "sample_id": sample_id,
                "source_sha256": first["source_sha256"],
                "spatial_off_png_sha256": first["spatial_off_png_sha256"],
                "spatial_on_png_sha256": first["spatial_on_png_sha256"],
                "spatial_off_npy_sha256": first["spatial_off_npy_sha256"],
                "spatial_on_npy_sha256": first["spatial_on_npy_sha256"],
                **metrics,
            }
        )

    gates = config["automatic_gates"]
    median = lambda key: float(np.median([row[key] for row in records]))
    worst = lambda key: float(max(row[key] for row in records))
    summary = {
        "sample_count": len(records),
        "exact_repeat": bool(exact_repeat),
        "spatial_off_matches_rf2_c0_png_hashes": bool(off_replay),
        "all_outputs_finite": all(row["finite"] for row in records),
        "all_outputs_bounded_0_1": all(row["bounded_0_1"] for row in records),
        "gold_median_spatial_effect_delta_e76": median("spatial_effect_delta_e76"),
        "gold_median_residual_after_local_contrast_match_delta_e76": median(
            "residual_after_local_contrast_match_delta_e76"
        ),
        "worst_local_contrast_explained_rgb_energy_fraction": float(
            max(
                row["local_contrast_fit"]["explained_rgb_energy_fraction"]
                for row in records
            )
        ),
        "worst_new_hard_clipping_fraction_vs_spatial_off": worst(
            "new_hard_clipping_fraction_vs_spatial_off"
        ),
        "worst_luma_gradient_amplification_p99_ratio": worst(
            "luma_gradient_amplification_p99_ratio"
        ),
        "worst_chroma_high_frequency_p99_ratio": worst(
            "chroma_high_frequency_p99_ratio"
        ),
        "worst_new_isolated_red_speckle_fraction": worst(
            "new_isolated_red_speckle_fraction"
        ),
        "gold_median_style_delta_e76_vs_input": median("style_delta_e76_vs_input"),
        "gold_median_non_basic_residual_delta_e76_vs_input": median(
            "non_basic_residual_delta_e76_vs_input"
        ),
    }
    checks = {
        "exact_repeat": summary["exact_repeat"] is gates["exact_repeat_required"],
        "spatial_off_replay": summary["spatial_off_matches_rf2_c0_png_hashes"]
        is gates["spatial_off_must_match_rf2_c0_png_hashes"],
        "finite": summary["all_outputs_finite"] is gates["all_outputs_finite"],
        "bounded": summary["all_outputs_bounded_0_1"]
        is gates["all_outputs_bounded_0_1"],
        "effect_min": summary["gold_median_spatial_effect_delta_e76"]
        >= gates["minimum_gold_median_spatial_effect_delta_e76"],
        "effect_max": summary["gold_median_spatial_effect_delta_e76"]
        <= gates["maximum_gold_median_spatial_effect_delta_e76"],
        "local_residual": summary[
            "gold_median_residual_after_local_contrast_match_delta_e76"
        ]
        >= gates["minimum_gold_median_residual_after_local_contrast_match_delta_e76"],
        "local_explained": summary[
            "worst_local_contrast_explained_rgb_energy_fraction"
        ]
        <= gates["maximum_local_contrast_explained_rgb_energy_fraction"],
        "clipping": summary["worst_new_hard_clipping_fraction_vs_spatial_off"]
        <= gates["maximum_worst_new_hard_clipping_fraction_vs_spatial_off"],
        "gradient": summary["worst_luma_gradient_amplification_p99_ratio"]
        <= gates["maximum_worst_luma_gradient_amplification_p99_ratio"],
        "chroma_hf": summary["worst_chroma_high_frequency_p99_ratio"]
        <= gates["maximum_worst_chroma_high_frequency_p99_ratio"],
        "red_speckle": summary["worst_new_isolated_red_speckle_fraction"]
        <= gates["maximum_worst_new_isolated_red_speckle_fraction"],
        "style": summary["gold_median_style_delta_e76_vs_input"]
        >= gates["minimum_gold_median_style_delta_e76_vs_input"],
        "non_basic": summary["gold_median_non_basic_residual_delta_e76_vs_input"]
        >= gates["minimum_gold_median_non_basic_residual_delta_e76_vs_input"],
    }
    if not checks["exact_repeat"] or not checks["spatial_off_replay"]:
        decision = "close_runtime_or_replay_failure"
    elif not checks["effect_min"]:
        decision = "close_effect_too_weak"
    elif not checks["local_residual"] or not checks["local_explained"]:
        decision = "close_simple_local_contrast_equivalent"
    elif not all(checks.values()):
        decision = "close_artifact_or_structure_gate"
    else:
        decision = "automatic_pass_requires_blind_visual_review"
    return {
        "summary": summary,
        "gate_checks": checks,
        "decision": decision,
        "visual_review_allowed": decision
        == "automatic_pass_requires_blind_visual_review",
        "records": records,
    }
