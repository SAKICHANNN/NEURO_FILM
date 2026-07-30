"""Shared optical-density grain ablation on fixed AO6 colour outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.ao6_procedural_filmfx_value import (
    _inventory,
    _read_rgb,
    _save_rgb,
    sha256_file,
)
from src.filmfx.compositor import composite_layers
from src.filmfx.effects import halation_layer
from src.filmfx.fast_blur import gaussian_filter_safe


class AO6DensityGrainError(ValueError):
    """Raised when the frozen density-grain contract or evidence drifts."""


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_contract(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("status") != "contract_frozen_before_render":
        raise AO6DensityGrainError("contract is not frozen")
    parent = config["parent"]
    parent_path = root / str(parent["decision_path"])
    if sha256_file(parent_path) != parent["decision_sha256"]:
        raise AO6DensityGrainError("BC0 parent decision drift")
    parent_decision = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent_decision["decision"] != parent["required_decision"]:
        raise AO6DensityGrainError("BC0 required decision drift")
    population = config["population"]
    rows = _inventory(root, root / str(population["base_directory"]))
    if _canonical_sha(rows) != population["base_inventory_sha256"]:
        raise AO6DensityGrainError("AO6 base inventory drift")
    if [row["id"] for row in rows] != list(population["expected_ids"]):
        raise AO6DensityGrainError("AO6 population drift")
    candidate = config["candidate"]
    if (
        candidate["density_sigma"] != 0.0225
        or candidate["halation_strength"] != 0.14
        or len(config["arms"]) != 3
    ):
        raise AO6DensityGrainError("BC1 candidate drift")
    return rows


def apply_linear_density_grain(
    encoded_rgb: np.ndarray,
    *,
    density_sigma: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one achromatic density field with an analytical bright-end guard."""

    encoded = np.asarray(encoded_rgb, dtype=np.float32)
    if (
        encoded.ndim != 3
        or encoded.shape[-1] != 3
        or not np.all(np.isfinite(encoded))
        or np.any(encoded < 0.0)
        or np.any(encoded > 1.0)
        or density_sigma <= 0.0
    ):
        raise AO6DensityGrainError("invalid density-grain input")
    linear = encoded_srgb_to_linear(encoded)
    luma = (
        encoded[..., 0] * 0.2126
        + encoded[..., 1] * 0.7152
        + encoded[..., 2] * 0.0722
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=encoded.shape[:2]).astype(np.float32)
    noise -= gaussian_filter_safe(noise, sigma=1.2)
    noise -= np.float32(noise.mean())
    noise /= np.float32(max(float(noise.std()), 1e-8))
    envelope = np.float32(0.45) + np.float32(0.75) * (
        np.float32(1.0) - luma
    )
    requested_density = noise * envelope * np.float32(density_sigma)

    max_channel = np.max(linear, axis=2)
    brighten = requested_density < 0.0
    allowed_brighten = np.zeros_like(requested_density)
    positive = max_channel > 0.0
    allowed_brighten[positive] = np.log10(
        np.float32(1.0) / max_channel[positive]
    )
    requested_brighten = np.maximum(-requested_density, np.float32(1e-20))
    scale = np.ones_like(requested_density)
    scale[brighten] = np.minimum(
        np.float32(1.0),
        allowed_brighten[brighten] / requested_brighten[brighten],
    )
    applied_density = requested_density * scale
    factor = np.power(
        np.float32(10.0), -applied_density, dtype=np.float32
    )
    output_linear = linear * factor[..., None]
    if np.any(output_linear < 0.0) or np.any(output_linear > 1.0 + 2e-6):
        raise AO6DensityGrainError("analytical density guard escaped cube")
    output_linear = np.minimum(output_linear, np.float32(1.0))
    output = linear_srgb_to_encoded(output_linear)

    source_sum = np.sum(linear, axis=2)
    output_sum = np.sum(output_linear, axis=2)
    eligible = source_sum > 1e-7
    source_chroma = linear[eligible] / source_sum[eligible, None]
    output_chroma = output_linear[eligible] / output_sum[eligible, None]
    chroma_drift = np.max(np.abs(source_chroma - output_chroma), axis=1)
    encoded_change = np.max(np.abs(output - encoded), axis=2)
    source_boundary = np.any((encoded <= 0.0) | (encoded >= 1.0), axis=2)
    output_boundary = np.any((output <= 0.0) | (output >= 1.0), axis=2)
    metrics = {
        "changed_pixel_fraction": float(
            np.mean(encoded_change > (0.5 / 255.0))
        ),
        "p999_abs_encoded_change": float(np.quantile(encoded_change, 0.999)),
        "max_abs_encoded_change": float(np.max(encoded_change)),
        "bright_guard_limited_fraction": float(np.mean(scale < 0.999999)),
        "linear_chromaticity_drift_p999": float(
            np.quantile(chroma_drift, 0.999)
        ),
        "linear_chromaticity_drift_max": float(np.max(chroma_drift)),
        "new_raw_clipping_fraction": float(
            np.mean(output_boundary & ~source_boundary)
        ),
    }
    return output.astype(np.float32), metrics


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    software_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    rows = validate_contract(root, config)
    arms = list(config["arms"])
    candidate = config["candidate"]
    records: list[dict[str, Any]] = []
    halation_supported_rows = 0
    for index, row in enumerate(rows):
        base = _read_rgb(root / str(row["path"]))
        seed = int(config["execution"]["seed"]) + index * 1009
        grain, metrics = apply_linear_density_grain(
            base, density_sigma=float(candidate["density_sigma"]), seed=seed
        )
        halo = halation_layer(
            grain, strength=float(candidate["halation_strength"])
        )
        full = composite_layers(grain, [halo], output_margin=0)
        if halo.alpha is not None and float(np.max(halo.alpha)) > 1e-4:
            halation_supported_rows += 1
        for arm_id, output in ((arms[1], grain), (arms[2], full)):
            relative = Path(str(arm_id)) / f"{row['id']}.png"
            output_sha = _save_rgb(
                output_dir / relative,
                output,
                compress_level=int(config["execution"]["png_compress_level"]),
            )
            records.append(
                {
                    "sample_id": row["id"],
                    "arm_id": arm_id,
                    "base_path": row["path"],
                    "base_sha256": row["sha256"],
                    "output_path": relative.as_posix(),
                    "output_sha256": output_sha,
                    "seed": seed,
                    "density_grain_metrics": metrics,
                }
            )
    report = {
        "schema": "neuro-film.u5-r2bc1-ao6-density-grain-value.v1",
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "halation_supported_rows": halation_supported_rows,
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha(report)
    return report


def evaluate_report(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    arms = list(config["arms"])
    grain_rows = [
        row for row in report["records"] if row["arm_id"] == arms[1]
    ]
    if len(grain_rows) != len(config["population"]["expected_ids"]):
        raise AO6DensityGrainError("incomplete density-grain report")
    metrics = [row["density_grain_metrics"] for row in grain_rows]
    gates = config["automatic_gates"]
    worst_clip = max(row["new_raw_clipping_fraction"] for row in metrics)
    worst_chroma = max(
        row["linear_chromaticity_drift_p999"] for row in metrics
    )
    median_changed = float(
        np.median([row["changed_pixel_fraction"] for row in metrics])
    )
    median_limited = float(
        np.median([row["bright_guard_limited_fraction"] for row in metrics])
    )
    worst_change = max(row["p999_abs_encoded_change"] for row in metrics)
    automatic_gates = {
        "new_raw_clipping": worst_clip
        <= gates["maximum_new_raw_clipping_fraction"],
        "linear_chromaticity": worst_chroma
        <= gates["maximum_linear_chromaticity_drift_p999"],
        "changed_support": median_changed
        >= gates["minimum_median_changed_pixel_fraction"],
        "bright_guard_support": median_limited
        <= gates["maximum_median_bright_guard_limited_fraction"],
        "encoded_tail": worst_change
        <= gates["maximum_worst_p999_abs_encoded_change"],
        "halation_support": report["halation_supported_rows"]
        >= gates["minimum_halation_supported_rows"],
    }
    return {
        "automatic_pass": all(automatic_gates.values()),
        "automatic_gates": automatic_gates,
        "worst_new_raw_clipping_fraction": worst_clip,
        "worst_linear_chromaticity_drift_p999": worst_chroma,
        "median_changed_pixel_fraction": median_changed,
        "median_bright_guard_limited_fraction": median_limited,
        "worst_p999_abs_encoded_change": worst_change,
        "halation_supported_rows": report["halation_supported_rows"],
    }


def build_blind_sheets(
    *,
    root: Path,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    records = {
        (str(row["sample_id"]), str(row["arm_id"])): row
        for row in report["records"]
    }
    ids = list(config["population"]["expected_ids"])[:9]
    arms = list(config["arms"])
    permutations = [
        [arms[0], arms[1], arms[2]],
        [arms[2], arms[0], arms[1]],
        [arms[1], arms[2], arms[0]],
    ]
    labels = ["A", "B", "C"]
    mappings: list[dict[str, str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for round_index, order in enumerate(permutations, start=1):
        mappings.append(dict(zip(labels, order, strict=True)))
        sheet = Image.new("RGB", (1320, 320 * len(ids)), (230, 230, 230))
        for y, sample_id in enumerate(ids):
            for x, arm_id in enumerate(order):
                if arm_id == arms[0]:
                    path = (
                        root
                        / str(config["population"]["base_directory"])
                        / f"{sample_id}.png"
                    )
                else:
                    path = output_dir.parent / records[(sample_id, arm_id)][
                        "output_path"
                    ]
                with Image.open(path) as image:
                    tile = ImageOps.contain(image.convert("RGB"), (420, 280))
                canvas = Image.new("RGB", (440, 320), "white")
                canvas.paste(tile, ((440 - tile.width) // 2, 28))
                ImageDraw.Draw(canvas).text((8, 6), labels[x], fill="black")
                sheet.paste(canvas, (x * 440, y * 320))
        sheet.save(
            output_dir / f"blind_round_{round_index}.png",
            format="PNG",
            compress_level=6,
        )
    return {
        "sample_ids": ids,
        "mappings": mappings,
        "mapping_commitment_sha256": _canonical_sha(mappings),
    }
