"""Fixed AO6 colour plus deterministic procedural FilmFX value ablation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.filmfx.compositor import composite_layers
from src.filmfx.effects import grain_residual_layer, halation_layer
from src.filmfx.fast_blur import gaussian_filter_safe


class AO6FilmFXValueError(ValueError):
    """Raised when a frozen parent, input, parameter, or output drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _inventory(root: Path, base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(base_dir.glob("*.png")):
        rows.append(
            {
                "id": path.stem,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def validate_contract(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("status") != "contract_frozen_before_render":
        raise AO6FilmFXValueError("contract is not frozen")
    parent = config["parent"]
    parent_path = root / str(parent["decision_path"])
    if sha256_file(parent_path) != parent["decision_sha256"]:
        raise AO6FilmFXValueError("AO7 parent decision drift")
    population = config["population"]
    base_dir = root / str(population["base_directory"])
    rows = _inventory(root, base_dir)
    if _canonical_sha(rows) != population["base_inventory_sha256"]:
        raise AO6FilmFXValueError("AO6 base inventory drift")
    ids = [row["id"] for row in rows]
    if ids != list(population["expected_ids"]):
        raise AO6FilmFXValueError("AO6 base population drift")
    arms = list(config["arms"])
    if [arm["id"] for arm in arms] != [
        "fixed_ao6_colour_only_t15_c35",
        "ao6_plus_mono_grain_012",
        "ao6_plus_mono_grain_012_simple_halation_014",
    ]:
        raise AO6FilmFXValueError("arm identity drift")
    if (
        arms[1].get("grain_strength") != 0.012
        or arms[1].get("grain_color") is not False
        or arms[2].get("halation_strength") != 0.14
    ):
        raise AO6FilmFXValueError("fixed FilmFX parameter drift")
    return rows


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            raise AO6FilmFXValueError(f"expected RGB PNG: {path}")
        return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb(path: Path, rgb: np.ndarray, *, compress_level: int) -> str:
    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
        or not np.all(np.isfinite(rgb))
        or np.any(rgb < 0.0)
        or np.any(rgb > 1.0)
    ):
        raise AO6FilmFXValueError("invalid FilmFX output")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.rint(rgb * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(
        path, format="PNG", compress_level=compress_level
    )
    return sha256_file(path)


def _effect_metrics(base: np.ndarray, output: np.ndarray) -> dict[str, float]:
    residual = output - base
    magnitude = np.max(np.abs(residual), axis=2)
    chroma = residual - residual.mean(axis=2, keepdims=True)
    high_frequency_chroma = chroma - gaussian_filter_safe(
        chroma, sigma=(1.2, 1.2, 0.0)
    )
    base_raw_boundary = np.any((base <= 0.0) | (base >= 1.0), axis=2)
    output_raw_boundary = np.any((output <= 0.0) | (output >= 1.0), axis=2)
    return {
        "changed_pixel_fraction": float(np.mean(magnitude > (0.5 / 255.0))),
        "median_abs_change": float(np.median(magnitude)),
        "max_abs_change": float(np.max(magnitude)),
        "new_raw_clipping_fraction": float(
            np.mean(output_raw_boundary & ~base_raw_boundary)
        ),
        "high_frequency_chroma_p999": float(
            np.quantile(np.max(np.abs(high_frequency_chroma), axis=2), 0.999)
        ),
    }


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    software_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    rows = validate_contract(root, config)
    execution = config["execution"]
    arms = list(config["arms"])
    records: list[dict[str, Any]] = []
    halation_supported_rows = 0
    for index, row in enumerate(rows):
        base_path = root / str(row["path"])
        base = _read_rgb(base_path)
        seed = int(execution["seed"]) + index * 1009
        grain = grain_residual_layer(
            base, strength=float(arms[1]["grain_strength"]), seed=seed, color=False
        )
        grain_output = composite_layers(
            base, [grain], output_margin=int(execution["output_margin_srgb8_codes"])
        )
        halo = halation_layer(base, strength=float(arms[2]["halation_strength"]))
        full_output = composite_layers(
            base,
            [grain, halo],
            output_margin=int(execution["output_margin_srgb8_codes"]),
        )
        if halo.alpha is not None and float(np.max(halo.alpha)) > 1e-4:
            halation_supported_rows += 1
        outputs = {
            arms[1]["id"]: grain_output,
            arms[2]["id"]: full_output,
        }
        for arm_id, output in outputs.items():
            relative = Path(str(arm_id)) / f"{row['id']}.png"
            output_sha = _save_rgb(
                output_dir / relative,
                output,
                compress_level=int(execution["png_compress_level"]),
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
                    **_effect_metrics(base, output),
                }
            )
    report = {
        "schema": "neuro-film.u5-r2bc0-ao6-procedural-filmfx-value.v1",
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
    gates = config["automatic_gates"]
    rows = list(report["records"])
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm_id"]), []).append(row)
    grain_id = "ao6_plus_mono_grain_012"
    full_id = "ao6_plus_mono_grain_012_simple_halation_014"
    expected = len(config["population"]["expected_ids"])
    if len(by_arm.get(grain_id, [])) != expected or len(by_arm.get(full_id, [])) != expected:
        raise AO6FilmFXValueError("incomplete effect report")
    worst_clip = max(float(row["new_raw_clipping_fraction"]) for row in rows)
    grain_changed = float(
        np.median([row["changed_pixel_fraction"] for row in by_arm[grain_id]])
    )
    grain_chroma = max(
        float(row["high_frequency_chroma_p999"]) for row in by_arm[grain_id]
    )
    automatic_gates = {
        "new_raw_clipping": worst_clip
        <= float(gates["maximum_new_raw_clipping_fraction"]),
        "grain_changed_support": grain_changed
        >= float(gates["minimum_median_changed_pixel_fraction_grain"]),
        "halation_support": int(report["halation_supported_rows"])
        >= int(gates["minimum_halation_supported_rows"]),
        "grain_chroma_speckle": grain_chroma
        <= float(gates["maximum_grain_high_frequency_chroma_p999"]),
    }
    return {
        "automatic_pass": all(automatic_gates.values()),
        "automatic_gates": automatic_gates,
        "worst_new_raw_clipping_fraction": worst_clip,
        "median_grain_changed_pixel_fraction": grain_changed,
        "worst_grain_high_frequency_chroma_p999": grain_chroma,
        "halation_supported_rows": int(report["halation_supported_rows"]),
    }


def build_blind_sheets(
    *,
    root: Path,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Build three pre-key A/B/C layouts for the fixed first-nine population."""

    validate_contract(root, config)
    records = {
        (str(row["sample_id"]), str(row["arm_id"])): row
        for row in report["records"]
    }
    ids = list(config["population"]["expected_ids"])[:9]
    arm_ids = [str(arm["id"]) for arm in config["arms"]]
    permutations = [
        [arm_ids[0], arm_ids[1], arm_ids[2]],
        [arm_ids[2], arm_ids[0], arm_ids[1]],
        [arm_ids[1], arm_ids[2], arm_ids[0]],
    ]
    labels = ["A", "B", "C"]
    mappings: list[dict[str, str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for round_index, order in enumerate(permutations, start=1):
        mapping = dict(zip(labels, order, strict=True))
        mappings.append(mapping)
        thumbs: list[list[Image.Image]] = []
        for sample_id in ids:
            row_images: list[Image.Image] = []
            for arm_id in order:
                if arm_id == arm_ids[0]:
                    path = root / config["population"]["base_directory"] / f"{sample_id}.png"
                else:
                    record = records[(sample_id, arm_id)]
                    path = output_dir.parent / str(record["output_path"])
                with Image.open(path) as image:
                    tile = ImageOps.contain(image.convert("RGB"), (420, 280))
                    canvas = Image.new("RGB", (440, 320), "white")
                    canvas.paste(tile, ((440 - tile.width) // 2, 28))
                    ImageDraw.Draw(canvas).text((8, 6), labels[len(row_images)], fill="black")
                    row_images.append(canvas)
            thumbs.append(row_images)
        sheet = Image.new("RGB", (1320, 320 * len(ids)), (230, 230, 230))
        for y, row_images in enumerate(thumbs):
            for x, tile in enumerate(row_images):
                sheet.paste(tile, (x * 440, y * 320))
        sheet_path = output_dir / f"blind_round_{round_index}.png"
        sheet.save(sheet_path, format="PNG", compress_level=6)
    return {
        "sample_ids": ids,
        "round_count": 3,
        "sheet_paths": [
            (output_dir / f"blind_round_{index}.png").as_posix()
            for index in range(1, 4)
        ],
        "mapping_commitment_sha256": _canonical_sha(mappings),
        "mappings": mappings,
    }
