"""Frozen U5.R2AI0 composition of two retained global colour operators."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.pipeline_color_baseline import (
    apply_output_margin,
    style_transfer_rgb,
)
from src.color_engine.safe_lab import SafeLabSourceContext
from src.color_engine.safe_lab_rgb_context import (
    style_transfer_rgb_with_source_context,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import (
    load_frozen_samples,
    new_hard_clipping_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.density_domain import operator_from_config


class CompositionFrontierError(ValueError):
    """Raised when the frozen composition contract or evidence drifts."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(root: Path, path: str, expected: str) -> Path:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise CompositionFrontierError(f"hash mismatch: {path}")
    return resolved


def candidate_bank(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    bank = config["candidate_bank"]
    candidates = []
    for order in bank["orders"]:
        for strength in bank["density_strengths"]:
            candidates.append(
                {
                    "candidate_id": str(bank["candidate_id_format"]).format(
                        order=order,
                        strength_percent=int(round(float(strength) * 100.0)),
                    ),
                    "order": str(order),
                    "density_strength": float(strength),
                }
            )
    if len(candidates) != int(bank["candidate_count"]):
        raise CompositionFrontierError("candidate count drift")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise CompositionFrontierError("duplicate candidate identity")
    if {row["order"] for row in candidates} != {
        "anchor_then_density",
        "density_then_anchor",
    }:
        raise CompositionFrontierError("unsupported composition order")
    return candidates


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if config["status"] != "contract_frozen_implementation_ready":
        raise CompositionFrontierError("contract is not frozen")
    samples = load_frozen_samples(root, config)
    anchor = config["parent_anchor"]
    density = config["parent_density"]
    implementation = config["implementation_inputs"]
    anchor_config = _load_json(
        _require_hash(root, anchor["config"], anchor["config_sha256"])
    )
    density_config = _load_json(
        _require_hash(root, density["config"], density["config_sha256"])
    )
    density_operator_config = _load_json(
        _require_hash(
            root,
            density["operator_config"],
            density["operator_config_sha256"],
        )
    )
    anchor_manifest = _load_json(
        _require_hash(root, anchor["manifest"], anchor["manifest_sha256"])
    )
    density_manifest = _load_json(
        _require_hash(root, density["manifest"], density["manifest_sha256"])
    )
    for key in ("anchor_renderer", "density_renderer", "density_operator"):
        _require_hash(
            root,
            implementation[key],
            implementation[f"{key}_sha256"],
        )
    stats = _load_json(
        _require_hash(
            root,
            implementation["film_color_stats"],
            implementation["film_color_stats_sha256"],
        )
    )
    anchor_decision = _load_json(root / anchor["decision"])
    density_decision = _load_json(root / density["decision"])
    if anchor_decision["qualified_candidate"] != anchor["candidate_id"]:
        raise CompositionFrontierError("anchor parent is not qualified")
    if density_decision["retained_candidate"] != density["candidate_id"]:
        raise CompositionFrontierError("density parent is not retained")
    if anchor_config["frozen_set_sha256"] != config["frozen_set_sha256"]:
        raise CompositionFrontierError("anchor frozen-set drift")
    if density_config["frozen_set_sha256"] != config["frozen_set_sha256"]:
        raise CompositionFrontierError("density frozen-set drift")
    if anchor["style"] not in stats["styles"]:
        raise CompositionFrontierError("anchor style statistics missing")
    witness_id = str(density["witness_id"])
    if witness_id not in density_operator_config["witnesses"]:
        raise CompositionFrontierError("density witness missing")
    return {
        "samples": samples,
        "candidates": candidate_bank(config),
        "anchor_stats": stats["styles"][anchor["style"]],
        "density_operator_config": density_operator_config,
        "anchor_manifest": anchor_manifest,
        "density_manifest": density_manifest,
    }


def build_operators(
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
) -> tuple[Any, Any]:
    anchor = config["parent_anchor"]
    density = config["parent_density"]
    density_config = validated["density_operator_config"]
    operator = operator_from_config(
        density_config["witnesses"][density["witness_id"]],
        exposure_floor=float(density_config["exposure_floor"]),
        matrix_minimum_determinant=float(
            density_config["parameter_bounds"]["matrix_minimum_determinant"]
        ),
        minimum_endpoint_span=float(
            density_config["parameter_bounds"]["minimum_endpoint_span"]
        ),
    )

    def apply_anchor(encoded: np.ndarray) -> np.ndarray:
        return style_transfer_rgb(
            np.asarray(encoded, dtype=np.float32),
            validated["anchor_stats"],
            str(anchor["style"]),
            float(anchor["strength"]),
            float(anchor["luma_strength"]),
            float(anchor["grain"]),
            int(anchor["seed"]),
            True,
            gamut_mode=str(anchor["gamut_mode"]),
            output_margin=0,
        )

    def apply_density(encoded: np.ndarray, strength: float) -> np.ndarray:
        linear = encoded_srgb_to_linear(np.asarray(encoded, dtype=np.float64))
        return linear_srgb_to_encoded(
            operator.apply(linear, strength=float(strength))
        )

    return apply_anchor, apply_density


def build_anchor_operator_with_source_context(
    config: Mapping[str, Any],
    validated: Mapping[str, Any],
    source_context: SafeLabSourceContext,
) -> Any:
    """Build the frozen anchor while reusing one explicit source context."""

    anchor = config["parent_anchor"]

    def apply_anchor(encoded: np.ndarray) -> np.ndarray:
        return style_transfer_rgb_with_source_context(
            np.asarray(encoded, dtype=np.float32),
            validated["anchor_stats"],
            str(anchor["style"]),
            float(anchor["strength"]),
            float(anchor["luma_strength"]),
            float(anchor["grain"]),
            int(anchor["seed"]),
            True,
            gamut_mode=str(anchor["gamut_mode"]),
            output_margin=0,
            source_context=source_context,
        )

    return apply_anchor


def compose_rgb(
    encoded: np.ndarray,
    *,
    order: str,
    density_strength: float,
    apply_anchor: Any,
    apply_density: Any,
    output_margin: int,
) -> np.ndarray:
    source = np.asarray(encoded)
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise CompositionFrontierError("source must be finite HxWx3 in [0,1]")
    if order == "anchor_then_density":
        result = apply_density(apply_anchor(source), density_strength)
    elif order == "density_then_anchor":
        result = apply_anchor(apply_density(source, density_strength))
    else:
        raise CompositionFrontierError("unsupported composition order")
    if not np.all(np.isfinite(result)):
        raise CompositionFrontierError("composition produced non-finite values")
    result = apply_output_margin(
        np.asarray(result, dtype=np.float64),
        int(output_margin),
    )
    return np.asarray(result, dtype=np.float32)


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    apply_anchor, apply_density = build_operators(config, validated)
    records = []
    for candidate in validated["candidates"]:
        destination_dir = output_dir / candidate["candidate_id"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, sample in validated["samples"].items():
            source_path = root / str(sample["source_path"])
            if sha256_file(source_path) != str(sample["source_sha256"]):
                raise CompositionFrontierError(f"source hash mismatch: {sample_id}")
            with Image.open(source_path) as image:
                source = (
                    np.asarray(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        dtype=np.float32,
                    )
                    / 255.0
                )
            result = compose_rgb(
                source,
                order=candidate["order"],
                density_strength=candidate["density_strength"],
                apply_anchor=apply_anchor,
                apply_density=apply_density,
                output_margin=int(config["candidate_bank"]["final_output_margin"]),
            )
            pixels = np.rint(result * 255.0).astype(np.uint8)
            destination = destination_dir / f"{sample_id}.png"
            Image.fromarray(pixels, mode="RGB").save(
                destination,
                format="PNG",
                compress_level=6,
            )
            records.append(
                {
                    **candidate,
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
                    "output": f"{candidate['candidate_id']}/{sample_id}.png",
                    "output_sha256": sha256_file(destination),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": config["software_commit"],
        "frozen_set_sha256": config["frozen_set_sha256"],
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(validated["samples"]),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_bytes(encoded)
    temporary.replace(output_dir / "manifest.json")
    return {
        "manifest_path": output_dir / "manifest.json",
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def evaluate_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    manifest = _load_json(manifest_path)
    expected = {
        (row["candidate_id"], sample_id)
        for row in validated["candidates"]
        for sample_id in validated["samples"]
    }
    records = {}
    for row in manifest.get("records", []):
        key = (str(row["candidate_id"]), str(row["sample_id"]))
        if key not in expected or key in records:
            raise CompositionFrontierError(f"invalid manifest record: {key}")
        records[key] = row
    if records.keys() != expected:
        raise CompositionFrontierError("incomplete manifest")

    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries = {}
    for candidate in validated["candidates"]:
        per_image = []
        for sample_id, sample in validated["samples"].items():
            record = records[(candidate["candidate_id"], sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            if sha256_file(output_path) != record["output_sha256"]:
                raise CompositionFrontierError("output hash mismatch")
            with Image.open(output_path) as output_image:
                output = ImageOps.exif_transpose(output_image)
                with Image.open(root / str(sample["source_path"])) as source_image:
                    source_size = ImageOps.exif_transpose(source_image).size
                if output.mode != "RGB" or output.size != source_size:
                    raise CompositionFrontierError("output decode/dimension mismatch")
            source_pixels = sample_rgb_image(
                root / str(sample["source_path"]), budget
            )
            output_pixels = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(
                source_pixels, output_pixels
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source_pixels, output_pixels, epsilon
                    ),
                    "output_sha256": record["output_sha256"],
                }
            )
        gold = [row for row in per_image if row["split"] == "gold"]
        stress = [row for row in per_image if row["split"] == "stress"]
        summary = {
            "order": candidate["order"],
            "density_strength": candidate["density_strength"],
            "gold_median_style_delta_e76": float(
                np.median([row["median_style_delta_e76"] for row in gold])
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median(
                    [row["median_non_basic_residual_delta_e76"] for row in gold]
                )
            ),
            "worst_gold_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in gold)
            ),
            "worst_stress_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in stress)
            ),
            "per_image": per_image,
        }
        gates = {
            "style_gain": summary["gold_median_style_delta_e76"]
            >= float(config["metrics"]["minimum_gold_median_style_delta_e76"]),
            "non_basic_gain": summary[
                "gold_median_non_basic_residual_delta_e76"
            ]
            >= float(
                config["metrics"][
                    "minimum_gold_median_non_basic_residual_delta_e76"
                ]
            ),
            "gold_clipping": summary["worst_gold_new_hard_clipping_fraction"]
            <= float(
                config["metrics"][
                    "maximum_worst_gold_new_hard_clipping_fraction"
                ]
            ),
        }
        summary["automatic_gates"] = gates
        summary["automatic_survivor"] = all(gates.values())
        summaries[candidate["candidate_id"]] = summary
    survivors = sorted(
        candidate_id
        for candidate_id, summary in summaries.items()
        if summary["automatic_survivor"]
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(validated["samples"]),
        "automatic_survivors": survivors,
        "automatic_decision": (
            "visual_gate_required" if survivors else "no_automatic_survivor"
        ),
        "candidates": summaries,
        "claim_ceiling": config["claim_ceiling"],
    }


def build_blind_sheets(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
    survivors: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    if not survivors:
        return {"rounds": [], "mapping": None, "reason": "empty survivors"}
    validated = validate_contract(root, config)
    manifest = _load_json(manifest_path)
    candidate_paths = {
        (str(row["candidate_id"]), str(row["sample_id"])): manifest_path.parent
        / str(row["output"])
        for row in manifest["records"]
    }
    anchor_manifest_path = root / str(config["parent_anchor"]["manifest"])
    density_manifest_path = root / str(config["parent_density"]["manifest"])
    anchor_paths = {
        (str(row["candidate_id"]), str(row["sample_id"])): root
        / str(row["output"])
        for row in validated["anchor_manifest"]["records"]
        if row["candidate_id"] == config["parent_anchor"]["candidate_id"]
    }
    density_paths = {
        (str(row["candidate_id"]), str(row["sample_id"])): density_manifest_path.parent
        / str(row["output"])
        for row in validated["density_manifest"]["records"]
        if row["candidate_id"] == config["parent_density"]["candidate_id"]
    }
    del anchor_manifest_path
    all_paths = {**candidate_paths, **anchor_paths, **density_paths}
    columns = [*config["visual"]["comparators"], *survivors]
    gold_ids = [
        sample_id
        for sample_id, sample in validated["samples"].items()
        if sample["split"] == "gold"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings = {}
    round_paths = []
    for round_index in range(1, int(config["visual"]["blind_rounds"]) + 1):
        shuffled = list(columns)
        random.Random(20260728 + round_index).shuffle(shuffled)
        labels = {
            candidate: chr(ord("A") + index)
            for index, candidate in enumerate(shuffled)
        }
        mappings[f"round_{round_index}"] = {
            label: candidate for candidate, label in labels.items()
        }
        tile_width, tile_height, header = 260, 180, 24
        sheet = Image.new(
            "RGB",
            (tile_width * len(shuffled), (tile_height + header) * len(gold_ids)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, sample_id in enumerate(gold_ids):
            for column_index, candidate in enumerate(shuffled):
                path = all_paths[(candidate, sample_id)]
                with Image.open(path) as image:
                    tile = ImageOps.contain(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        (tile_width, tile_height),
                    )
                x = column_index * tile_width + (tile_width - tile.width) // 2
                y = row_index * (tile_height + header) + header
                sheet.paste(tile, (x, y))
                draw.text(
                    (column_index * tile_width + 4, y - header + 4),
                    f"{labels[candidate]} | {sample_id}",
                    fill="black",
                )
        path = output_dir / f"blind_round_{round_index}.png"
        sheet.save(path, format="PNG", compress_level=6)
        round_paths.append(path)
    mapping_path = output_dir / "private_mapping.json"
    mapping_path.write_text(
        json.dumps(mappings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rounds": round_paths,
        "mapping": mapping_path,
    }


__all__ = [
    "CompositionFrontierError",
    "build_blind_sheets",
    "build_operators",
    "candidate_bank",
    "compose_rgb",
    "evaluate_bank",
    "render_bank",
    "validate_contract",
]
