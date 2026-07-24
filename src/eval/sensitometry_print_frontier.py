"""Real-image B0 frontier for the fixed U2.2B sensitometry-print operator."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.eval.density_witness_frontier import encoded_srgb_to_linear, linear_srgb_to_encoded
from src.eval.global_frontier import load_frozen_samples, new_hard_clipping_fraction, sha256_file
from src.eval.sensitometry_print_composition import build_composition
from src.real_film.gold_matrix_transplant import sample_rgb_image, style_and_basic_residual
from src.roll2film.sensitometry_gauge import NeutralAxisGaugeOperator


class SensitometryFrontierError(ValueError):
    """Raised when frozen I0 inputs or rendered evidence drift."""


def candidate_bank(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for strength in config["strengths"]:
        value = float(strength)
        rows.append({
            "candidate_id": str(config["candidate_id_format"]).format(strength_percent=int(round(value * 100))),
            "strength": value,
        })
    if len(rows) != int(config["candidate_count"]) or len({row["candidate_id"] for row in rows}) != len(rows):
        raise SensitometryFrontierError("candidate bank does not match the frozen contract")
    return rows


def _load_hashed(root: Path, config: Mapping[str, Any], path_key: str, hash_key: str) -> dict[str, Any]:
    path = root / str(config[path_key])
    if sha256_file(path) != str(config[hash_key]):
        raise SensitometryFrontierError(f"{path_key} hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    parent = _load_hashed(root, config, "sensitometry_config", "sensitometry_config_sha256")
    composition = _load_hashed(root, config, "composition_config", "composition_config_sha256")
    print_config = _load_hashed(root, config, "print_config", "print_config_sha256")
    inherited = _load_hashed(root, config, "inherited_frontier_config", "inherited_frontier_config_sha256")
    for key, value in config["metrics"].items():
        if inherited["metrics"].get(key) != value:
            raise SensitometryFrontierError(f"inherited metric drift: {key}")
    manifests = {}
    for prefix in ("global", "density"):
        path_key = f"{prefix}_comparator_manifest"
        hash_key = f"{prefix}_comparator_manifest_sha256"
        manifests[prefix] = _load_hashed(root, config, path_key, hash_key)
    operator = build_composition(composition, parent, print_config)
    if "neutral_gauge_config" in config:
        gauge = _load_hashed(
            root,
            config,
            "neutral_gauge_config",
            "neutral_gauge_config_sha256",
        )
        if gauge["base_composition_config"] != config["composition_config"] or gauge["sensitometry_config"] != config["sensitometry_config"] or gauge["print_config"] != config["print_config"]:
            raise SensitometryFrontierError("neutral gauge parent config lineage mismatch")
        operator = NeutralAxisGaugeOperator.from_base(operator, int(gauge["gauge_knots"]))
    return {
        "operator": operator,
        "samples": load_frozen_samples(root, config),
        "candidates": candidate_bank(config),
        "comparators": manifests,
    }


def render_bank(*, root: Path, config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    validated = validate_contract(root, config)
    source_arrays = {}
    for sample_id, sample in validated["samples"].items():
        source_path = root / str(sample["source_path"])
        with Image.open(source_path) as image:
            source_arrays[sample_id] = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float64) / 255.0
    records = []
    for candidate in validated["candidates"]:
        candidate_dir = output_dir / candidate["candidate_id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, sample in validated["samples"].items():
            source_encoded = source_arrays[sample_id]
            source_linear = encoded_srgb_to_linear(source_encoded)
            transformed = validated["operator"].apply(source_linear)
            strength = float(candidate["strength"])
            output_linear = source_linear + strength * (transformed - source_linear)
            output_encoded = linear_srgb_to_encoded(output_linear)
            path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(np.rint(output_encoded * 255.0).astype(np.uint8), mode="RGB").save(path, format="PNG", compress_level=6)
            records.append({
                **candidate,
                "sample_id": sample_id,
                "source_sha256": sample["source_sha256"],
                "output": f"{candidate['candidate_id']}/{sample_id}.png",
                "output_sha256": sha256_file(path),
            })
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "frozen_set_sha256": config["frozen_set_sha256"],
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(validated["samples"]),
        "neutral_gauge_config_sha256": config.get("neutral_gauge_config_sha256"),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_bytes(encoded)
    return {"manifest": manifest, "manifest_path": path, "manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def evaluate_bank(*, root: Path, config: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    validated = validate_contract(root, config)
    samples = validated["samples"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {(row["candidate_id"], sample_id) for row in validated["candidates"] for sample_id in samples}
    records = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected or key in records:
            raise SensitometryFrontierError(f"unexpected or duplicate manifest record: {key}")
        records[key] = row
    if records.keys() != expected:
        raise SensitometryFrontierError("incomplete render manifest")
    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries = {}
    for candidate in validated["candidates"]:
        per_image = []
        for sample_id, sample in samples.items():
            record = records[(candidate["candidate_id"], sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            if sha256_file(output_path) != record["output_sha256"]:
                raise SensitometryFrontierError("render output hash mismatch")
            with Image.open(output_path) as output_image, Image.open(root / str(sample["source_path"])) as source_image:
                output = ImageOps.exif_transpose(output_image)
                source = ImageOps.exif_transpose(source_image)
                if output.mode != "RGB" or output.size != source.size:
                    raise SensitometryFrontierError("render decode or dimension mismatch")
            source_pixels = sample_rgb_image(root / str(sample["source_path"]), budget)
            output_pixels = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(source_pixels, output_pixels)
            per_image.append({
                "sample_id": sample_id,
                "split": sample["split"],
                "median_style_delta_e76": style,
                "median_non_basic_residual_delta_e76": residual,
                "new_hard_clipping_fraction": new_hard_clipping_fraction(source_pixels, output_pixels, epsilon),
                "output_sha256": record["output_sha256"],
            })
        gold = [row for row in per_image if row["split"] == "gold"]
        stress = [row for row in per_image if row["split"] == "stress"]
        summary = {
            "strength": candidate["strength"],
            "gold_median_style_delta_e76": float(np.median([row["median_style_delta_e76"] for row in gold])),
            "gold_median_non_basic_residual_delta_e76": float(np.median([row["median_non_basic_residual_delta_e76"] for row in gold])),
            "worst_gold_new_hard_clipping_fraction": float(max(row["new_hard_clipping_fraction"] for row in gold)),
            "worst_stress_new_hard_clipping_fraction": float(max(row["new_hard_clipping_fraction"] for row in stress)),
            "per_image": per_image,
        }
        summary["automatic_gates"] = {
            "style_floor": summary["gold_median_style_delta_e76"] >= float(config["metrics"]["minimum_gold_median_style_delta_e76"]),
            "non_basic_floor": summary["gold_median_non_basic_residual_delta_e76"] >= float(config["metrics"]["minimum_gold_median_non_basic_residual_delta_e76"]),
            "gold_clipping": summary["worst_gold_new_hard_clipping_fraction"] <= float(config["metrics"]["maximum_worst_gold_new_hard_clipping_fraction"]),
        }
        summary["automatic_survivor"] = all(summary["automatic_gates"].values())
        summaries[candidate["candidate_id"]] = summary
    shortlist = shortlist_candidates(summaries, config)
    return {
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(samples),
        "gold_sample_count": sum(row["split"] == "gold" for row in samples.values()),
        "stress_sample_count": sum(row["split"] == "stress" for row in samples.values()),
        "all_source_and_output_hashes_verified": True,
        "all_outputs_rgb_and_dimension_matched": True,
        "automatic_survivors": sorted(key for key, value in summaries.items() if value["automatic_survivor"]),
        "shortlist": shortlist,
        "automatic_decision": "visual_gate_required" if shortlist else "no_automatic_survivor",
        "candidates": summaries,
    }


def shortlist_candidates(summaries: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> list[str]:
    rows = [(candidate_id, row) for candidate_id, row in summaries.items() if row["automatic_survivor"]]
    rows.sort(key=lambda item: (-float(item[1]["gold_median_non_basic_residual_delta_e76"]), -float(item[1]["gold_median_style_delta_e76"]), float(item[1]["strength"]), item[0]))
    return [candidate_id for candidate_id, _ in rows[: int(config["shortlist"]["maximum_strengths"])]]


def build_blind_sheets(*, root: Path, config: Mapping[str, Any], manifest_path: Path, shortlist: Sequence[str], output_dir: Path) -> dict[str, Any]:
    validated = validate_contract(root, config)
    candidate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {(str(row["candidate_id"]), str(row["sample_id"])): manifest_path.parent / str(row["output"]) for row in candidate_manifest["records"]}
    for prefix, manifest in validated["comparators"].items():
        comparator_manifest_path = root / str(config[f"{prefix}_comparator_manifest"])
        for row in manifest["records"]:
            candidate_id = str(row["candidate_id"])
            if candidate_id in config["comparators"]:
                paths[(candidate_id, str(row["sample_id"]))] = _resolve_manifest_output(
                    root, comparator_manifest_path, str(row["output"])
                )
    columns = [*config["comparators"], *shortlist]
    gold_ids = [sample_id for sample_id, row in validated["samples"].items() if row["split"] == "gold"]
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings = {}
    round_paths = []
    for round_index in range(1, int(config["visual"]["blind_rounds"]) + 1):
        shuffled = list(columns)
        random.Random(int(config["render_seed"]) + round_index).shuffle(shuffled)
        labels = {candidate: chr(ord("A") + index) for index, candidate in enumerate(shuffled)}
        mappings[f"round_{round_index}"] = {label: candidate for candidate, label in labels.items()}
        canvas = _contact_sheet(root, validated["samples"], gold_ids, shuffled, labels, paths)
        path = output_dir / f"blind_round_{round_index}.png"
        canvas.save(path, format="PNG", compress_level=6)
        round_paths.append(str(path.relative_to(root)).replace("\\", "/"))
    mapping_path = output_dir / "private_mapping.json"
    mapping_path.write_text(json.dumps(mappings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rounds": round_paths, "mapping": str(mapping_path.relative_to(root)).replace("\\", "/")}


def _resolve_manifest_output(root: Path, manifest_path: Path, output: str) -> Path:
    """Resolve the two explicit output conventions used by frozen comparators."""

    relative = Path(output)
    if relative.is_absolute():
        raise SensitometryFrontierError("comparator output must be relative")
    if relative.parts and relative.parts[0] == "outputs":
        return root / relative
    return manifest_path.parent / relative


def _contact_sheet(root: Path, samples: Mapping[str, Mapping[str, Any]], gold_ids: Sequence[str], columns: Sequence[str], labels: Mapping[str, str], paths: Mapping[tuple[str, str], Path]) -> Image.Image:
    tile_width, tile_height, header = 300, 210, 26
    canvas = Image.new("RGB", ((len(columns) + 1) * tile_width, len(gold_ids) * (tile_height + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(gold_ids):
        y = row_index * (tile_height + header)
        row_paths = [root / str(samples[sample_id]["source_path"]), *(paths[(candidate, sample_id)] for candidate in columns)]
        names = [f"INPUT {sample_id}", *(labels[candidate] for candidate in columns)]
        for column_index, (path, name) in enumerate(zip(row_paths, names)):
            with Image.open(path) as image:
                tile = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (tile_width, tile_height), method=Image.Resampling.LANCZOS)
            x = column_index * tile_width
            canvas.paste(tile, (x, y + header))
            draw.text((x + 5, y + 5), name, fill="black")
    return canvas


__all__ = ["SensitometryFrontierError", "build_blind_sheets", "candidate_bank", "evaluate_bank", "render_bank", "shortlist_candidates", "validate_contract"]
