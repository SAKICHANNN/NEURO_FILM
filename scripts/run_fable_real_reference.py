import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color_engine.gamut import compress_source_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.srgb_transfer import encoded_srgb_to_linear, linear_srgb_to_encoded
from src.roll2film.baselines import fit_joint_basic_adjustment
from src.roll2film.operators import AffineColorOperator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_gamut(source: np.ndarray, target: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    if target.shape != source.shape or not np.isfinite(target).all():
        raise ValueError("invalid full-frame target")
    source_lab = linear_rgb_to_lab(source.astype(np.float32), working_space="linear_srgb")
    target_lab = linear_rgb_to_lab(target.astype(np.float32), working_space="linear_srgb")
    mapped_lab = compress_source_to_working_gamut(
        source_lab, target_lab, working_space="linear_srgb",
        iterations=config["gamut_iterations"], tolerance=config["roundoff_tolerance"],
    )
    mapped = lab_to_linear_rgb(mapped_lab, working_space="linear_srgb").astype(np.float64)
    if not np.isfinite(mapped).all():
        raise ValueError("nonfinite gamut output")
    clamped = np.clip(mapped, 0.0, 1.0)
    clamp_delta = np.abs(clamped - mapped)
    if float(clamp_delta.max()) > config["roundoff_tolerance"]:
        raise ValueError("gamut output requires more than numerical roundoff clipping")
    metrics = {"policy": config["gamut_policy"], "stages": {}}
    requested_lab_distance = np.linalg.norm(target_lab - source_lab, axis=-1)
    retained_lab_distance = np.linalg.norm(mapped_lab - source_lab, axis=-1)
    active = requested_lab_distance > 1e-6
    ratio = np.ones_like(requested_lab_distance)
    ratio[active] = retained_lab_distance[active] / requested_lab_distance[active]
    metrics["retained_lab_displacement_ratio"] = {
        "minimum": float(ratio.min()), "mean": float(ratio.mean()),
        "quantiles_05_50_95": np.quantile(ratio, [0.05, 0.5, 0.95]).tolist(),
        "fraction_below_0999": float((ratio < 0.999).mean()),
        "fraction_below_01": float((ratio < 0.1).mean()),
        "active_pixel_fraction": float(active.mean()),
    }
    for name, values in (("unconstrained", target), ("gamut_before_roundoff", mapped), ("final", clamped)):
        outside = (values < 0.0) | (values > 1.0)
        tolerance = config["roundoff_tolerance"]
        substantive = (values < -tolerance) | (values > 1.0 + tolerance)
        metrics["stages"][name] = {
            "minimum": float(values.min()), "maximum": float(values.max()),
            "out_of_gamut_channel_fraction": float(outside.mean()),
            "out_of_gamut_pixel_fraction": float(outside.any(axis=-1).mean()),
            "beyond_tolerance_pixel_fraction": float(substantive.any(axis=-1).mean()),
        }
    metrics.update({
        "roundoff_clamped_channel_fraction": float((clamp_delta > 0).mean()),
        "roundoff_clamped_pixel_fraction": float((clamp_delta > 0).any(axis=-1).mean()),
        "maximum_roundoff_clamp": float(clamp_delta.max()),
        "gamut_mapping_linear_rmse": float(np.sqrt(np.mean((clamped - target) ** 2))),
    })
    return clamped, metrics


def main():
    started = time.perf_counter()
    config_path = ROOT / "configs/fable_real_reference_baseline_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["input_manifest"]
    boxes_path = ROOT / config["review_boxes"]
    if digest(manifest_path) != config["input_manifest_sha256"] or digest(boxes_path) != config["review_boxes_sha256"]:
        raise ValueError("input manifest or review boxes changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    if boxes["input_manifest_sha256"] != digest(manifest_path):
        raise ValueError("review box manifest binding mismatch")
    if len(manifest["sources"]) != 4 or len(boxes["rows"]) != 9:
        raise ValueError("fixed development frame changed")
    images = {}
    for entry in [manifest["reference"], *manifest["sources"]]:
        path = ROOT / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("input image hash mismatch")
        with Image.open(path) as image:
            if list(image.size) != entry["size_wh"] or image.mode != "RGB":
                raise ValueError("input image shape or mode mismatch")
            images[entry["path"]] = np.asarray(image).copy()
    for box in boxes["rows"]:
        source = next(row for row in manifest["sources"] if row["path"] == box["source_path"])
        if source["sha256"] != box["source_sha256"]:
            raise ValueError("review source binding mismatch")
        x0, y0, x1, y1 = box["box_xyxy_exclusive"]
        width, height = source["size_wh"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("review box outside source")
        crop = images[source["path"]][y0:y1, x0:x1]
        if hashlib.sha256(crop.tobytes()).hexdigest() != box["source_rgb_bytes_sha256"]:
            raise ValueError("review crop pixels changed")
    reference = encoded_srgb_to_linear(images[manifest["reference"]["path"]] / 255.0)
    reference_pixels = reference.reshape(-1, 3)
    reference_indices = np.linspace(0, len(reference_pixels) - 1, config["samples"], dtype=np.int64)
    reference_samples = reference_pixels[reference_indices]
    output = ROOT / config["output"]
    output.mkdir(exist_ok=False)
    rows = []
    for source_index, entry in enumerate(manifest["sources"]):
        start = time.perf_counter()
        folder = output / f"source_{source_index:02d}"
        folder.mkdir()
        original = images[entry["path"]]
        linear = encoded_srgb_to_linear(original / 255.0)
        pixels = linear.reshape(-1, 3)
        indices = np.linspace(0, len(pixels) - 1, config["samples"], dtype=np.int64)
        fit_start = time.perf_counter()
        operator = fit_joint_basic_adjustment(pixels[indices], reference_samples)
        fit_seconds = time.perf_counter() - fit_start
        payload = operator.to_dict()
        (folder / "operator.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        target = operator.apply(linear)
        replay = AffineColorOperator.from_dict(payload).apply(pixels[indices])
        if not np.array_equal(replay, operator.apply(pixels[indices])):
            raise ValueError("operator serialization replay mismatch")
        final, gamut = final_gamut(linear, target, config)
        baseline = np.floor(linear_srgb_to_encoded(final) * 255.0 + 0.5).astype(np.uint8)
        roundtrip = np.floor(linear_srgb_to_encoded(linear) * 255.0 + 0.5).astype(np.uint8)
        if not np.array_equal(roundtrip, original):
            raise ValueError("source RGB8 roundtrip changed pixels")
        row = {
            "source_index": source_index, "source_path": entry["path"], "source_sha256": entry["sha256"],
            "reference_sha256": manifest["reference"]["sha256"], "size_wh": entry["size_wh"],
            "fit_seconds": fit_seconds, "gamut": gamut,
            "linear_rmse_vs_source": float(np.sqrt(np.mean((final - linear) ** 2))),
            "code_rmse_vs_source": float(np.sqrt(np.mean((baseline.astype(float) - original) ** 2))),
            "source_endpoint_code_channel_fraction": float(((original == 0) | (original == 255)).mean()),
            "baseline_endpoint_code_channel_fraction": float(((baseline == 0) | (baseline == 255)).mean()),
            "operator_sha256": digest(folder / "operator.json"), "artifacts": [],
        }
        for arm, codes in (("original", original), ("baseline", baseline)):
            path = folder / f"{arm}.png"
            Image.fromarray(codes).save(path)
            with Image.open(path) as saved:
                if not np.array_equal(np.asarray(saved), codes):
                    raise ValueError("saved PNG verification failed")
            row["artifacts"].append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "size_wh": entry["size_wh"]})
            for box in boxes["rows"]:
                if box["source_path"] != entry["path"]:
                    continue
                x0, y0, x1, y1 = box["box_xyxy_exclusive"]
                crop_path = folder / f"{arm}_{box['label']}.png"
                crop = codes[y0:y1, x0:x1]
                Image.fromarray(crop).save(crop_path)
                with Image.open(crop_path) as saved:
                    if not np.array_equal(np.asarray(saved), crop):
                        raise ValueError("saved crop verification failed")
                row["artifacts"].append({"path": str(crop_path.relative_to(ROOT)), "sha256": digest(crop_path), "box": box["box_xyxy_exclusive"], "size_wh": [x1-x0, y1-y0]})
        row["wall_seconds"] = time.perf_counter() - start
        (folder / "receipt.json").write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
        rows.append(row)
        print(json.dumps({"source_index": source_index, "seconds": row["wall_seconds"], "code_rmse": row["code_rmse_vs_source"]}), flush=True)
    if sum(len(row["artifacts"]) for row in rows) != 26:
        raise ValueError("expected eight full images and eighteen review crops")
    code_paths = [Path(__file__), ROOT / "src/roll2film/baselines.py", ROOT / "src/roll2film/operators.py", ROOT / "src/color_engine/srgb_transfer.py", ROOT / "src/color_engine/lab.py", ROOT / "src/color_engine/gamut.py", ROOT / "src/preprocess/color_management.py"]
    report = {
        "status": "BASELINE_RENDERED_VISUAL_REVIEW_PENDING_UNPROMOTED", "claim": config["claim"],
        "config": config, "config_sha256": digest(config_path), "rows": rows,
        "source_files": {str(path.relative_to(ROOT)): digest(path) for path in code_paths},
        "wall_seconds": time.perf_counter() - started,
        "fits": 4, "full_frame_pngs": 8, "review_crop_pngs": 18,
        "loss_correspondence": "unpaired source/reference summary vectors; no pixelwise pairing",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
