import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color_engine.srgb_transfer import encoded_srgb_to_linear
from src.color_match.research.fable_reference_response import (
    bernstein, color_data_loss, conditional_targets, encode_preview, encoded_to_oklab,
    fit_color, fit_tone, render_image, render_lab, soft_gamut,
)
from src.color_match.srgb_icc_profile import srgb_icc_profile_v1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_png(path, codes):
    profile = srgb_icc_profile_v1()
    Image.fromarray(codes).save(path, icc_profile=profile)
    with Image.open(path) as image:
        if not np.array_equal(np.asarray(image), codes) or image.info.get("icc_profile") != profile:
            raise ValueError("PNG pixel/profile verification failed")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def input_codes(entry):
    path = ROOT/entry["path"]
    if digest(path) != entry["sha256"]:
        raise ValueError("input hash mismatch")
    with Image.open(path) as image:
        if list(image.size) != entry["size_wh"] or image.mode != "RGB" or image.getexif().get(274, 1) != 1:
            raise ValueError("input dimensions/mode/orientation changed")
        profile = image.info.get("icc_profile")
        if profile is not None and not profile_colorimetry_matches(profile):
            raise ValueError("input profile is not the verified common sRGB profile")
        codes = np.asarray(image).copy()
    return codes, {"orientation": 1, "color_policy": "sRGB ICC identical to pinned profile except allowed creation timestamp bytes24:36; no second transform" if profile else "untagged assumed sRGB; colorimetry not established", "icc_sha256": hashlib.sha256(profile).hexdigest() if profile else None}


def profile_colorimetry_matches(profile):
    pinned = srgb_icc_profile_v1()
    return len(profile) == len(pinned) and profile[:24] == pinned[:24] and profile[36:] == pinned[36:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--continue-prefit", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.run, args.continue_prefit)) != 1:
        raise ValueError("choose exactly one of --prepare or --run")
    config_path = ROOT/"configs/fable_real_reference_conditional_v1.json"
    config = json.loads(config_path.read_text())
    manifest_path = ROOT/config["input_manifest"]
    boxes_path = ROOT/config["review_boxes"]
    if digest(manifest_path) != config["input_manifest_sha256"] or digest(boxes_path) != config["review_boxes_sha256"]:
        raise ValueError("frozen input metadata changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    if boxes["input_manifest_sha256"] != digest(manifest_path) or len(manifest["sources"]) != 4 or len(boxes["rows"]) != 9:
        raise ValueError("development frame changed")
    output = ROOT/config["output"]
    code_files = [Path(__file__), ROOT/"src/color_match/research/fable_reference_response.py", ROOT/"src/color_engine/oklch_local_minde.py", ROOT/"src/preprocess/color_management.py", ROOT/"src/color_engine/srgb_transfer.py", ROOT/"src/color_match/srgb_icc_profile.py", ROOT/"tests/test_fable_reference_response.py"]
    bindings = {str(path.relative_to(ROOT)): digest(path) for path in code_files}
    if args.prepare:
        output.mkdir(exist_ok=False)
        adoption = {
            "status": "PROSPECTIVE_ADOPTION_BEFORE_REAL_FITS", "config_sha256": digest(config_path),
            "science_result_sha256": digest(ROOT/config["science_result"]), "source_bindings": bindings,
            "scope": config["claim"], "algorithm": "tone Bernstein9 plus conditional color78; nested color4 shares targets and final gamut; optics and texture off",
            "clarifications": [
                "Use existing Rec2020/Oklab APIs through explicit sRGB/Rec2020 matrices; no duplicate conversion constants.",
                "Known source PNG ICC matches pinned sRGB exactly; no second ICC transform. Reference is untagged assumed sRGB. Orientation1 verified; otherwise reject before fit.",
                "Independent PCG64(seed20260920) generator per input, choice without replacement, retain draw order and indices. Reused reference sample is fixed across all four sources.",
                "Weighted CDF groups exact chroma ties, uses weighted midranks, and inverse quantiles linearly interpolate grouped midrank knots with endpoint extension.",
                "Absent chromatic support fixes gain1/rotation0; absent neutral support fixes tint0. Unsupported controls otherwise extend via declared smoothness, with no fabricated target.",
                "Internal SLSQP variable scaling(1,.1,.01 for gain/rotation/tint) preserves original QP objective/bounds; max200, ftol1e-8, no restart.",
                "Inverse Oklab ray RGB is cubic; derive coefficients by invoking existing inverse at C0,+1,-1,+2, split intervals at each channel derivative critical point, then24bisections in the first exiting interval. Explicit upper bracket expansion; fail if unavailable.",
                "Soft gamut knee0.95 uses fixed L/hue, measured <=2e-6 final roundoff only. Identical for simple/full/tone_only. Old baseline source-segment gamut remains frozen as a separately invalid negative control.",
                "Statistics/optimization use float64, source Oklab adapter accepts float32 Rec2020, renderer evaluates existing float64 color APIs in row tiles and stores linear float32. This intentional precision extension avoids new float32 conversion formulas and must pass tile/roundtrip checks.",
                "No human preference, reference-semantic correctness, or stock calibration inferred from numeric tests. Root visual review controls development verdict; one bounded run, no rescue."
            ],
            "engineering_checks": "Before --run: five focused tests covering conversion, tile/full, gray/hue ramps, first gamut exit across extreme-lightness rays, weighted ties, analytic-vs-finite-difference solver gradients, tone/color constraints and nested objective. Probe images must also be viewed.",
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__},
        }
        write_json(output/"adoption.json", adoption)
        alpha = np.linspace(0, 1, 9)
        coefficients = np.r_[np.ones(36), np.zeros(42)]
        levels = np.linspace(0, 1, 1024)
        gray = np.stack([levels, levels, levels], axis=-1)[None, ...].repeat(64, axis=0)
        rendered, _ = render_image(gray, alpha, coefficients, config)
        save_png(output/"gray_ramp.png", encode_preview(rendered))
        hue = np.linspace(-np.pi, np.pi, 1024)
        chroma = np.linspace(0, .4, 128)
        lab = np.stack([np.full((128,1024), .6), chroma[:,None]*np.cos(hue)[None,:], chroma[:,None]*np.sin(hue)[None,:]], axis=-1)
        linear, _ = soft_gamut(lab, config)
        save_png(output/"hue_chroma_ramp.png", encode_preview(linear))
        print(str(output/"adoption.json"))
        return
    adoption = json.loads((output/"adoption.json").read_text(encoding="utf-8"))
    if adoption["source_bindings"] != bindings or adoption["config_sha256"] != digest(config_path) or adoption["science_result_sha256"] != digest(ROOT/config["science_result"]):
        raise ValueError("prospective code/config/science bindings changed")
    claim_name = "run_claim.json"
    if args.continue_prefit:
        old = json.loads((output/"prefit_admission_report.json").read_text(encoding="utf-8"))
        if len(old["rows"]) != 4 or any(row.get("error") != "input profile is not the verified common sRGB profile" or row["arms"] for row in old["rows"]):
            raise ValueError("only the preserved zero-solver ICC admission may continue")
        if list(output.glob("source_*/coefficients.json")) or list(output.glob("source_*/targets.npz")):
            raise ValueError("cannot continue a run that already executed a solver")
        claim_name = "prefit_continuation_claim.json"
    with (output/claim_name).open("x", encoding="utf-8") as handle:
        json.dump({"status": "CLAIMED_ONCE", "adoption_sha256": digest(output/"adoption.json")}, handle)
    started = time.perf_counter()
    reference_codes, reference_policy = input_codes(manifest["reference"])
    reference = encoded_to_oklab(reference_codes/255.0).reshape(-1,3)
    reference_indices = np.random.default_rng(config["seed"]).choice(len(reference), min(len(reference),config["sample_limit"]), replace=False)
    np.save(output/"reference_sampling_indices.npy", reference_indices)
    reference_samples = reference[reference_indices]
    reference_artifact = save_png(output/"reference.png", reference_codes)
    report = {"status": "RUNNING", "claim": config["claim"], "config_sha256": digest(config_path), "adoption_path": str((output/"adoption.json").relative_to(ROOT)), "adoption_sha256": digest(output/"adoption.json"), "reference": {**reference_artifact, "source_path": manifest["reference"]["path"], "source_sha256": manifest["reference"]["sha256"], "color_policy": reference_policy}, "rows": []}
    for source_index, entry in enumerate(manifest["sources"]):
        if time.perf_counter()-started > config["compute_budget_seconds"]:
            report["status"] = "COMPUTE_BUDGET_EXCEEDED_NO_RESTART"
            write_json(output/"report.json", report)
            return
        source_start = time.perf_counter()
        folder = output/f"source_{source_index:02d}"
        folder.mkdir(exist_ok=args.continue_prefit)
        row = {"source_index": source_index, "source_path": entry["path"], "source_sha256": entry["sha256"], "size_wh": entry["size_wh"], "fit_status": "STARTED", "arms": {}, "coefficients_path": str((folder/"coefficients.json").relative_to(ROOT)), "metrics_path": str((folder/"metrics.json").relative_to(ROOT))}
        report["rows"].append(row)
        try:
            codes, row["color_policy"] = input_codes(entry)
            source_lab = encoded_to_oklab(codes/255.0)
            flat = source_lab.reshape(-1,3)
            indices = np.random.default_rng(config["seed"]).choice(len(flat), min(len(flat),config["sample_limit"]), replace=False)
            np.save(folder/"sampling_indices.npy", indices)
            samples = flat[indices]
            alpha, tone_receipt = fit_tone(samples, reference_samples, config)
            coefficients = {"alpha": alpha.tolist(), "tone_solver": tone_receipt}
            write_json(folder/"coefficients.json", coefficients)
            if not tone_receipt["valid"]:
                raise ValueError("tone solver invalid; retained, not restarted")
            targets = conditional_targets(samples, reference_samples, alpha, config)
            arrays = {"source_lab": samples, "tone_lightness": targets["tone_lightness"], "chromatic_weight": targets["weight"], "target_sum": targets["target_sum"], "target_squared_constant": np.asarray(targets["constant"])}
            for band in targets["neutral"]:
                arrays[f"neutral_{band['band']}_weights"] = band["weights"]
                arrays[f"neutral_{band['band']}_target"] = band["target"]
            np.savez(folder/"targets.npz", **arrays)
            simple, simple_receipt = fit_color(targets, config, simple=True)
            coefficients.update({"simple": simple.tolist(), "simple_solver": simple_receipt})
            write_json(folder/"coefficients.json", coefficients)
            if not simple_receipt["valid"]:
                raise ValueError("simple solver invalid; retained, not restarted")
            full, full_receipt = fit_color(targets, config, simple=False, initial=simple)
            coefficients.update({"full": full.tolist(), "full_solver": full_receipt})
            write_json(folder/"coefficients.json", coefficients)
            if not full_receipt["valid"]:
                raise ValueError("full solver invalid; retained, not restarted")
            if full_receipt["objective"] > simple_receipt["objective"]+1e-6:
                raise ValueError("full objective worse than feasible nested initialization")
            row["fit_status"] = "VALID_NO_CHROMATIC_FIT" if targets["chromatic_cells"] == 0 else "VALID_RENDERED_VISUAL_PENDING"
            metrics = {"support": targets["support"], "chromatic_cells": targets["chromatic_cells"], "source_supported_pixel_fraction": targets["source_supported_pixel_fraction"], "neutral_bands": [band["band"] for band in targets["neutral"]], "arms": {}}
            parameters = {"simple": simple, "full": full, "tone_only": np.r_[np.ones(36),np.zeros(42)]}
            for arm in config["arms"]:
                if arm == "original":
                    linear = encoded_srgb_to_linear(codes/255.0).astype(np.float32)
                    preview = codes
                    arm_metrics = {"change_median": 0.0, "change_p90": 0.0, "change_fraction_gt_002": 0.0}
                else:
                    linear, gamut = render_image(codes/255.0, alpha, parameters[arm], config)
                    preview = encode_preview(linear)
                    rendered_lab = encoded_to_oklab(response_encoded(linear)).reshape(-1, 3)
                    difference = np.linalg.norm(rendered_lab-flat, axis=1)
                    pregamut = render_lab(samples, alpha, parameters[arm], config)
                    post_samples = rendered_lab[indices]
                    arm_metrics = {"gamut": gamut, "change_median": float(np.median(difference)), "change_p90": float(np.quantile(difference,.9)), "change_fraction_gt_002": float(np.mean(difference>.02)), "target_loss_before_gamut": color_data_loss(pregamut[:,1:],targets,config), "target_loss_after_gamut": color_data_loss(post_samples[:,1:],targets,config)}
                if linear.shape != codes.shape or not np.isfinite(linear).all() or linear.min()<0 or linear.max()>1:
                    raise ValueError("invalid native output")
                linear_path = folder/f"{arm}_linear.npy"
                np.save(linear_path, linear)
                artifact = save_png(folder/f"{arm}.png", preview)
                artifact.update({"linear_path": str(linear_path.relative_to(ROOT)), "linear_sha256": digest(linear_path), "crops": []})
                for box in boxes["rows"]:
                    if box["source_path"] != entry["path"]:
                        continue
                    x0,y0,x1,y1 = box["box_xyxy_exclusive"]
                    original_crop = codes[y0:y1,x0:x1]
                    if hashlib.sha256(original_crop.tobytes()).hexdigest() != box["source_rgb_bytes_sha256"]:
                        raise ValueError("frozen review crop changed")
                    crop = save_png(folder/f"{arm}_{box['label']}.png", preview[y0:y1,x0:x1])
                    crop.update({"label": box["label"], "box_xyxy_exclusive": box["box_xyxy_exclusive"]})
                    artifact["crops"].append(crop)
                row["arms"][arm] = artifact
                metrics["arms"][arm] = arm_metrics
            row["wall_seconds"] = time.perf_counter()-source_start
            write_json(folder/"metrics.json", metrics)
            row["metrics_sha256"] = digest(folder/"metrics.json")
            row["coefficients_sha256"] = digest(folder/"coefficients.json")
        except Exception as error:
            row["fit_status"] = "INVALID_RETAINED_NO_RESTART"
            row["error"] = str(error)
            write_json(folder/"failure.json", {"error": str(error), "wall_seconds": time.perf_counter()-source_start})
        write_json(output/"report.json", report)
        print(json.dumps({"source_index":source_index,"status":row["fit_status"],"seconds":time.perf_counter()-source_start}),flush=True)
    report["wall_seconds"] = time.perf_counter()-started
    report["status"] = "COMPLETED_VISUAL_REVIEW_PENDING_UNPROMOTED" if all(row["fit_status"].startswith("VALID") for row in report["rows"]) else "COMPLETED_WITH_INVALID_OUTCOMES_NO_RESTART"
    report["artifact_counts"] = {"source_full_pngs":sum(len(row["arms"]) for row in report["rows"]),"crop_pngs":sum(len(arm["crops"]) for row in report["rows"] for arm in row["arms"].values())}
    write_json(output/"report.json",report)


def response_encoded(linear):
    from src.color_engine.srgb_transfer import linear_srgb_to_encoded
    return linear_srgb_to_encoded(linear)


if __name__ == "__main__":
    main()
