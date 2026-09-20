import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.color_engine.srgb_transfer import encoded_srgb_to_linear, linear_srgb_to_encoded
from src.color_match.srgb_icc_profile import srgb_icc_profile_v1
from src.film_physics.fable_status_m_positive import StatusMPositive, redistribute_highlights
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicCurve, ManufacturerCharacteristicPrior


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def numerical_preflight(model, config, output):
    a, b, eps = (model.references[k] for k in ("a", "b", "epsilon"))
    values = list(np.linspace(0, 1, config["numerics"]["ramp_samples"])) + [config["parameters"]["gray"]]
    for curve in model.prior.curves:
        for q in curve.log_exposure_knots:
            if a < q < b:
                x = (2 ** (8 * (q - a) / (b - a) - 8) - eps) / (1 - eps)
                values.extend([np.nextafter(x, 0), x, np.nextafter(x, 1)])
    x = np.unique(values)
    neutral = np.repeat(x[:, None], 3, axis=1)
    y = model.apply(neutral)
    anchors = np.repeat(np.array([0., config["parameters"]["gray"], 1.])[:, None], 3, axis=1)
    delta = 1e-6
    slope = (model.apply(anchors[1:2] + delta) - model.apply(anchors[1:2] - delta)) / (2 * delta)
    probe = np.zeros((43, 51, 3))
    probe[0, 0] = (1, 0, 1)
    probe[21, 25] = 1
    probe[-5:, -7:, 1] = 1
    e = model.pseudo_exposure(probe)
    h = redistribute_highlights(e, eps, config["parameters"]["optics"])
    constants = []
    for value in [0., .5, .75, 1.]:
        field = model.pseudo_exposure(np.full((29, 37, 3), value))
        constants.append(float(np.max(np.abs(redistribute_highlights(field, eps, config["parameters"]["optics"]) - field))))
    metrics = dict(anchor_error=float(np.max(np.abs(model.apply(anchors) - anchors))), pivot_slope_error=float(np.max(np.abs(slope - config["parameters"]["contrast"]))), min_ramp_step=float(np.diff(y, axis=0).min()), float32_error=float(np.max(np.abs(model.apply(neutral.astype(np.float32)).astype(np.float32) - y))), constant_error=max(constants), mass_error=float(np.max(np.abs((h - e).sum(axis=(0, 1))))), highlight_bounds=[float(h.min()), float(h.max())])
    gates = dict(finite=bool(np.isfinite(y).all()), bounded=bool(y.min() >= 0 and y.max() <= 1), anchors=metrics["anchor_error"] <= config["numerics"]["anchor_tolerance"], slope=metrics["pivot_slope_error"] <= config["numerics"]["slope_tolerance"], monotonic=metrics["min_ramp_step"] >= -config["numerics"]["monotonic_tolerance"], float32=metrics["float32_error"] <= config["numerics"]["float32_tolerance"], constant=metrics["constant_error"] <= 1e-14, mass=metrics["mass_error"] <= config["numerics"]["mass_tolerance"], exposure_bounds=bool(h.min() >= eps and h.max() <= 1), replay=bool(np.array_equal(h, redistribute_highlights(e, eps, config["parameters"]["optics"]))))
    curves = tuple(ManufacturerCharacteristicCurve(c.layer, np.array([a, (a + b) / 2, b]), np.linspace(r["D0"], r["D1"], 3)) for c, r in zip(model.prior.curves, model.references["channels"]))
    alternate = ManufacturerCharacteristicPrior(curves, model.prior.source_evidence_id, dict(model.prior.measurement_context))
    ablation = StatusMPositive.build(alternate, config["parameters"])
    ya = ablation.apply(neutral)
    chromatic = np.stack([x, 1 - x, np.full_like(x, .18)], axis=-1)
    yc, yca = model.apply(chromatic), ablation.apply(chromatic)
    derivative_x = np.linspace(0, 1, config["numerics"]["ramp_samples"])
    derivative_y = model.apply(np.repeat(derivative_x[:, None], 3, axis=1))
    np.savez(output / "synthetic_ramps.npz", input=neutral, photometry=y, derivative_regular_input=derivative_x, derivative_regular_grid=np.gradient(derivative_y, derivative_x, axis=0), straight_characteristic=ya, chromatic_input=chromatic, chromatic=yc, chromatic_straight=yca)
    metrics["curve_ablation_neutral_max_abs"] = float(np.max(np.abs(y - ya)))
    metrics["curve_ablation_chromatic_max_abs"] = float(np.max(np.abs(yc - yca)))
    result = dict(passed=all(gates.values()), gates=gates, metrics=metrics, straight_curve_references=ablation.references, float32_scope="float32 input/output with float64 internal evaluation; not native float32 arithmetic", scope="synthetic attribution and numerical validity only; not appearance evidence")
    write_json(output / "preflight.json", result)
    if not result["passed"]:
        raise ValueError("numerical preflight failed")
    return result


def save_arm(folder, name, linear, boxes):
    if linear.dtype != np.float64 or not np.isfinite(linear).all() or linear.min() < 0 or linear.max() > 1:
        raise ValueError("invalid unclipped linear output")
    float_path = folder / f"{name}.npy"
    np.save(float_path, linear.astype(np.float32))
    codes = np.rint(linear_srgb_to_encoded(linear) * 255).astype(np.uint8)
    path = folder / f"{name}.png"
    icc = srgb_icc_profile_v1()
    Image.fromarray(codes).save(path, icc_profile=icc)
    with Image.open(path) as image:
        if not np.array_equal(np.asarray(image), codes) or image.info.get("icc_profile") != icc:
            raise ValueError("PNG roundtrip failed")
    stored = np.load(float_path)
    if stored.shape != linear.shape or not np.isfinite(stored).all() or not np.array_equal(stored, linear.astype(np.float32)):
        raise ValueError("float roundtrip failed")
    crops = []
    for box in boxes:
        x0, y0, x1, y1 = box["box_xyxy_exclusive"]
        crop_path = folder / f"{name}_{box['label']}.png"
        crop_codes = codes[y0:y1, x0:x1]
        Image.fromarray(crop_codes).save(crop_path, icc_profile=icc)
        with Image.open(crop_path) as image:
            if not np.array_equal(np.asarray(image), crop_codes):
                raise ValueError("crop roundtrip failed")
        crops.append(dict(label=box["label"], path=crop_path.relative_to(ROOT).as_posix(), sha256=sha(crop_path), box_xyxy_exclusive=box["box_xyxy_exclusive"]))
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), float_path=float_path.relative_to(ROOT).as_posix(), float_sha256=sha(float_path), float_domain="linear sRGB float32 HWC", crops=crops, min=float(linear.min()), max=float(linear.max()), clipping_fraction=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fable_status_m_positive_v1.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_claim.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(config_sha256=sha(config_path), mode="one batch, no retry"), handle)
    started = time.perf_counter()
    report = dict(type="fable_status_m_positive_interim", scope=config["scope"], algorithm="status_m_generic_positive_with_optional_highlight_redistribution_v1", status="RUNNING", config=dict(path=args.config, sha256=sha(config_path)), provenance={}, rows=[], logs=[])
    try:
        for key in ("prior", "inputs", "review_boxes", "science", "source_recheck"):
            pin = config[key]
            if sha(ROOT / pin["path"]) != pin["sha256"]:
                raise ValueError(f"binding changed: {key}")
            report["provenance"][key] = pin
        source_recheck = json.loads((ROOT / config["source_recheck"]["path"]).read_text(encoding="utf-8"))
        for pin in source_recheck["items"]:
            if sha(ROOT / pin["path"]) != pin["sha256"]:
                raise ValueError("manufacturer source identity changed")
        report["provenance"]["manufacturer_sources"] = source_recheck["items"]
        code_paths = [Path(__file__), ROOT / "src/film_physics/fable_status_m_positive.py", ROOT / "tests/test_fable_status_m_positive.py", ROOT / "src/film_physics/manufacturer_characteristic.py", ROOT / "src/color_engine/srgb_transfer.py", ROOT / "src/color_match/srgb_icc_profile.py"]
        report["provenance"]["code"] = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in code_paths]
        prior = ManufacturerCharacteristicPrior.from_dict(json.loads((ROOT / config["prior"]["path"]).read_text(encoding="utf-8"))["prior"])
        model = StatusMPositive.build(prior, config["parameters"])
        adoption = dict(config=config, config_sha256=sha(config_path), provenance=report["provenance"], derived=model.references, phase="FROZEN_BEFORE_PHOTOGRAPHIC_RENDER", claim=report["scope"])
        write_json(output / "adoption.json", adoption)
        report["numerical"] = numerical_preflight(model, config, output)
        manifest = json.loads((ROOT / config["inputs"]["path"]).read_text(encoding="utf-8"))
        boxes = json.loads((ROOT / config["review_boxes"]["path"]).read_text(encoding="utf-8"))["rows"]
        pinned_profile = srgb_icc_profile_v1()
        for index, source in enumerate(manifest["sources"]):
            row_started = time.perf_counter()
            path = ROOT / source["path"]
            if sha(path) != source["sha256"]:
                raise ValueError("source identity changed")
            with Image.open(path) as image:
                profile = image.info.get("icc_profile")
                if image.mode != "RGB" or list(image.size) != source["size_wh"] or image.getexif().get(274, 1) != 1:
                    raise ValueError("source shape/mode/orientation changed")
                if profile is None or len(profile) != len(pinned_profile) or profile[:24] != pinned_profile[:24] or profile[36:] != pinned_profile[36:]:
                    raise ValueError("source ICC colorimetry differs from pinned sRGB")
                codes = np.array(image)
            selected = [box for box in boxes if box["source_path"] == source["path"]]
            for box in selected:
                x0, y0, x1, y1 = box["box_xyxy_exclusive"]
                if box["source_sha256"] != source["sha256"] or hashlib.sha256(codes[y0:y1, x0:x1].tobytes()).hexdigest() != box["source_rgb_bytes_sha256"]:
                    raise ValueError("fixed source crop changed")
            linear = encoded_srgb_to_linear(codes.astype(np.float64) / 255)
            folder = output / f"source_{index:02d}"
            folder.mkdir()
            row = dict(source_index=index, source_path=source["path"], source_sha256=source["sha256"], size_wh=source["size_wh"], source_icc_sha256=hashlib.sha256(profile).hexdigest(), arms={})
            report["rows"].append(row)
            for arm in config["arms"]:
                rendered = linear if arm == "original" else model.apply(linear, optics=arm == "photometry_optics_proxy")
                row["arms"][arm] = save_arm(folder, arm, rendered, selected)
                row["arms"][arm]["mean_absolute_linear_change"] = float(np.mean(np.abs(rendered - linear)))
            row["duration_seconds"] = time.perf_counter() - row_started
            write_json(output / "report.json", report)
        counts = dict(fullframes=sum(len(r["arms"]) for r in report["rows"]), crops=sum(len(a["crops"]) for r in report["rows"] for a in r["arms"].values()))
        if counts != dict(fullframes=12, crops=27):
            raise ValueError("output count mismatch")
        report["counts"] = counts
        report["status"] = "COMPLETE_AWAITING_ACTUAL_VISUAL_REVIEW"
        report["logs"].append("One batch; zero optimization, retries, clipping, texture or display-layer halation; no appearance verdict.")
    except Exception as exc:
        report["status"] = "FAILED_NO_RETRY"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["duration_seconds"] = time.perf_counter() - started
        write_json(output / "report.json", report)
    print(json.dumps(dict(status=report["status"], report=str(output / "report.json"), seconds=report["duration_seconds"])))


if __name__ == "__main__":
    main()
