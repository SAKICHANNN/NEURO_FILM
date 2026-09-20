import ast
import base64
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    config_path = ROOT / "configs/fable_ektar_external_control_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["output"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_claim.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(config_sha256=sha(config_path), pid=os.getpid(), executable=sys.executable), handle)
    started = time.perf_counter()
    report = dict(status="INITIALIZING", scope=config["scope"], config_sha256=sha(config_path), rows=[], runtime={}, logs=[])
    try:
        if Path(sys.executable).resolve() != (ROOT / config["runtime_python"]).resolve() or platform.python_version() != config["python_version"]:
            raise RuntimeError("wrong isolated Python runtime")
        for key in ("configure_source", "original_config", "original_admission", "inputs", "review_boxes", "icc_source"):
            if sha(ROOT / config[key]["path"]) != config[key]["sha256"]:
                raise RuntimeError(f"pinned binding changed: {key}")
        external = (ROOT / config["external_root"]).resolve()
        tree = subprocess.check_output(["git", "-c", "safe.directory=" + config["external_git_safe_directory"], "-C", str(ROOT / config["external_git_objects"]), "ls-tree", "-r", "-z", config["external_revision"]])
        file_count = 0
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            metadata, path_bytes = entry.split(b"\t", 1)
            mode, kind, identity = metadata.split()
            if kind != b"blob":
                raise RuntimeError("unexpected external git entry")
            path = external / path_bytes.decode("utf-8")
            payload = path.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if actual != identity.decode():
                raise RuntimeError(f"external blob identity mismatch: {path}")
            file_count += 1
        sys.dont_write_bytecode = True
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[key] = str(config["threads"])
        os.environ["NUMBA_CACHE_DIR"] = str(out / "numba_cache")
        os.environ["MPLCONFIGDIR"] = str(out / "matplotlib")
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(external / "src"))
        import numpy as np
        from PIL import Image
        import spektrafilm
        from spektrafilm import init_params, digest_params, simulate
        icc_tree = ast.parse((ROOT / config["icc_source"]["path"]).read_text(encoding="utf-8"))
        constants = {node.targets[0].id: ast.literal_eval(node.value) for node in icc_tree.body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ("_SRGB_ICC_PROFILE_BASE64", "SRGB_ICC_PROFILE_SHA256")}
        icc = base64.b64decode(constants["_SRGB_ICC_PROFILE_BASE64"], validate=True)
        if hashlib.sha256(icc).hexdigest() != constants["SRGB_ICC_PROFILE_SHA256"]:
            raise RuntimeError("pinned ICC bytes mismatch")

        for name, module in list(sys.modules.items()):
            if name == "spektrafilm" or name.startswith("spektrafilm."):
                location = getattr(module, "__file__", None)
                if location and not Path(location).resolve().is_relative_to(external):
                    raise RuntimeError(f"mixed external package source: {name}: {location}")
        versions = {}
        for name in ["numpy", "scipy", "scikit-image", "colour-science", "Pillow", "numba", "OpenImageIO", "rawpy", "pyfftw"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "distribution metadata unavailable; no installation performed"
        versions["spektrafilm_source_project"] = tomllib.loads((external / "pyproject.toml").read_text(encoding="utf-8"))["project"].get("version", "dynamic")
        report["runtime"] = dict(pid=os.getpid(), python=sys.executable, python_version=platform.python_version(), external_module=str(Path(spektrafilm.__file__).resolve()), external_revision=config["external_revision"], verified_git_blob_count=file_count, tree_listing_sha256=hashlib.sha256(tree).hexdigest(), gpu_used=False, threads=config["threads"], versions=versions)
        configure_text = (ROOT / config["configure_source"]["path"]).read_text(encoding="utf-8")
        function = next(node for node in ast.parse(configure_text).body if isinstance(node, ast.FunctionDef) and node.name == "configure")
        namespace = dict(init_params=init_params, digest_params=digest_params)
        exec(compile(ast.Module(body=[function], type_ignores=[]), config["configure_source"]["path"], "exec"), namespace)

        def parameter_record(params):
            body = dataclasses.asdict(params)
            encoded = json.dumps(body, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else str(value), allow_nan=True)
            controls = {key: value for key, value in json.loads(encoded).items() if key not in ("film", "print")}
            return dict(complete_parameter_sha256=hashlib.sha256(encoded.encode()).hexdigest(), non_profile_controls=controls)

        params = namespace["configure"](config["profile"], config["policy"])
        identity = parameter_record(params)
        report["parameter_identity"] = identity
        report["provenance"] = dict(config=config, wrapper_sha256=sha(Path(__file__)), parameter_identity=identity, profile_arrays_copied=False)
        write_json(out / "adoption.json", report["provenance"])
        write_json(out / "runtime.json", report["runtime"])
        report["status"] = "SYNTHETIC_SMOKE"
        write_json(out / "report.json", report)
        print(json.dumps(dict(stage="runtime_verified", runtime=report["runtime"])), flush=True)
        np.random.seed(config["seed"])
        toy = np.repeat(np.linspace(0, 1, 8)[None, :, None], 8, axis=0)
        toy = np.repeat(toy, 3, axis=2)
        toy[1, 1] = (1, 0, 0)
        toy[2, 2] = (0, 1, 0)
        toy[3, 3] = (0, 0, 1)
        smoke_started = time.perf_counter()
        smoke = np.asarray(simulate(toy, params, digest_params_first=False), dtype=np.float64)
        report["smoke"] = dict(shape=list(smoke.shape), finite=bool(np.isfinite(smoke).all()), minimum=float(smoke.min()), maximum=float(smoke.max()), seconds=time.perf_counter() - smoke_started)
        write_json(out / "smoke.json", report["smoke"])
        if smoke.shape != toy.shape or not np.isfinite(smoke).all() or smoke.min() < 0 or smoke.max() > 1:
            raise RuntimeError("synthetic smoke failed shape/finiteness/range")
        manifest = json.loads((ROOT / config["inputs"]["path"]).read_text(encoding="utf-8"))
        boxes = json.loads((ROOT / config["review_boxes"]["path"]).read_text(encoding="utf-8"))["rows"]
        if len(manifest["sources"]) != 4 or len(boxes) != 9:
            raise RuntimeError("unexpected source/crop count")

        def save_arm(folder, name, values, selected):
            path = folder / f"{name}.png"
            float_path = folder / f"{name}.npy"
            np.save(float_path, values)
            codes = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(codes).save(path, icc_profile=icc)
            with Image.open(path) as check:
                if not np.array_equal(np.asarray(check), codes) or check.info.get("icc_profile") != icc:
                    raise RuntimeError("PNG roundtrip failure")
            crops = []
            for box in selected:
                x0, y0, x1, y1 = box["box_xyxy_exclusive"]
                crop_path = folder / f"{name}_{box['label']}.png"
                crop = codes[y0:y1, x0:x1]
                Image.fromarray(crop).save(crop_path, icc_profile=icc)
                with Image.open(crop_path) as check:
                    if not np.array_equal(np.asarray(check), crop):
                        raise RuntimeError("crop roundtrip failure")
                crops.append(dict(label=box["label"], path=crop_path.relative_to(ROOT).as_posix(), sha256=sha(crop_path), box_xyxy_exclusive=box["box_xyxy_exclusive"]))
            return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), float_path=float_path.relative_to(ROOT).as_posix(), float_sha256=sha(float_path), float_domain="encoded sRGB float64 returned by renderer before wrapper quantization clipping", crops=crops)

        report["status"] = "RENDERING"
        for index, source in enumerate(manifest["sources"]):
            row_started = time.perf_counter()
            path = ROOT / source["path"]
            if sha(path) != source["sha256"]:
                raise RuntimeError("source identity mismatch")
            with Image.open(path) as image:
                profile = image.info.get("icc_profile")
                if image.mode != "RGB" or list(image.size) != source["size_wh"] or image.getexif().get(274, 1) != 1:
                    raise RuntimeError("source mode/shape/orientation mismatch")
                untagged_allowed = profile is None and source["sha256"] == config["untagged_srgb_source_sha256"]
                if not untagged_allowed and (profile is None or len(profile) != len(icc) or profile[:24] != icc[:24] or profile[36:] != icc[36:]):
                    raise RuntimeError("source ICC is not pinned sRGB except timestamp")
                codes = np.array(image)
            selected = [box for box in boxes if box["source_path"] == source["path"]]
            for box in selected:
                x0, y0, x1, y1 = box["box_xyxy_exclusive"]
                if box["source_sha256"] != source["sha256"] or hashlib.sha256(codes[y0:y1, x0:x1].tobytes()).hexdigest() != box["source_rgb_bytes_sha256"]:
                    raise RuntimeError("source crop binding mismatch")
            rgb = codes.astype(np.float64) / 255
            params = namespace["configure"](config["profile"], config["policy"])
            if parameter_record(params)["complete_parameter_sha256"] != identity["complete_parameter_sha256"]:
                raise RuntimeError("render parameter identity changed")
            print(json.dumps(dict(stage="source_start", source_index=index, size_wh=source["size_wh"])), flush=True)
            np.random.seed(config["seed"])
            candidate = np.asarray(simulate(rgb, params, digest_params_first=False), dtype=np.float64)
            if candidate.shape != rgb.shape:
                raise RuntimeError("native output shape changed")
            finite = bool(np.isfinite(candidate).all())
            epsilon = config["new_hard_clipping_epsilon"]
            new_clip = ((candidate <= epsilon) & (rgb > epsilon)) | ((candidate >= 1 - epsilon) & (rgb < 1 - epsilon))
            row = dict(
                source_index=index, size_wh=source["size_wh"], source_path=source["path"], source_sha256=source["sha256"],
                metrics=dict(finite=finite, preclip_min=float(candidate.min()) if finite else None,
                             preclip_max=float(candidate.max()) if finite else None,
                             wrapper_clipping_fraction=float(np.mean((candidate < 0) | (candidate > 1))),
                             new_hard_clipping_fraction=float(np.mean(new_clip))),
                parameter_sha256=identity["complete_parameter_sha256"],
                input_color_policy="untagged assumed sRGB for this pinned source only" if untagged_allowed else "pinned sRGB ICC except creation timestamp; renderer performs sole decode",
            )
            report["rows"].append(row)
            if not finite:
                np.save(out / f"source_{index:02d}_invalid.npy", candidate)
                raise RuntimeError("renderer returned nonfinite values")
            row["metrics"].update(mean_absolute_rgb_change=float(np.mean(np.abs(candidate - rgb))), rms_rgb_change=float(np.sqrt(np.mean((candidate - rgb) ** 2))), maximum_absolute_rgb_change=float(np.max(np.abs(candidate - rgb))), changed_code_fraction=float(np.mean(np.any(np.rint(np.clip(candidate, 0, 1) * 255).astype(np.uint8) != codes, axis=-1))))
            folder = out / f"source_{index:02d}"
            folder.mkdir()
            row["original"] = save_arm(folder, "original", rgb, selected)
            row["candidate"] = save_arm(folder, "candidate", candidate, selected)
            row["metrics"]["seconds"] = time.perf_counter() - row_started
            write_json(out / "report.json", report)
            print(json.dumps(dict(stage="source_complete", source_index=index, metrics=row["metrics"])), flush=True)
            if candidate.min() < 0 or candidate.max() > 1:
                raise RuntimeError("renderer range failure; preserved floats and clipped display, no rescue")
        report["status"] = "COMPLETE_AWAITING_VISUAL_REVIEW"
        report["logs"].append("Four native CPU forwards plus one synthetic smoke. No resampling, fitting, GPU, noise or parameter rescue. Preclip refers to returned renderer floats; internal handling unchanged.")
    except Exception as exc:
        report["status"] = "FAILED_NO_RETRY"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_json(out / "report.json", report)
    print(json.dumps(dict(status=report["status"], report=str(out / "report.json"), seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
