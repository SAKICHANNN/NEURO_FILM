import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import tifffile
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits

from src.eval import tst_reference_response as core

CONFIG = ROOT / "configs/tst_reference_check_v1.json"
STAGES = ("prepare", "predict", "targets", "score")


def save(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")
    pending.replace(path)


def category(stage):
    return "feature" if stage in ("prepare", "targets") else "learning"


def load_model(path):
    model = json.loads(path.read_text())
    for projection in model["projections"].values():
        for key in ("center", "components", "score_mean", "score_scale"):
            projection[key] = np.asarray(projection[key])
        for block in projection["standards"]:
            for key in ("mean", "scale", "active"):
                block[key] = np.asarray(block[key])
    for head in model["heads"].values():
        head["coefficients"] = np.asarray(head["coefficients"])
    return model


def predict_operators(model, source_stats, source_vgg, reference_stats, reference_vgg):
    zx = core.project(model["projections"]["source"], [source_stats[None], source_vgg[None]])[0]
    zb = core.project(model["projections"]["statistics"], [reference_stats[None]])[0]
    zc = core.project(model["projections"]["image"], [reference_stats[None], reference_vgg[None]])[0]
    vectors = {"A": np.r_[1., zx, np.zeros(6)], "B": np.r_[1., zx, zb], "C": np.r_[1., zx, zc]}
    return {a: (vectors[a] @ model["heads"][a]["coefficients"]).reshape(343, 3) for a in "ABC"}


def verify_crops(crops, decoded):
    assert crops["status"] == "SOURCE_ONLY_CROPS_FROZEN"
    assert [g["group"] for g in crops["groups"]] == list(range(24, 32))
    for group in crops["groups"]:
        source = decoded[f'{group["group"]:02d}_X']
        assert group["source_sha256"] == source["srgb16_sha256"]
        width, height = source["encoded_size"]
        assert set(group["regions"]) == {"face", "text", "dark", "bright"}
        for label, box in group["regions"].items():
            if box is None:
                assert group["absence_reasons"].get(label)
            else:
                assert len(box) == 4 and all(type(v) is int for v in box)
                x, y, w, h = box
                assert 0 <= x and 0 <= y and 0 < w <= 256 and 0 < h <= 256
                assert x+w <= width and y+h <= height


def preflight(stage):
    cfg = json.loads(CONFIG.read_text())
    for entry in cfg["pins"].values():
        assert core.digest(ROOT/entry["path"]) == entry["sha256"]
    assert core.digest(Path(__file__)) == cfg["entry_sha256"]
    assert [g["group_index"] for g in cfg["groups"]] == list(range(24, 32))
    assert all(g["split"] == "development_check" for g in cfg["groups"])
    frozen_groups = [json.loads(line) for line in (ROOT/cfg["pins"]["groups"]["path"]).read_text().splitlines()]
    assert cfg["groups"] == frozen_groups[24:]
    head = json.loads((ROOT/cfg["pins"]["head"]["path"]).read_text())
    lookup = {entry["path"]: entry for entry in head["files"]}
    assert all(entry == lookup[entry["path"]] for entry in cfg["files"])
    assert cfg["scoring"] == json.loads((ROOT/cfg["pins"]["fit_config"]["path"]).read_text())["scoring"]
    for kind, pin in (("feature", "feature_report"), ("learning", "fit_report")):
        original = json.loads((ROOT/cfg["pins"][pin]["path"]).read_text())
        assert original["status"] == "COMPLETE_NOT_PROMOTED"
        assert cfg["initial_remaining_cpu_seconds"][kind] == original["cumulative_budget_ledger"]["remaining_cpu_seconds"]
    required = {r[k] for g in cfg["groups"] for r in g["triplets"] for k in ("content", "reference", "gt")}
    assert required == {f["path"] for f in cfg["files"]} and len(required) == len(cfg["files"]) == 40
    assert sum(f["bytes"] for f in cfg["files"]) == cfg["total_bytes"]
    out = ROOT/cfg["output"]
    remaining = dict(cfg["initial_remaining_cpu_seconds"])
    prior = {}
    checks = {"stage": stage, "config_sha256": core.digest(CONFIG), "entry_sha256": cfg["entry_sha256"],
              "status": "READY", "prior_reports": {}}
    for previous in STAGES[:STAGES.index(stage)]:
        path = out/previous/"report.json"
        if not path.exists():
            checks["status"] = "WAIT_"+previous.upper()
            break
        report = json.loads(path.read_text())
        assert report["status"] == "COMPLETE_PHASE" and report["checks"]["config_sha256"] == core.digest(CONFIG)
        for artifact in report["artifacts"]:
            assert core.digest(out/previous/artifact["path"]) == artifact["sha256"]
        remaining[category(previous)] -= report["accounted_cpu_seconds"]
        checks["prior_reports"][previous] = core.digest(path)
        prior[previous] = report
    if stage == "predict" and checks["status"] == "READY":
        path = out/"source_crops.json"
        if not path.exists():
            checks["status"] = "WAIT_SOURCE_ONLY_CROPS"
        else:
            verify_crops(json.loads(path.read_text()), prior["prepare"]["decoded"])
            checks["source_crops_sha256"] = core.digest(path)
    checks["remaining_cpu_seconds"] = remaining
    assert remaining[category(stage)] > 0
    return cfg, checks, prior


def decoder_module(cfg, directory):
    source = ROOT/cfg["pins"]["decoder"]["path"]
    spec = importlib.util.spec_from_file_location("tst_decode", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = directory
    return module


def read16(path):
    value = tifffile.imread(path)
    assert value.dtype == np.uint16 and value.ndim == 3 and value.shape[-1] == 3
    return value.astype(np.float64)/65535


def write16(path, value, icc):
    tifffile.imwrite(path, np.rint(np.clip(value, 0, 1)*65535).astype(np.uint16), photometric="rgb",
                     extratags=[(34675, "B", len(icc), icc, False)])


def scoring_pixels(source, source_sha, seed):
    return core.scoring_folds(*source.shape[:2], source_sha, seed)[1]


def prepare(cfg, directory):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(cfg["seed"])
    for entry in cfg["files"]:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert entry["revision"] == cfg["revision"]
        path = directory/"originals"/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix+".part")
        url = f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{cfg["revision"]}/{entry["path"]}'
        count = 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=60) as response, partial.open("xb") as stream:
            while chunk := response.read(65536):
                count += len(chunk)
                assert count <= entry["bytes"]
                stream.write(chunk)
        assert count == entry["bytes"] and core.digest(partial) == entry["etag"]
        partial.rename(path)
        print("acquired", entry["path"], flush=True)
    shutil.copytree(ROOT/cfg["terms_source"], directory/"terms")
    decoder, model = decoder_module(cfg, directory), core.load_vgg(ROOT)
    decoded, features = {}, {}
    for group in cfg["groups"]:
        g, (first, second) = group["group_index"], group["triplets"]
        for role, relative in (("X", first["content"]), ("R1", first["reference"]), ("R2", second["reference"])):
            key = f"{g:02d}_{role}"
            _, meta = decoder.display_decode(directory/"originals"/relative, directory/"display_srgb16"/(key+".tif"))
            meta["original_path"] = relative
            decoded[key] = meta
            stats, vgg = core.descriptors(read16(directory/meta["srgb16_path"]), model)
            features[key+"_stats"], features[key+"_vgg"] = stats, vgg
            print("described", key, flush=True)
    np.savez(directory/"features.npz", **features)
    return {"decoded": decoded, "original_files": 40, "feature_forwards": 24,
            "target_pixel_decodes": 0, "review": "Only source X previews may be used to freeze protected crops before predictions."}


def predict(cfg, directory, prior):
    out = ROOT/cfg["output"]
    model = load_model(ROOT/cfg["pins"]["model"]["path"])
    cache = np.load(out/"prepare/features.npz", allow_pickle=False)
    crops = json.loads((out/"source_crops.json").read_text())
    verify_crops(crops, prior["prepare"]["decoded"])
    save(directory/"source_crops.json", crops)
    operators, audit = {}, []
    for group in cfg["groups"]:
        g = group["group_index"]
        source_meta = prior["prepare"]["decoded"][f"{g:02d}_X"]
        source_path = out/"prepare"/source_meta["srgb16_path"]
        source = read16(source_path)
        with tifffile.TiffFile(source_path) as handle:
            icc = handle.pages[0].tags[34675].value
        source_stats, source_vgg = cache[f"{g:02d}_X_stats"], cache[f"{g:02d}_X_vgg"]
        previous_a = None
        for j in (1, 2):
            values = predict_operators(model, source_stats, source_vgg, cache[f"{g:02d}_R{j}_stats"], cache[f"{g:02d}_R{j}_vgg"])
            if previous_a is not None:
                assert np.array_equal(values["A"], previous_a)
            previous_a = values["A"]
            for arm, nodes in values.items():
                key = f"{g:02d}_Y{j}_{arm}"
                operators[key] = nodes
                flat = source.reshape(-1, 3)
                predicted = np.empty_like(flat)
                for start in range(0, len(flat), 32768):
                    predicted[start:start+32768] = core.render(flat[start:start+32768], nodes)
                predicted = predicted.reshape(source.shape)
                for strength in (1., .8):
                    image = source*(1-strength)+predicted*strength
                    name = key+("_native" if strength == 1 else "_strength08")
                    write16(directory/(name+".tif"), image, icc)
                    for label, box in crops["groups"][g-24]["regions"].items():
                        if box:
                            x, y, w, h = box
                            Image.fromarray(np.rint(image[y:y+h, x:x+w]*255).astype(np.uint8)).save(directory/(name+"_"+label+".png"), icc_profile=icc)
            audit.append({"group": g, "reference": j, "input_keys": [f"{g:02d}_X", f"{g:02d}_R{j}"],
                          "input_sha256": [source_meta["srgb16_sha256"], prior["prepare"]["decoded"][f"{g:02d}_R{j}"]["srgb16_sha256"]]})
    np.savez(directory/"operators.npz", **operators)
    prediction_files = [{"path": p.name, "sha256": core.digest(p)} for p in sorted(directory.glob("*.tif"))]
    assert len(prediction_files) == 96
    save(directory/"prediction_lock.json", {"outputs": prediction_files, "operators_sha256": core.digest(directory/"operators.npz"),
          "source_crops_sha256": core.digest(directory/"source_crops.json"), "inference_input_audit": audit, "Y_pixels_read": 0})
    return {"prediction_files": 96, "target_pixel_decodes": 0, "prediction_lock_sha256": core.digest(directory/"prediction_lock.json")}


def targets(cfg, directory, prior):
    out = ROOT/cfg["output"]
    assert prior["predict"]["target_pixel_decodes"] == 0
    decoder = decoder_module(cfg, directory)
    decoded = {}
    for group in cfg["groups"]:
        g = group["group_index"]
        for j, row in enumerate(group["triplets"], 1):
            key = f"{g:02d}_Y{j}"
            _, meta = decoder.display_decode(out/"prepare/originals"/row["gt"], directory/"display_srgb16"/(key+".tif"))
            assert meta["encoded_size"] == prior["prepare"]["decoded"][f"{g:02d}_X"]["encoded_size"]
            decoded[key] = meta
    return {"decoded": decoded, "target_pixel_decodes": 16, "prediction_lock_preceded_target_decode": True,
            "alignment_limit": "Native dimensions checked; no resizing. Detailed correspondence interpretation remains conditional on source/target visual review."}


def switching(predictions, targets):
    p1, p2 = [rgb2lab(v) for v in predictions]
    y1, y2 = [rgb2lab(v) for v in targets]
    def error(a, b):
        return float(deltaE_ciede2000(a, b).mean())
    difference = error(y1, y2)
    margin = (error(p1, y2)+error(p2, y1)-error(p1, y1)-error(p2, y2))/(2*max(difference, 1))
    dp, dy = (p2-p1).ravel(), (y2-y1).ravel()
    npred, ntarget = np.linalg.norm(dp), np.linalg.norm(dy)
    cosine = float(dp @ dy / (npred*ntarget)) if npred*ntarget > 1e-15 else 0.
    ratio = float(npred/ntarget) if ntarget > 1e-15 else 0.
    return {"D": difference, "margin": margin, "cosine": cosine, "norm_ratio": ratio,
            "pass": difference >= 2 and margin >= .1 and cosine >= .5 and .5 <= ratio <= 1.5}


def score(cfg, directory, prior):
    out = ROOT/cfg["output"]
    operators = np.load(out/"predict/operators.npz", allow_pickle=False)
    groups, teachers = [], []
    for group in cfg["groups"]:
        g = group["group_index"]
        source_meta = prior["prepare"]["decoded"][f"{g:02d}_X"]
        source = read16(out/"prepare"/source_meta["srgb16_path"])
        fit, check = core.scoring_folds(*source.shape[:2], source_meta["srgb16_sha256"], cfg["seed"])
        x = source.reshape(-1, 3)
        factor = core.teacher_factor(x[fit])
        targets_rgb, predictions = [], {arm: [] for arm in "ABC"}
        row = {"group": g, "targets": []}
        for j in (1, 2):
            target_meta = prior["targets"]["decoded"][f"{g:02d}_Y{j}"]
            y = read16(out/"targets"/target_meta["srgb16_path"]).reshape(-1, 3)
            assert len(x) == len(y)
            values, residual = core.teacher_solve(factor, y[fit])
            teacher = core.render(x[check], values)
            lab_x, lab_y = rgb2lab(x[check]), rgb2lab(y[check])
            identity_error = float(deltaE_ciede2000(lab_x, lab_y).mean())
            teacher_error = float(deltaE_ciede2000(rgb2lab(teacher), lab_y).mean())
            teachers.append({"group": g, "target": j, "error": teacher_error, "identity_error": identity_error,
                             "normal_residual": residual, "adequate": teacher_error <= max(1., .25*identity_error)})
            np.save(directory/f"{g:02d}_Y{j}_teacher.npy", values)
            targets_rgb.append(y[check])
            target_row = {"target": j, "identity_error": identity_error, "arms": {}}
            for arm in "ABC":
                predicted = core.render(x[check], operators[f"{g:02d}_Y{j}_{arm}"])
                predictions[arm].append(predicted)
                errors = deltaE_ciede2000(rgb2lab(predicted), lab_y)
                target_row["arms"][arm] = {"error_mean": float(errors.mean()), "error_p95": float(np.quantile(errors, .95)),
                    "change08_mean": float(deltaE_ciede2000(rgb2lab(.2*x[check]+.8*predicted), lab_x).mean()),
                    "new_boundary_vs_X": float(np.any(((predicted <= 0) | (predicted >= 1)) & (x[check] > 0) & (x[check] < 1), axis=1).mean()),
                    "new_boundary_vs_Y": float(np.any(((predicted <= 0) | (predicted >= 1)) & (y[check] > 0) & (y[check] < 1), axis=1).mean())}
            row["targets"].append(target_row)
        row["switches"] = {arm: switching(predictions[arm], targets_rgb) for arm in "ABC"}
        assert abs(row["switches"]["A"]["margin"]) < 1e-12
        groups.append(row)
        save(directory/"partial_metrics.json", {"groups": groups, "teachers": teachers})
    identity = np.array([np.mean([t["identity_error"] for t in g["targets"]]) for g in groups])
    errors = {a: np.array([np.mean([t["arms"][a]["error_mean"] for t in g["targets"]]) for g in groups]) for a in "ABC"}
    q = {a: float(np.mean(v/np.maximum(identity, 1))) for a, v in errors.items()}
    wins = {a: int(np.sum(errors["C"] < errors[a])) for a in "AB"}
    appearance = q["C"] <= .8 and all(q["C"] <= .9*q[a] and wins[a] >= 7 for a in "AB")
    switches = sum(g["switches"]["C"]["pass"] for g in groups)
    return {"groups": groups, "teachers": teachers, "Q": q, "C_wins_over": wins,
            "appearance_numeric_pass": appearance, "C_switch_groups": switches,
            "numeric_mechanism_pass_requires_direction_review": appearance and switches >= 6,
            "representation_adequate_targets": sum(t["adequate"] for t in teachers),
            "all_groups_denominator": 8, "all_targets_denominator": 16,
            "photographic_review": "PENDING root source-crop and full-image review of all16C strength0.8 outputs; numeric change alone is not comfort.",
            "direction_review": "PENDING fixed assigned-vs-swapped all8; no target removal from numerical scoring.",
            "claim": "No promotion before directional>=6/8 and photographic criteria; file-owner independence only."}


def worker(stage, directory):
    cfg, checks, prior = preflight(stage)
    assert checks == json.loads((directory/"lock.json").read_text()) and checks["status"] == "READY"
    threadpool_limits(limits=2)
    save(directory/"report.json", {"status": "RUNNING", "checks": checks})
    function = {"prepare": prepare, "predict": predict, "targets": targets, "score": score}[stage]
    result = function(cfg, directory) if stage == "prepare" else function(cfg, directory, prior)
    result.update({"status": "COMPLETE_PHASE", "checks": checks, "worker_pid": os.getpid(), "worker_cpu_seconds": time.process_time()})
    result["artifacts"] = [{"path": str(p.relative_to(directory)), "sha256": core.digest(p)}
                           for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in ("report.json", "report.pending", "worker.log")]
    result["worker_cpu_seconds"] = time.process_time()
    save(directory/"report.json", result)


def launch(stage, cfg, checks):
    directory = ROOT/cfg["output"]/stage
    directory.mkdir(parents=True, exist_ok=False)
    save(directory/"lock.json", checks)
    shutil.copyfile(CONFIG, directory/"config.json")
    started, owned, peaks, cpu_peaks, reason = time.monotonic(), {}, {}, {}, None
    cpu_limit = checks["remaining_cpu_seconds"][category(stage)]
    with (directory/"worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__)), "--stage", stage, "--worker", str(directory)], stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        owned[root.pid] = root.create_time()
        while process.poll() is None:
            try:
                for child in root.children(recursive=True):
                    owned[child.pid] = child.create_time()
                for pid, created in list(owned.items()):
                    try:
                        member = psutil.Process(pid)
                        if member.create_time() != created:
                            continue
                        memory, cpu = member.memory_info(), member.cpu_times()
                        peaks[pid] = max(peaks.get(pid, 0), memory.rss, getattr(memory, "peak_wset", 0))
                        cpu_peaks[pid] = max(cpu_peaks.get(pid, 0), cpu.user+cpu.system)
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                break
            except psutil.AccessDenied:
                reason = "RESOURCE_UNKNOWN"
            if sum(peaks.values()) > cfg["rss_bytes"]:
                reason = "RSS_LIMIT"
            if sum(cpu_peaks.values()) >= cpu_limit:
                reason = "CPU_LIMIT"
            if time.monotonic()-started >= cfg["wall_seconds_per_stage"]:
                reason = "WALL_LIMIT"
            if reason:
                for pid, created in reversed(list(owned.items())):
                    try:
                        member = psutil.Process(pid)
                        if member.create_time() == created:
                            member.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(.1)
        code = process.wait()
    report = json.loads((directory/"report.json").read_text()) if (directory/"report.json").exists() else {}
    if "worker_pid" in report:
        pid = report["worker_pid"]
        cpu_peaks[pid] = max(cpu_peaks.get(pid, 0), report["worker_cpu_seconds"])
    spent = sum(cpu_peaks.values())
    if spent > cpu_limit:
        reason = "CPU_FINAL_ACCOUNTING_LIMIT"
    report["accounted_cpu_seconds"] = spent
    report["supervisor"] = {"exit_code": code, "reason": reason, "remaining_category_cpu_seconds": max(0, cpu_limit-spent),
                            "wall_seconds": time.monotonic()-started, "tree_peak_rss_bytes": sum(peaks.values()),
                            "owned_pid_create_times": owned, "sampling": "0.1s tree sampling, final worker process CPU, conservative sum of process memory peaks"}
    if reason or code:
        report["status"] = "TECHNICAL_INCONCLUSIVE"
    save(directory/"report.json", report)
    print(report["status"], report["supervisor"])
    return 0 if report["status"] == "COMPLETE_PHASE" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.stage, args.worker)
    else:
        config, state, _ = preflight(args.stage)
        if args.run:
            assert state["status"] == "READY"
            raise SystemExit(launch(args.stage, config, state))
        print(json.dumps(state, indent=2))
