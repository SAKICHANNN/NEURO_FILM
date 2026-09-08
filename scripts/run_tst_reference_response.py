import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import tifffile
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits

from src.eval import tst_reference_response as core

CONFIG = ROOT / "configs/tst_reference_response_v1.json"


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


def preflight(stage):
    config = json.loads(CONFIG.read_text())
    if "implementation_pins" in config:
        assert core.digest(Path(__file__)) == config["implementation_pins"]["entry"]
        assert core.digest(Path(core.__file__)) == config["implementation_pins"]["core"]
    for path, pin in (("input_report", "input_report_sha256"), ("science_result", "science_sha256"),
                      ("admission_path", "admission_sha256")):
        assert core.digest(ROOT/config[path]) == config[pin]
    parent = json.loads((ROOT/config["input_report"]).read_text())
    admission = json.loads((ROOT/config["admission_path"]).read_text())
    assert admission["status"] == "FROZEN_MECHANISM_ADMISSION"
    assert config["budget"]["threads"] == 2
    assert [r["group_index"] for r in parent["groups"]] == list(range(24))
    inputs = {}
    for group in parent["groups"]:
        assert group["split"] == "development_fit"
        for role, entry in group["arms"].items():
            path = (ROOT/config["input_report"]).parent/entry["srgb16_path"]
            assert core.digest(path) == entry["srgb16_sha256"]
            inputs[f'{group["group_index"]:02d}_{role}'] = {"path": str(path), "sha256": entry["srgb16_sha256"]}
    checks = {"stage": stage, "config_sha256": core.digest(CONFIG), "script_sha256": core.digest(Path(__file__)),
              "core_sha256": core.digest(Path(core.__file__)), "inputs": inputs,
              "status": "READY_FOR_ROOT_REVIEW", "check_pixels": 0}
    assert core.digest(ROOT/config["alignment_report"]) == config["alignment_sha256"]
    alignment = json.loads((ROOT/config["alignment_report"]).read_text())
    eligible = [(g["group"], role) for g in admission["groups"] if g["direction"] == "compatible"
                for role, state in g["detail_status"].items() if state == "admissible"]
    assert set(eligible) == {(r["group"], r["role"]) for r in alignment["rows"]}
    for row in alignment["rows"]:
        assert row["sha256"] == [inputs[f'{row["group"]:02d}_X']["sha256"],
                                  inputs[f'{row["group"]:02d}_{row["role"]}']["sha256"]]
        assert all(region["best"]["dx"] == region["best"]["dy"] == 0 for region in row["regions"])
    checks["alignment_sha256"] = config["alignment_sha256"]
    checks["conditional_rows"] = len(eligible)
    if stage == "fit":
        feature_dir = ROOT/config["output"]/"features"
        if not (feature_dir/"report.json").exists() or not config["alignment_sha256"]:
            checks["status"] = "WAIT_FEATURES_AND_ALIGNMENT"
        else:
            fr = json.loads((feature_dir/"report.json").read_text())
            assert fr["status"] == "COMPLETE_NOT_PROMOTED"
            assert fr["checks"]["config_sha256"] == core.digest(CONFIG)
            assert core.digest(feature_dir/"features.npz") == fr["feature_sha256"]
            assert core.digest(ROOT/config["alignment_report"]) == config["alignment_sha256"]
            checks["feature_sha256"] = fr["feature_sha256"]
            checks["alignment_sha256"] = config["alignment_sha256"]
    return config, admission, checks


def image(inputs, key):
    array = tifffile.imread(inputs[key]["path"])
    assert array.dtype == np.uint16 and array.ndim == 3 and array.shape[-1] == 3
    return array.astype(np.float64)/65535


def student_fit(stats, vgg, teacher_rows, excluded_group=None, unconditional=False):
    fit_groups = [g for g in range(24) if g != excluded_group]
    source_keys = [f"{g:02d}_X" for g in fit_groups]
    reference_keys = [f"{g:02d}_R{j}" for g in fit_groups for j in (1, 2)]
    def blocks(keys, include_vgg=True):
        result = [np.stack([stats[k] for k in keys])]
        if include_vgg:
            result.append(np.stack([vgg[k] for k in keys]))
        return result
    projections = {"source": core.fit_projection(blocks(source_keys)),
                   "statistics": core.fit_projection(blocks(reference_keys, False)),
                   "image": core.fit_projection(blocks(reference_keys))}
    rows = [r for r in teacher_rows if r["group"] != excluded_group and (unconditional or r["conditional"])]
    if unconditional:
        averaged = []
        for g in sorted({r["group"] for r in rows}):
            subset = [r for r in rows if r["group"] == g]
            averaged.append({**subset[0], "values": np.mean([r["values"] for r in subset], axis=0)})
        rows = averaged
    if not rows:
        return None
    keys_x = [f'{r["group"]:02d}_X' for r in rows]
    keys_r = [f'{r["group"]:02d}_R{r["target"]}' for r in rows]
    zx = core.project(projections["source"], blocks(keys_x))
    zr_stat = core.project(projections["statistics"], blocks(keys_r, False))
    zr_image = core.project(projections["image"], blocks(keys_r))
    arm_features = {"A": np.column_stack([np.ones(len(rows)), zx, np.zeros_like(zx)])}
    if not unconditional:
        arm_features.update({"B": np.column_stack([np.ones(len(rows)), zx, zr_stat]),
                             "C": np.column_stack([np.ones(len(rows)), zx, zr_image])})
    heads = core.matched_heads(arm_features, np.stack([r["values"].ravel() for r in rows]), [r["group"] for r in rows])
    return {"projections": projections, "heads": heads, "fit_rows": [(r["group"], r["target"]) for r in rows]}


def predicted_operators(model, stats, vgg, group, target):
    x, r = f"{group:02d}_X", f"{group:02d}_R{target}"
    zx = core.project(model["projections"]["source"], [stats[x][None], vgg[x][None]])[0]
    zstat = core.project(model["projections"]["statistics"], [stats[r][None]])[0]
    zimage = core.project(model["projections"]["image"], [stats[r][None], vgg[r][None]])[0]
    features = {"A": np.r_[1., zx, np.zeros(6)], "B": np.r_[1., zx, zstat], "C": np.r_[1., zx, zimage]}
    return {arm: (features[arm] @ head["coefficients"]).reshape(343, 3) for arm, head in model["heads"].items()}


def worker(stage, directory):
    config, admission, checks = preflight(stage)
    assert checks == json.loads((directory/"lock.json").read_text())
    assert checks["status"] == "READY_FOR_ROOT_REVIEW"
    threadpool_limits(limits=2)
    report = {"status": "RUNNING", "checks": checks, "rows": [], "check_pixels": 0, "model_forwards": 0}
    save(directory/"report.json", report)
    if stage == "features":
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.manual_seed(config["seed"])
        model = core.load_vgg(ROOT)
        cache = {}
        for g in range(24):
            for role in ("X", "R1", "R2"):
                key = f"{g:02d}_{role}"
                stats, vgg = core.descriptors(image(checks["inputs"], key), model)
                cache[key+"_stats"], cache[key+"_vgg"] = stats, vgg
                report["rows"].append({"key": key, "input_sha256": checks["inputs"][key]["sha256"]})
                report["model_forwards"] += 1
                save(directory/"report.json", report)
        np.savez(directory/"features.npz", **cache)
        report["feature_sha256"] = core.digest(directory/"features.npz")
    else:
        cache = np.load(ROOT/config["output"]/"features/features.npz", allow_pickle=False)
        stats = {k[:-6]: cache[k] for k in cache.files if k.endswith("_stats")}
        vgg = {k[:-4]: cache[k] for k in cache.files if k.endswith("_vgg")}
        teacher_rows, informative = [], 0
        conditional_group_count = sum(g["direction"] == "compatible" and
                                      any(s == "admissible" for s in g["detail_status"].values()) for g in admission["groups"])
        for group in admission["groups"]:
            if group["direction"] != "compatible" and conditional_group_count >= 3:
                continue
            g = group["group"]
            x_image = image(checks["inputs"], f"{g:02d}_X")
            fit, check = core.scoring_folds(*x_image.shape[:2], checks["inputs"][f"{g:02d}_X"]["sha256"], config["seed"])
            x = x_image.reshape(-1, 3)
            factor = core.teacher_factor(x[fit])
            pair = []
            for j in (1, 2):
                if group["detail_status"][f"Y{j}"] != "admissible":
                    continue
                y_image = image(checks["inputs"], f"{g:02d}_Y{j}")
                assert y_image.shape == x_image.shape
                y = y_image.reshape(-1, 3)
                values, residual = core.teacher_solve(factor, y[fit])
                predicted = core.render(x[check], values)
                row = {"group": g, "target": j, "conditional": group["direction"] == "compatible", "values": values,
                       "normal_residual": residual, "teacher_check_de00": float(deltaE_ciede2000(rgb2lab(predicted), rgb2lab(y[check])).mean())}
                teacher_rows.append(row)
                report["rows"].append({k: v for k, v in row.items() if k != "values"})
                pair.append(y[check].copy())
            if len(pair) == 2 and group["direction"] == "compatible":
                difference = float(deltaE_ciede2000(rgb2lab(pair[0]), rgb2lab(pair[1])).mean())
                informative += difference >= 2
                report.setdefault("pair_differences", []).append({"group": g, "mean_de00": difference})
            save(directory/"report.json", report)
        conditional_groups = {r["group"] for r in teacher_rows if r["conditional"]}
        report["informative_pairs"] = informative
        report["conditional_rows"] = sum(r["conditional"] for r in teacher_rows)
        unconditional = len(conditional_groups) < 3
        model = student_fit(stats, vgg, teacher_rows, unconditional=unconditional)
        save(directory/"model.json", model)
        save(directory/"teachers.json", teacher_rows)
        if informative < 8:
            report["route"] = "UNCONDITIONAL_CALIBRATION_NO_GO_REFERENCE" if unconditional else "OWNER_LOO_CONSUMED_CALIBRATION"
            groups = sorted({r["group"] for r in teacher_rows if unconditional or r["conditional"]})
            for g in groups:
                loo = student_fit(stats, vgg, teacher_rows, excluded_group=g, unconditional=unconditional)
                if loo is None:
                    report["rows"].append({"loo_group": g, "status": "NO_REMAINING_TRAINING_GROUP"})
                    continue
                for row in [r for r in teacher_rows if r["group"] == g and (unconditional or r["conditional"])]:
                    operators = predicted_operators(loo, stats, vgg, g, row["target"])
                    save(directory/f'loo_{g:02d}_Y{row["target"]}.json', operators)
                    x_image = image(checks["inputs"], f"{g:02d}_X")
                    _, scoring = core.scoring_folds(*x_image.shape[:2], checks["inputs"][f"{g:02d}_X"]["sha256"], config["seed"])
                    x = x_image.reshape(-1, 3)[scoring]
                    y = image(checks["inputs"], f'{g:02d}_Y{row["target"]}').reshape(-1, 3)[scoring]
                    report.setdefault("loo_scores", []).append({"group": g, "target": row["target"],
                        "identity_de00": float(deltaE_ciede2000(rgb2lab(x), rgb2lab(y)).mean()),
                        "arm_de00": {arm: float(deltaE_ciede2000(rgb2lab(core.render(x, values)), rgb2lab(y)).mean())
                                      for arm, values in operators.items()}})
                    save(directory/"report.json", report)
        else:
            report["route"] = "HEADS_FIT_CHECK_STILL_UNOPENED"
        report["model_sha256"] = core.digest(directory/"model.json")
        report["teacher_sha256"] = core.digest(directory/"teachers.json")
    report["worker_pid"] = os.getpid()
    report["worker_cpu_seconds"] = time.process_time()
    report["status"] = "COMPLETE_NOT_PROMOTED"
    save(directory/"report.json", report)


def launch(stage, config, checks):
    directory = ROOT/config["output"]/stage
    directory.mkdir(parents=True, exist_ok=False)
    save(directory/"lock.json", checks)
    save(directory/"config.json", config)
    started = time.monotonic()
    owned, peaks, cpu_peaks = {}, {}, {}
    reason = None
    with (directory/"worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__)), "--stage", stage, "--worker", str(directory)],
                                   stdout=log, stderr=subprocess.STDOUT)
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
                reason = "RESOURCE_STATE_UNKNOWN"
            if sum(peaks.values()) > config["budget"]["tree_rss_bytes"]:
                reason = "RSS_LIMIT"
            cpu_limit = config["budget"]["feature_cumulative_process_cpu_seconds" if stage == "features"
                                         else "learning_render_scoring_cumulative_process_cpu_seconds"]
            if sum(cpu_peaks.values()) >= cpu_limit:
                reason = "CPU_LIMIT"
            if time.monotonic()-started >= config["budget"]["stage_wall_seconds"]:
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
    accounted = sum(cpu_peaks.values())
    cpu_limit = config["budget"]["feature_cumulative_process_cpu_seconds" if stage == "features"
                                 else "learning_render_scoring_cumulative_process_cpu_seconds"]
    if report.get("worker_cpu_seconds", 0) > config["budget"]["feature_cumulative_process_cpu_seconds" if stage == "features"
                                                            else "learning_render_scoring_cumulative_process_cpu_seconds"]:
        reason = "CPU_LIMIT_FINAL_ACCOUNTING"
    report["supervisor"] = {"exit_code": code, "reason": reason, "cpu_seconds_observed": sum(cpu_peaks.values()),
                            "wall_seconds": time.monotonic()-started, "tree_historical_peak_rss": sum(peaks.values()),
                            "owned_pids": owned, "limitation": "0.1s PID/create-time bound process-tree sampling; final sub-sample CPU tail may be unobserved."}
    report["cumulative_budget_ledger"] = {"category": "feature_extraction" if stage == "features" else "learning_render_scoring",
        "total_cpu_limit_seconds": cpu_limit, "accounted_cpu_seconds": accounted,
        "remaining_cpu_seconds": max(0., cpu_limit-accounted),
        "rule": "Any future check phase in this category must subtract this expenditure and all intervening same-category expenditure from600; no fresh600 allowance. Preserve these hashes with check protocol."}
    if accounted > cpu_limit:
        reason = "CPU_LIMIT_FINAL_ACCOUNTING"
        report["supervisor"]["reason"] = reason
    if reason or code:
        report["status"] = "TECHNICAL_INCONCLUSIVE"
    save(directory/"report.json", report)
    print(report["status"])
    return 0 if report["status"] == "COMPLETE_NOT_PROMOTED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features", "fit"], required=True)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.stage, args.worker)
    else:
        cfg, _, state = preflight(args.stage)
        if args.run:
            assert state["status"] == "READY_FOR_ROOT_REVIEW"
            raise SystemExit(launch(args.stage, cfg, state))
        print(json.dumps({k: v for k, v in state.items() if k != "inputs"}, indent=2))
