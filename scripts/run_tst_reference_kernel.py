import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tifffile
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from threadpoolctl import threadpool_limits

from src.eval import tst_reference_kernel as kernel
from src.eval import tst_reference_response as core

CONFIG = ROOT/"configs/tst_reference_kernel_v1.json"
PROTOCOL_CONSTANTS = {
    "kernel_eigen_cutoff": 1e-10, "strength": 1., "render_chunk_pixels": 32768,
    "fit_joint_groups": 18, "fit_singletons": 3, "Q_ceiling": .35, "oracle_Q_allowance": .10,
    "development_joint_groups": 6, "baseline_Q_improvement": .05, "baseline_group_wins": 6,
    "appearance_absolute": 1., "appearance_identity_fraction": .25,
}


def preflight():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["protocol_constants"] == PROTOCOL_CONSTANTS
    assert cfg["threads"] == 2 and cfg["dimension"] == 7
    assert cfg["arms"] == ["A", "B", "C6", "Cfull"]
    assert kernel.EIGEN_RELATIVE_CUTOFF == PROTOCOL_CONSTANTS["kernel_eigen_cutoff"]
    for name, sha in cfg["pins"].items():
        assert core.digest(ROOT/name) == sha, name
    assert cfg["entry_sha256"] == core.digest(Path(__file__))
    assert len(cfg["fit_rows"]) == 43 and len(cfg["development_rows"]) == 16
    saved_rows = json.loads((ROOT/cfg["old_model"]).read_text())["fit_rows"]
    assert [[r["group"], r["target"]] for r in cfg["fit_rows"]] == saved_rows
    assert len({r["group"] for r in cfg["fit_rows"]}) == 23
    assert {r["group"] for r in cfg["development_rows"]} == set(range(24, 32))
    for meta in cfg["images"].values():
        assert core.digest(ROOT/meta["path"]) == meta["sha256"]
    state = {"status": "READY", "config_sha256": core.digest(CONFIG),
             "remaining_cpu_seconds": {"learning": cfg["cpu_seconds"]},
             "scope": "43 fixed fit rows; consumed development8; no feature forward/download"}
    spec = importlib.util.spec_from_file_location("frozen_check", ROOT/"scripts/run_tst_reference_check.py")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return cfg, state, legacy


def features(old, cache, keys):
    xs = np.stack([cache[f'{r["group"]:02d}_X_stats'] for r in keys])
    xv = np.stack([cache[f'{r["group"]:02d}_X_vgg'] for r in keys])
    rs = np.stack([cache[f'{r["group"]:02d}_R{r["target"]}_stats'] for r in keys])
    rv = np.stack([cache[f'{r["group"]:02d}_R{r["target"]}_vgg'] for r in keys])
    p = old["projections"]
    source = core.project(p["source"], [xs, xv])
    stats = kernel.pre_pca(p["image"]["standards"][:1], [rs])
    references = {"A": None, "B": stats,
                  "C6": core.project(p["image"], [rs, rv]),
                  "Cfull": kernel.pre_pca(p["image"]["standards"], [rs, rv])}
    return source, references


def read_image(cfg, key):
    values = tifffile.imread(ROOT/cfg["images"][key]["path"])
    assert values.dtype == np.uint16 and values.ndim == 3 and values.shape[-1] == 3
    return values.astype(np.float64)/65535


def sampled_source(cfg, group, fold):
    key = f"{group:02d}_X"
    source = read_image(cfg, key)
    indices = core.scoring_folds(*source.shape[:2], cfg["images"][key]["sha256"], cfg["seed"], cfg["pixels_per_fold"])[fold]
    return source.reshape(-1, 3)[indices], indices, source.shape


def metrics(cfg, rows, operators, legacy):
    result = []
    for g in dict.fromkeys(r["group"] for r in rows):
        x, indices, shape = sampled_source(cfg, g, 1)
        lab_x = rgb2lab(x)
        targets, predictions, target_rows = [], {a: [] for a in operators}, []
        for r in (r for r in rows if r["group"] == g):
            key = f'{g:02d}_Y{r["target"]}'
            whole = read_image(cfg, key)
            assert whole.shape == shape
            y = whole.reshape(-1, 3)[indices]
            targets.append(y)
            lab_y = rgb2lab(y)
            identity = float(deltaE_ciede2000(lab_x, lab_y).mean())
            tr = {"target": r["target"], "identity_error": identity, "arms": {}}
            for arm, nodes in operators.items():
                p = core.render(x, nodes[key])
                predictions[arm].append(p)
                error = deltaE_ciede2000(rgb2lab(p), lab_y)
                tr["arms"][arm] = {"error_mean": float(error.mean()), "error_p95": float(np.quantile(error, .95)),
                    "appearance_pass": bool(error.mean() <= max(1., .25*identity)),
                    "change_native_mean": float(deltaE_ciede2000(rgb2lab(p), lab_x).mean()),
                    "new_boundary_vs_X": float(np.any(((p <= 0) | (p >= 1)) & (x > 0) & (x < 1), axis=1).mean()),
                    "new_boundary_vs_Y": float(np.any(((p <= 0) | (p >= 1)) & (y > 0) & (y < 1), axis=1).mean())}
            target_rows.append(tr)
        switches = {a: legacy.switching(p, targets) for a, p in predictions.items()} if len(targets) == 2 else {}
        if switches:
            assert np.array_equal(*predictions["A"])
        result.append({"group": g, "targets": target_rows, "switches": switches})
    summaries = {}
    for subset_name, selected in (("all_groups", result), ("complete_pairs", [r for r in result if len(r["targets"]) == 2])):
        identity = np.array([np.mean([t["identity_error"] for t in r["targets"]]) for r in selected])
        errors = {a: np.array([np.mean([t["arms"][a]["error_mean"] for t in r["targets"]]) for r in selected]) for a in operators}
        summaries[subset_name] = {"groups": len(selected), "Q": {a: float(np.mean(e/np.maximum(identity, 1))) for a, e in errors.items()},
            "Cfull_wins_over": {a: int(np.sum(errors["Cfull"] < e)) for a, e in errors.items() if a != "Cfull"},
            "joint_appearance_switch_pass": {a: sum(bool(r["switches"]) and r["switches"][a]["pass"] and all(t["arms"][a]["appearance_pass"] for t in r["targets"]) for r in selected) for a in operators},
            "singleton_appearance_pass": {a: sum(len(r["targets"]) == 1 and r["targets"][0]["arms"][a]["appearance_pass"] for r in selected) for a in operators}}
    return {"groups": result, "summary": summaries}


def worker(directory):
    cfg, state, legacy = preflight()
    assert state == json.loads((directory/"lock.json").read_text())
    threadpool_limits(limits=cfg["threads"])
    legacy.save(directory/"report.json", {"status": "RUNNING", "checks": state})
    old = legacy.load_model(ROOT/cfg["old_model"])
    cache = {}
    for path in cfg["feature_caches"]:
        with np.load(ROOT/path, allow_pickle=False) as data:
            cache.update({k: data[k] for k in data.files})
    xfit, rfit = features(old, cache, cfg["fit_rows"])
    models, phis = {}, {}
    for arm in cfg["arms"]:
        models[arm], phis[arm] = kernel.kernel_features(xfit, rfit[arm])
    groups = []
    for g in dict.fromkeys(r["group"] for r in cfg["fit_rows"]):
        x, indices, shape = sampled_source(cfg, g, 0)
        row_ids = [i for i, r in enumerate(cfg["fit_rows"]) if r["group"] == g]
        targets = []
        for i in row_ids:
            row = cfg["fit_rows"][i]
            whole = read_image(cfg, f'{g:02d}_Y{row["target"]}')
            assert whole.shape == shape
            targets.append(whole.reshape(-1, 3)[indices])
        group = kernel.sufficient(x, targets, row_ids, cfg["dimension"])
        delta = np.stack([y-x for y in targets])
        group["constant"] = float(np.einsum("ipc,ij,jpc->", delta, kernel.pair_matrix(len(row_ids)), delta)/len(x))
        groups.append(group)
    reg = kernel.regularizer(cfg["dimension"], cfg["smoothness"], cfg["residual_l2"])
    beta = len(groups)*cfg["weight_l2"]/(len(cfg["fit_rows"])*cfg["dimension"]**3)
    phis["freegrid"] = np.eye(len(cfg["fit_rows"]))
    systems = {a: kernel.normal_system(phi, groups, reg, 0. if a == "freegrid" else beta) for a, phi in phis.items()}
    weights = {a: np.zeros_like(system[2]) for a, system in systems.items()}
    solver_states = {a: [] for a in systems}
    for channel in range(3):
        for arm, (operator, preconditioner, rhs, _) in systems.items():
            weights[arm][:, :, channel], solved = kernel.solve_channel(operator, preconditioner, rhs[:, :, channel], cfg["rtol"], cfg["maxiter"])
            solver_states[arm].append(solved)
            legacy.save(directory/"solver_states.json", solver_states)
            np.savez(directory/"partial_weights.npz", **weights)
            print("channel", channel, arm, solved, flush=True)
    if not all(s["solved"] for states in solver_states.values() for s in states):
        legacy.save(directory/"report.json", {"status": "NUMERICAL_INCOMPLETE", "solver_states": solver_states,
                    "worker_pid": os.getpid(), "worker_cpu_seconds": time.process_time()})
        return
    losses = {}
    for a, phi in phis.items():
        u = np.einsum("nr,rdc->ndc", phi, weights[a])
        losses[a] = kernel.objective_from_statistics(u, groups, reg, weights[a], 0. if a == "freegrid" else beta)
        losses[a]["full_objective"] = losses[a]["normalized_without_constant"]+sum(g["constant"] for g in groups)/(len(groups)*3)
        if a != "freegrid":
            models[a]["weights"] = weights[a]
    legacy.save(directory/"models.json", models)
    legacy.save(directory/"losses.json", losses)
    np.savez(directory/"weights.npz", **weights)
    all_rows = cfg["fit_rows"]+cfg["development_rows"]
    xall, rall = features(old, cache, all_rows)
    operators = {}
    for arm in cfg["arms"]:
        predicted = kernel.predict_operators(models[arm], xall, rall[arm])
        operators[arm] = {f'{r["group"]:02d}_Y{r["target"]}': v for r, v in zip(all_rows, predicted, strict=True)}
    prediction_dir = directory/"predictions"
    prediction_dir.mkdir()
    crop_data = json.loads((ROOT/cfg["source_crops"]).read_text())
    legacy.save(directory/"source_crops.json", crop_data)
    for row in all_rows:
        g, j = row["group"], row["target"]
        source_path = ROOT/cfg["images"][f"{g:02d}_X"]["path"]
        source = read_image(cfg, f"{g:02d}_X")
        with tifffile.TiffFile(source_path) as handle:
            icc = handle.pages[0].tags[34675].value
        flat = source.reshape(-1, 3)
        for arm in cfg["arms"] if g >= 24 else ["Cfull"]:
            predicted = np.empty_like(flat)
            key = f"{g:02d}_Y{j}"
            for start in range(0, len(flat), 32768):
                predicted[start:start+32768] = core.render(flat[start:start+32768], operators[arm][key])
            predicted = predicted.reshape(source.shape)
            legacy.write16(prediction_dir/f"{key}_{arm}_native.tif", predicted, icc)
            if g >= 24 and arm == "Cfull":
                for label, box in crop_data["groups"][g-24]["regions"].items():
                    if box:
                        x, y, w, h = box
                        Image.fromarray(np.rint(predicted[y:y+h, x:x+w]*255).astype(np.uint8)).save(prediction_dir/f"{key}_{arm}_{label}.png", icc_profile=icc)
    np.savez(directory/"operators.npz", **{f"{key}_{a}": v for a, values in operators.items() for key, v in values.items()})
    files = [{"path": str(p.relative_to(directory)), "sha256": core.digest(p)} for p in sorted(prediction_dir.glob("*.tif"))]
    assert len(files) == 107
    legacy.save(directory/"prediction_lock.json", {"outputs": files, "operators_sha256": core.digest(directory/"operators.npz"),
        "crops": [{"path": str(p.relative_to(directory)), "sha256": core.digest(p)} for p in sorted(prediction_dir.glob("*.png"))],
        "models_sha256": core.digest(directory/"models.json"), "source_crops_sha256": core.digest(directory/"source_crops.json"),
        "development_Y_pixel_reads_so_far": 0, "claim": "Post-hoc consumed owner-separated development, not unseen."})
    fit_operators = {**operators, "freegrid": {f'{r["group"]:02d}_Y{r["target"]}': values for r, values in zip(cfg["fit_rows"], weights["freegrid"], strict=True)}}
    fit_metrics = metrics(cfg, cfg["fit_rows"], fit_operators, legacy)
    legacy.save(directory/"fit_metrics.json", fit_metrics)
    dev_metrics = metrics(cfg, cfg["development_rows"], operators, legacy)
    legacy.save(directory/"development_metrics.json", dev_metrics)
    old_fit = json.loads((ROOT/cfg["old_fit_replay"]).read_text())
    old_dev = json.loads((ROOT/cfg["old_development_metrics"]).read_text())
    historical = {"strength": 1., "old_C_fit_paired20_Q": old_fit["Q"]["C"],
        "new_Cfull_fit_paired20_Q": fit_metrics["summary"]["complete_pairs"]["Q"]["Cfull"],
        "old_C_development8_Q": old_dev["Q"]["C"],
        "new_Cfull_development8_Q": dev_metrics["summary"]["all_groups"]["Q"]["Cfull"],
        "limit": "Numeric native1 comparison; prior uniform0.8 photographicFAIL not overwritten."}
    fit = fit_metrics["summary"]["all_groups"]
    dev = dev_metrics["summary"]["all_groups"]
    gates = {"strong_fit": fit["joint_appearance_switch_pass"]["Cfull"] >= 18 and fit["singleton_appearance_pass"]["Cfull"] == 3 and fit["Q"]["Cfull"] <= min(.35, fit["Q"]["freegrid"]+.10),
             "development_mechanism": dev["joint_appearance_switch_pass"]["Cfull"] >= 6 and dev["Q"]["Cfull"] <= .35 and dev["Q"]["A"]-dev["Q"]["Cfull"] >= .05 and dev["Cfull_wins_over"]["A"] >= 6,
             "feature_benefit": dev["Q"]["B"]-dev["Q"]["Cfull"] >= .05 and dev["Cfull_wins_over"]["B"] >= 6}
    artifacts = [{"path": str(p.relative_to(directory)), "sha256": core.digest(p)} for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in ("report.json", "worker.log")]
    legacy.save(directory/"report.json", {"status": "COMPLETE_PHASE", "checks": state, "gates": gates,
        "fit_summary": fit_metrics["summary"], "development_summary": dev_metrics["summary"], "solver_states": solver_states,
        "historical_native1_comparison": historical,
        "losses": losses, "kernel_ranks": {a: model["rank"] for a, model in models.items()}, "artifacts": artifacts,
        "photographic_review": "PENDING all16Cfull native1 and fixed crops,6/8bothcomfortable-rich-changed-distinct and0severe observed failures; no promotion",
        "worker_pid": os.getpid(), "worker_cpu_seconds": time.process_time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--stage", default="experiment", choices=["experiment"])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        cfg, checks, legacy = preflight()
        if args.run:
            legacy.__file__ = __file__
            legacy.CONFIG = CONFIG
            code = legacy.launch("experiment", cfg, checks)
            report_path = ROOT/cfg["output"]/"experiment/report.json"
            report = json.loads(report_path.read_text())
            if report["supervisor"]["reason"] in ("RSS_LIMIT", "CPU_LIMIT", "CPU_FINAL_ACCOUNTING_LIMIT", "WALL_LIMIT"):
                report["status"] = "RESOURCE_INCOMPLETE"
                legacy.save(report_path, report)
            raise SystemExit(code)
        print(json.dumps(checks, indent=2))
