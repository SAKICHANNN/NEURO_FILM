import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tst_reference_fit_replay_v1"
PROTOCOL = OUTPUT / "protocol.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    cfg = json.loads(PROTOCOL.read_text())
    for path, expected in cfg["pins"].items():
        assert digest(ROOT/path) == expected, path
    assert cfg["script_sha256"] == digest(Path(__file__))
    spec = importlib.util.spec_from_file_location("frozen_check", ROOT/"scripts/run_tst_reference_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = json.loads((ROOT/"outputs/tst_reference_response_v1/fit/report.json").read_text())
    groups = [r["group"] for r in report["pair_differences"] if r["mean_de00"] >= 2]
    assert groups == cfg["groups"] and len(groups) == 20
    for g in groups:
        for role in ("X", "Y1", "Y2"):
            meta = report["checks"]["inputs"][f"{g:02d}_{role}"]
            assert digest(Path(meta["path"])) == meta["sha256"]
    checks = {"status": "READY", "protocol_sha256": digest(PROTOCOL),
              "remaining_cpu_seconds": {"learning": cfg["cpu_seconds"]}}
    return cfg, checks, module, report


def worker(directory):
    cfg, checks, m, fit_report = prepare()
    assert checks == json.loads((directory/"lock.json").read_text())
    m.threadpool_limits(limits=2)
    m.save(directory/"report.json", {"status": "RUNNING", "checks": checks})
    model = m.load_model(ROOT/"outputs/tst_reference_response_v1/fit/model.json")
    cache = m.np.load(ROOT/"outputs/tst_reference_response_v1/features/features.npz", allow_pickle=False)
    teachers = json.loads((ROOT/"outputs/tst_reference_response_v1/fit/teachers.json").read_text())
    teacher_map = {(r["group"], r["target"]): r for r in teachers}
    rows = []
    for g in cfg["groups"]:
        xkey = f"{g:02d}_X"
        meta = fit_report["checks"]["inputs"][xkey]
        source = m.tifffile.imread(meta["path"])
        assert source.dtype == m.np.uint16
        _, held = m.core.scoring_folds(*source.shape[:2], meta["sha256"], cfg["seed"])
        x = source.reshape(-1, 3)[held].astype(float)/65535
        lab_x = m.rgb2lab(x)
        row = {"group": g, "targets": []}
        predictions, targets = {a: [] for a in "ABC"}, []
        for j in (1, 2):
            ymeta = fit_report["checks"]["inputs"][f"{g:02d}_Y{j}"]
            raw = m.tifffile.imread(ymeta["path"])
            assert raw.dtype == m.np.uint16 and raw.shape == source.shape
            y = raw.reshape(-1, 3)[held].astype(float)/65535
            targets.append(y)
            lab_y = m.rgb2lab(y)
            identity = float(m.deltaE_ciede2000(lab_x, lab_y).mean())
            rkey = f"{g:02d}_R{j}"
            operators = m.predict_operators(model, cache[xkey+"_stats"], cache[xkey+"_vgg"],
                                           cache[rkey+"_stats"], cache[rkey+"_vgg"])
            teacher = m.core.render(x, m.np.asarray(teacher_map[g, j]["values"]))
            tr = {"target": j, "identity_error": identity,
                  "teacher_error": float(m.deltaE_ciede2000(m.rgb2lab(teacher), lab_y).mean()), "arms": {}}
            assert abs(tr["teacher_error"]-teacher_map[g, j]["teacher_check_de00"]) < 1e-10
            for arm in "ABC":
                p = m.core.render(x, operators[arm])
                predictions[arm].append(p)
                errors = m.deltaE_ciede2000(m.rgb2lab(p), lab_y)
                tr["arms"][arm] = {"error_mean": float(errors.mean()),
                    "error_p95": float(m.np.quantile(errors, .95)),
                    "change08_mean": float(m.deltaE_ciede2000(m.rgb2lab(.2*x+.8*p), lab_x).mean()),
                    "new_boundary_vs_X": float(m.np.any(((p <= 0) | (p >= 1)) & (x > 0) & (x < 1), axis=1).mean()),
                    "new_boundary_vs_Y": float(m.np.any(((p <= 0) | (p >= 1)) & (y > 0) & (y < 1), axis=1).mean())}
            row["targets"].append(tr)
        row["switches"] = {a: m.switching(predictions[a], targets) for a in "ABC"}
        assert m.np.array_equal(*predictions["A"])
        rows.append(row)
        m.save(directory/"partial_metrics.json", {"groups": rows})
    identity = m.np.array([m.np.mean([t["identity_error"] for t in r["targets"]]) for r in rows])
    errors = {a: m.np.array([m.np.mean([t["arms"][a]["error_mean"] for t in r["targets"]]) for r in rows]) for a in "ABC"}
    result = {"status": "COMPLETE_PHASE", "checks": checks, "groups": rows,
              "Q": {a: float(m.np.mean(v/m.np.maximum(identity, 1))) for a, v in errors.items()},
              "C_wins_over": {a: int(m.np.sum(errors["C"] < errors[a])) for a in "AB"},
              "switch_pass_groups": {a: sum(r["switches"][a]["pass"] for r in rows) for a in "ABC"},
              "groups_denominator": 20, "targets_denominator": 40,
              "teacher_solves": 0, "head_fits": 0, "feature_forwards": 0,
              "claim": "Consumed fit-owner replay; held pixels are not held owners. No promotion, retuning or generalization claim.",
              "worker_pid": os.getpid(), "worker_cpu_seconds": time.process_time()}
    m.save(directory/"report.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="replay", choices=["replay"])
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        config, state, module, _ = prepare()
        if args.run:
            module.__file__ = __file__
            module.CONFIG = PROTOCOL
            raise SystemExit(module.launch("replay", config, state))
        print(json.dumps(state, indent=2))
