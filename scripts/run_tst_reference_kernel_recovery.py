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
CONFIG = ROOT/"configs/tst_reference_kernel_recovery_v1.json"


def runtime():
    from src.eval import tst_reference_response as core

    cfg = json.loads(CONFIG.read_text())
    for path, expected in cfg["pins"].items():
        assert core.digest(ROOT/path) == expected, path
    assert cfg["entry_sha256"] == core.digest(Path(__file__))
    previous = json.loads((ROOT/cfg["recovery"]["report"]).read_text())
    old_cfg = json.loads((ROOT/cfg["recovery"]["config"]).read_text())
    assert previous["status"] == "NUMERICAL_INCOMPLETE"
    exceptions = {"pins", "entry_sha256", "identity", "output", "cpu_seconds", "wall_seconds_per_stage", "recovery"}
    assert {k: v for k, v in cfg.items() if k not in exceptions} == {k: v for k, v in old_cfg.items() if k not in exceptions}
    assert all(cfg["pins"].get(path) == value for path, value in old_cfg["pins"].items())
    assert cfg["cpu_seconds"] == old_cfg["cpu_seconds"]-previous["accounted_cpu_seconds"]
    assert abs(cfg["wall_seconds_per_stage"]-(old_cfg["wall_seconds_per_stage"]-previous["supervisor"]["wall_seconds"])) < 1e-9
    assert cfg["output"] != old_cfg["output"]
    assert all(not s["solved"] for a in cfg["arms"] for s in previous["solver_states"][a])
    assert all(s["solved"] for s in previous["solver_states"]["freegrid"])
    path = ROOT/"scripts/run_tst_reference_kernel.py"
    spec = importlib.util.spec_from_file_location("frozen_experiment", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    import numpy as np

    from src.eval.tst_reference_kernel_recovery import RecoveryKernel

    with np.load(ROOT/cfg["recovery"]["weights"], allow_pickle=False) as values:
        assert set(values.files) == {*cfg["arms"], "freegrid"}
        freegrid = values["freegrid"].copy()
    assert freegrid.shape == (43, 343, 3) and np.isfinite(freegrid).all()
    entry.kernel = RecoveryKernel(freegrid, previous["solver_states"]["freegrid"], cfg["rtol"], cfg["maxiter"])
    entry.CONFIG = CONFIG
    entry.__file__ = __file__
    original_preflight = entry.preflight

    def preflight():
        config, state, legacy = original_preflight()
        state["recovery"] = {"original_report_sha256": core.digest(ROOT/cfg["recovery"]["report"]),
                             "original_cpu_seconds": previous["accounted_cpu_seconds"],
                             "original_wall_seconds": previous["supervisor"]["wall_seconds"],
                             "freegrid_reused_channels": 3, "conditional_new_channels": 12,
                             "loss_kernel_hessian_rhs_tolerance_unchanged": True}
        return config, state, legacy

    entry.preflight = preflight
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--stage", default="experiment", choices=["experiment"])
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    entry = runtime()
    if args.worker:
        _, _, legacy = entry.preflight()
        try:
            entry.worker(args.worker)
        finally:
            legacy.save(args.worker/"preconditioner_audit.json", entry.kernel.audit)
            report_path = args.worker/"report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                report["worker_pid"] = os.getpid()
                report["worker_cpu_seconds"] = time.process_time()
                legacy.save(report_path, report)
    else:
        cfg, checks, legacy = entry.preflight()
        if args.run:
            legacy.__file__ = __file__
            legacy.CONFIG = CONFIG
            code = legacy.launch("experiment", cfg, checks)
            directory = ROOT/cfg["output"]/"experiment"
            report = json.loads((directory/"report.json").read_text())
            report["recovery_ledger"] = {"original_report_sha256": checks["recovery"]["original_report_sha256"],
                "total_cpu_seconds": checks["recovery"]["original_cpu_seconds"]+report["accounted_cpu_seconds"],
                "total_active_wall_seconds": checks["recovery"]["original_wall_seconds"]+report["supervisor"]["wall_seconds"],
                "freegrid_reused": True, "original_attempt_unchanged": True}
            if report["supervisor"]["reason"] in ("RSS_LIMIT", "CPU_LIMIT", "CPU_FINAL_ACCOUNTING_LIMIT", "WALL_LIMIT"):
                report["status"] = "RESOURCE_INCOMPLETE"
            if (directory/"preconditioner_audit.json").exists():
                report["preconditioner_audit_sha256"] = entry.core.digest(directory/"preconditioner_audit.json")
                if "artifacts" in report:
                    report["artifacts"].append({"path": "preconditioner_audit.json", "sha256": report["preconditioner_audit_sha256"]})
            legacy.save(directory/"report.json", report)
            raise SystemExit(code)
        print(json.dumps(checks, indent=2))
