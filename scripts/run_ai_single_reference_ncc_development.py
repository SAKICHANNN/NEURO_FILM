from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_ai_single_reference_batch_development as baseline
from scripts.run_ai_vcg_reference import checked_json, read_rgb, sha

CONFIG = "configs/ai_single_reference_ncc_development_v1.json"
ROUND = "ai_single_reference_ncc_development_v1"
OWNED = [CONFIG, "scripts/run_ai_single_reference_ncc_development.py",
         "tests/test_ai_single_reference_ncc_development.py"]


def ncc_conditioning(content, style, preprocess):
    unchanged, small = preprocess(np.asarray(content)[None], style, 512, True)
    if not np.array_equal(unchanged[0], np.asarray(content)):
        raise ValueError("NCC changed the original image")
    return small


def render_ncc(content, lut):
    if lut.shape != (64, 64, 3) or lut.dtype != np.float32 or not np.isfinite(lut).all():
        raise ValueError("NCC LUT representation mismatch")
    return content.filter(ImageFilter.Color3DLUT(16, np.clip(lut, 0, 1).flatten()))


def verify_assets(root: Path = ROOT) -> dict:
    cfg = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    original = baseline.verify_assets(root)
    if (cfg["round"] != ROUND or cfg["output"] != f"outputs/{ROUND}"
        or cfg["ncc"] is not True or cfg["training"] or cfg["automatic_promotion"]
        or cfg["arms"] != original["config"]["arms"] + ["learned_no_precorrection"]
        or cfg["baseline_config_sha256"] != original["config_sha256"]
        or cfg["cache"]["source_index"] != 2
        or any(value != original["config"]["budget"][key] for key, value in cfg["budget"].items())
        or set(cfg["budget"]) != {"maximum_new_forwards", "maximum_worker_wall_seconds",
                                 "maximum_cuda_allocated_bytes", "minimum_free_gpu_mib_before_loading",
                                 "maximum_existing_gpu_used_mib", "maximum_idle_gpu_utilization_percent",
                                 "minimum_output_free_bytes"}):
        raise ValueError("Fixed NCC contract changed")
    report_path = root / cfg["baseline_report"]
    report = checked_json(report_path, cfg["baseline_report_sha256"])
    lock = checked_json(report_path.parent / "execution_lock.json", cfg["baseline_lock_sha256"])
    if (report["config_sha256"] != original["config_sha256"]
        or report["config"] != original["config"]
        or lock["implementation_sha256"] != original["implementation_sha256"]
        or report["status"] != "COMPLETE_DEVELOPMENT_RENDER_PENDING_ALL_NINE_VISUAL_REVIEW"
        or report["new_forwards"] != 8 or report["cached_rows"] != 1
        or [row["source_index"] for row in report["rows"]] != list(range(9))):
        raise ValueError("Original batch execution identity mismatch")
    for row, source in zip(report["rows"], original["sources"], strict=True):
        if (row["source_sha256"] != source["sha256"]
            or row["reference_sha256"] != original["config"]["reference"]["sha256"]
            or set(row["arms"]) != set(original["config"]["arms"])):
            raise ValueError("Baseline pair identity mismatch")
        for arm, meta in row["arms"].items():
            path = report_path.parent / meta["path"]
            if path.name != f"{row['source_index']:02d}_{arm}.png" or sha(path) != meta["sha256"]:
                raise ValueError("Baseline PNG identity mismatch")
            image = read_rgb(path)
            if list(image.size) != source["size"] or meta["size"] != source["size"]:
                raise ValueError("Baseline PNG size mismatch")
    cache = cfg["cache"]
    cached_path = root / cache["report"]
    cached = checked_json(cached_path, cache["report_sha256"])
    if (cached["parent_report_sha256"] != original["config"]["cache"]["report_sha256"]
        or sha(root / cache["runner"]) != cache["runner_sha256"]
        or cached["stage"] != "rendered_pending_visual_review"
        or cached["rows"][2]["id"] != 2
        or cached["rows"][2]["lut_sha256"] != cache["lut_sha256"]
        or cached["rows"][2]["output_sha256"] != cache["output_sha256"]):
        raise ValueError("NCC cache provenance mismatch")
    lut_path, image_path = cached_path.parent / "02_lut.npy", cached_path.parent / "02_ncc.png"
    if sha(lut_path) != cache["lut_sha256"] or sha(image_path) != cache["output_sha256"]:
        raise ValueError("NCC cached bytes changed")
    preprocess = baseline.load_preprocess(root, original["parent"])
    reference = read_rgb(root / original["config"]["reference"]["path"])
    style = np.asarray(reference.resize((512, 512), Image.Resampling.BICUBIC))
    for source in original["sources"]:
        content = read_rgb(Path(source["path"]))
        ncc_conditioning(content, style, preprocess)
        if source["index"] == 2:
            payload = baseline.png_bytes(render_ncc(content, np.load(lut_path, allow_pickle=False)))
            if hashlib.sha256(payload).hexdigest() != cache["output_sha256"]:
                raise ValueError("NCC cached PNG reconstruction mismatch")
    return {"config": cfg, "config_sha256": sha(root / CONFIG),
            "implementation_sha256": {path: sha(root / path) for path in OWNED},
            "baseline_implementation_sha256": original["implementation_sha256"],
            "parent": original["parent"], "sources": original["sources"],
            "reference": original["config"]["reference"], "baseline_report": report,
            "baseline_directory": str(report_path.parent),
            "cache_lut": str(lut_path), "cache_image": str(image_path),
            "cached_ncc_png_reconstruction_exact": True, "baseline_pngs_verified": 36,
            "ncc_unchanged_inputs_verified": 9, "model_forwards": 0, "training_updates": 0}


def validate_attempts(base: Path, digest: str) -> None:
    if not base.exists():
        return
    if json.loads((base / "scope.json").read_text()) != {"round": ROUND, "config_sha256": digest}:
        raise ValueError("NCC output ownership mismatch")
    for path in base.iterdir():
        if path.name == "scope.json":
            continue
        if not path.is_dir() or not path.name.startswith("attempt-"):
            raise ValueError("Unknown NCC attempt artifact")
        report = json.loads((path / "report.json").read_text())
        if report["new_forwards"] != 0 or report["status"] not in {"DEFERRED_GPU_BUSY", "DEFERRED_RESOURCE_STATE_UNKNOWN"}:
            raise ValueError("NCC prior attempt forbids repetition")
        events = path / "events.jsonl"
        if events.exists() and any(json.loads(line).get("stage") == "FORWARD_START" for line in events.read_text().splitlines()):
            raise ValueError("NCC prior forward forbids repetition")


def preflight(root: Path = ROOT) -> dict:
    bundle = verify_assets(root)
    output_root = (root / "outputs").resolve()
    out = root / bundle["config"]["output"]
    if out.resolve().parent != output_root or out.resolve().drive.upper() != "P:":
        raise ValueError("Direct P-backed output required")
    validate_attempts(out, bundle["config_sha256"])
    if shutil.disk_usage(output_root).free < bundle["config"]["budget"]["minimum_output_free_bytes"]:
        raise ValueError("Insufficient output space")
    snapshot = baseline.gpu_snapshot()
    return {**bundle, "gpu": snapshot, "status": baseline.resource_admission(snapshot, bundle["config"]["budget"]),
            "gpu_model_loaded": False, "output_created": False}


def run_worker(lock_path: Path) -> int:
    lock = json.loads(lock_path.read_text())
    out = lock_path.parent
    started = time.monotonic()
    report = {"status": "STARTING", "config_sha256": lock["config_sha256"], "rows": [],
              "new_forwards": 0, "cached_rows": 0, "training_updates": 0,
              "independent_assessment": False, "product_promotion": False}
    torch = None
    try:
        bundle = verify_assets()
        if (any(bundle[key] != lock[key] for key in ("config_sha256", "implementation_sha256", "baseline_implementation_sha256"))
            or out.parent.resolve() != (ROOT / bundle["config"]["output"]).resolve()):
            raise ValueError("NCC execution-lock drift")
        cfg, budget = bundle["config"], bundle["config"]["budget"]
        report["config"] = cfg
        snapshot = baseline.gpu_snapshot()
        report["gpu_before_initialization"] = snapshot
        admission = baseline.resource_admission(snapshot, budget, own_pid=os.getpid())
        if admission != "READY_FOR_EXPLICIT_LOCAL_INFERENCE":
            report["status"] = admission
            return 0
        import torch as torch_module

        from scripts.run_ai_vcg_reference import initialize

        torch = torch_module
        torch.cuda.set_per_process_memory_fraction(budget["maximum_cuda_allocated_bytes"] / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        predict, preprocess = initialize(bundle["parent"])
        reference = read_rgb(ROOT / bundle["reference"]["path"])
        style = np.asarray(reference.resize((512, 512), Image.Resampling.BICUBIC))
        for row, old in zip(bundle["sources"], bundle["baseline_report"]["rows"], strict=True):
            index = row["index"]
            baseline.check_execution_budget(report["new_forwards"], time.monotonic() - started, torch.cuda.max_memory_allocated(), budget)
            content = read_rgb(Path(row["path"]))
            small = ncc_conditioning(content, style, preprocess)
            if index == 2:
                lut = np.load(bundle["cache_lut"], allow_pickle=False)
                report["cached_rows"] += 1
                origin = "HASH_VERIFIED_NCC_CACHE_NO_NEW_FORWARD"
            else:
                state = baseline.gpu_snapshot()
                if not state.get("available") or any(pid != os.getpid() for pid in state.get("known_compute_pids", [])):
                    raise RuntimeError("GPU ownership unknown or another compute process appeared")
                baseline.check_execution_budget(report["new_forwards"], time.monotonic() - started, torch.cuda.max_memory_allocated(), budget, before_new_forward=True)
                report["new_forwards"] += 1
                baseline.record_event(out, {"stage": "FORWARD_START", "source_index": index, "new_forward_number": report["new_forwards"]})
                lut = predict(small, style)
                torch.cuda.synchronize()
                origin = "NEW_NCC_CONDITIONED_INFERENCE"
            output = render_ncc(content, lut)
            payload = baseline.png_bytes(output)
            if index == 2 and hashlib.sha256(payload).hexdigest() != cfg["cache"]["output_sha256"]:
                raise ValueError("NCC cached publication changed")
            record = {"source_index": index, "source_sha256": row["sha256"], "reference_sha256": bundle["reference"]["sha256"],
                      "origin": origin, "arms": {}, "lut_minimum": float(lut.min()), "lut_maximum": float(lut.max()),
                      "lut_out_of_unit_range_fraction": float(np.mean((lut < 0) | (lut > 1)))}
            with (out / f"{index:02d}_lut.npy").open("xb") as handle:
                np.save(handle, lut, allow_pickle=False)
            record["lut_sha256"] = sha(out / f"{index:02d}_lut.npy")
            sheet = Image.new("RGB", (2000, 374), "white")
            for col, arm in enumerate(cfg["arms"]):
                if arm == "learned_no_precorrection":
                    image = output
                    data = payload
                else:
                    meta = old["arms"][arm]
                    data = (Path(bundle["baseline_directory"]) / meta["path"]).read_bytes()
                    if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                        raise ValueError("Baseline changed before copy")
                    image = read_rgb(Path(bundle["baseline_directory"]) / meta["path"])
                path = out / f"{index:02d}_{arm}.png"
                with path.open("xb") as handle:
                    handle.write(data)
                record["arms"][arm] = {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size": list(image.size)}
                thumb = image.copy()
                thumb.thumbnail((396, 310))
                sheet.paste(thumb, (col * 400, 25))
                ImageDraw.Draw(sheet).text((col * 400 + 4, 6), f"{index:02d} {arm}", fill="black")
            ImageDraw.Draw(sheet).text((4, 341), "Reference 18: Nick Rudzicz | CC-BY-4.0 | author-reported Portra 400, not calibrated | consumed development", fill="black")
            ImageDraw.Draw(sheet).text((4, 356), bundle["reference"]["source_page"], fill="black")
            with (out / f"{index:02d}_comparison.png").open("xb") as handle:
                sheet.save(handle, format="PNG")
            report["rows"].append(record)
            baseline.record_event(out, {"stage": "SOURCE_COMPLETE", "record": record})
            print("source_complete", index, origin, flush=True)
            baseline.check_execution_budget(report["new_forwards"], time.monotonic() - started, torch.cuda.max_memory_allocated(), budget)
        if len(report["rows"]) != 9 or report["new_forwards"] != 8 or report["cached_rows"] != 1:
            raise ValueError("Incomplete NCC batch")
        report["status"] = "COMPLETE_NCC_DEVELOPMENT_PENDING_ALL_NINE_VISUAL_REVIEW"
        return 0
    except Exception as error:  # noqa: BLE001 - Preserve failed external-model attempts; never retry them.
        report["status"] = "FAILED_PARTIAL_OUTPUTS_PRESERVED"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        return 1
    finally:
        report["worker_seconds"] = time.monotonic() - started
        if torch is not None and torch.cuda.is_initialized():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        with (out / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    try:
        result = preflight()
        if not args.run or result["status"] != "READY_FOR_EXPLICIT_LOCAL_INFERENCE":
            print(json.dumps(result, indent=2))
            return 0
        snapshot = baseline.gpu_snapshot()
        status = baseline.resource_admission(snapshot, result["config"]["budget"])
        if status != "READY_FOR_EXPLICIT_LOCAL_INFERENCE":
            print(json.dumps({"status": status, "gpu": snapshot, "output_created": False}))
            return 0
        base = ROOT / result["config"]["output"]
        validate_attempts(base, result["config_sha256"])
        if not base.exists():
            base.mkdir()
            with (base / "scope.json").open("x") as handle:
                json.dump({"round": ROUND, "config_sha256": result["config_sha256"]}, handle)
        out = base / f"attempt-{len(list(base.glob('attempt-*'))) + 1:04d}"
        out.mkdir()
        lock_path = out / "execution_lock.json"
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump({**result, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}, handle, indent=2)
        outcome = baseline.supervise_worker([sys.executable, str(Path(__file__).resolve()), "--worker", str(lock_path)], out, result["config"]["budget"]["maximum_worker_wall_seconds"])
        with (out / "supervisor.json").open("x") as handle:
            json.dump(outcome, handle, indent=2)
        print(json.dumps(outcome, indent=2))
        return 0 if outcome.get("exit_code") == 0 else 1
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(json.dumps({"status": "BLOCKED_PREFLIGHT", "error": type(error).__name__, "detail": str(error), "gpu_model_loaded": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
