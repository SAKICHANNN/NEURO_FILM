from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
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
from scripts.run_ai_vcg_reference import checked_json, read_rgb, sha

CONFIG = "configs/ai_single_reference_batch_development_v1.json"
OWNED = [CONFIG, "scripts/run_ai_single_reference_batch_development.py",
         "tests/test_ai_single_reference_batch_development.py"]


def png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def simple_image(image: Image.Image, config: dict) -> Image.Image:
    x = np.asarray(image, dtype=np.float32) / np.float32(255)
    pivot, slope = np.float32(config["pivot"]), np.float32(config["slope"])
    y = np.clip((x - pivot) * slope + pivot, 0, 1)
    return Image.fromarray(np.rint(y * np.float32(255)).astype(np.uint8))


def validate_contract(cfg: dict, parent: dict) -> None:
    if (
        [row["index"] for row in cfg["sources"]] != list(range(9))
        or cfg["reference"]["index"] != 13
        or Path(cfg["reference"]["path"]).name != "18.jpg"
        or cfg["cache"]["source_index"] != 2
        or cfg["cache"]["report_row"] != 2
        or cfg["arms"] != ["identity", "simple", "precorrection", "learned"]
        or cfg["inference"] != {"seed": 48, "steps": 25, "size": 512,
                               "ncc": False, "lut_size": 16, "node_clip": [0.0, 1.0]}
        or cfg["simple_control"]["slope"] != 1.15
        or cfg["simple_control"]["pivot"] != 0.5
        or any(cfg[key] for key in ("training", "downloads", "cloud", "automatic_promotion"))
    ):
        raise ValueError("Frozen single-reference development contract changed")
    if any(parent[key] != cfg["inference"][key] for key in ("seed", "steps", "size")):
        raise ValueError("Parent inference parameter mismatch")
    budget = cfg["budget"]
    if (
        budget["maximum_new_forwards"] != 8
        or budget["maximum_worker_wall_seconds"] != 600
        or budget["maximum_cuda_allocated_bytes"] != 9126805504
        or budget["minimum_free_gpu_mib_before_loading"] < 9216
        or not 0 <= budget["maximum_existing_gpu_used_mib"] <= 2560
        or not 0 <= budget["maximum_idle_gpu_utilization_percent"] <= 10
    ):
        raise ValueError("Frozen resource budget changed")


def load_preprocess(root: Path, parent: dict):
    path = root / "data/ai_models/video_color_grading_v1/source/utils/util.py"
    if sha(path) != parent["source_hashes"]["utils/util.py"]:
        raise ValueError("Preprocessing source drift")
    nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(node, ast.FunctionDef)
             and node.name in {"vars", "transfer", "preprocess"}]
    if len(nodes) != 3:
        raise ValueError("Preprocessing function mismatch")
    scope = {"np": np, "Image": Image}
    exec(  # noqa: S102 - Only the three reviewed hash-bound official functions.
        compile(ast.Module(nodes, []), "hash_bound_vcg_preprocess", "exec"), scope
    )
    return scope["preprocess"]


def verify_cache(root: Path, cfg: dict, parent: dict, rows: list, preprocess) -> dict:
    cache = cfg["cache"]
    report_path = root / cache["report"]
    original = checked_json(report_path, cache["report_sha256"])
    if original["config"] != parent:
        raise ValueError("Cached inference configuration mismatch")
    row = original["rows"][cache["report_row"]]
    pair = row["pair"]
    source = rows[cache["source_index"]]
    if (
        pair["content_index"] != source["index"]
        or Path(pair["content_path"]).resolve() != Path(source["path"]).resolve()
        or pair["content_sha256"] != source["sha256"]
        or pair["reference_index"] != cfg["reference"]["index"]
        or Path(pair["reference_path"]).resolve() != (root / cfg["reference"]["path"]).resolve()
        or pair["reference_sha256"] != cfg["reference"]["sha256"]
    ):
        raise ValueError("Cached pair identity mismatch")
    lut_path = root / cache["lut"]
    if sha(lut_path) != cache["lut_sha256"]:
        raise ValueError("Cached LUT identity mismatch")
    lut = np.load(lut_path, allow_pickle=False)
    if lut.shape != (64, 64, 3) or lut.dtype != np.float32 or not np.isfinite(lut).all():
        raise ValueError("Cached LUT representation mismatch")
    content = read_rgb(Path(source["path"]))
    reference = read_rgb(root / cfg["reference"]["path"])
    style = np.asarray(reference.resize((512, 512), Image.Resampling.BICUBIC))
    corrected, _ = preprocess(np.asarray(content)[None], style, 512, False)
    correction = Image.fromarray(corrected[0])
    images = {
        "identity": content,
        "simple": simple_image(content, cfg["simple_control"]),
        "precorrection": correction,
        "learned": correction.filter(ImageFilter.Color3DLUT(16, np.clip(lut, 0, 1).flatten())),
    }
    validated = {}
    for arm, image in images.items():
        path = report_path.parent / f"{cache['report_row']:02d}_{arm}.png"
        expected = row["images"][arm]
        if sha(path) != expected or hashlib.sha256(png_bytes(image)).hexdigest() != expected:
            raise ValueError(f"Cached {arm} PNG does not reconstruct exactly")
        with Image.open(path) as old:
            if old.mode != "RGB" or old.size != content.size or image.size != content.size:
                raise ValueError(f"Cached {arm} spatial/colour representation mismatch")
        validated[arm] = {"path": str(path), "sha256": expected, "size": list(image.size)}
    return {"source_index": source["index"], "all_four_png_reconstructions_exact": True,
            "images": validated, "lut_sha256": cache["lut_sha256"], "new_forward_required": False}


def verify_assets(root: Path = ROOT) -> dict:
    cfg = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    parent = checked_json(root / cfg["parent_config"], cfg["parent_config_sha256"])
    validate_contract(cfg, parent)
    if sha(root / cfg["parent_driver"]) != cfg["parent_driver_sha256"]:
        raise ValueError("Parent driver drift")
    ref = cfg["reference"]
    manifest = checked_json(root / ref["manifest"], ref["manifest_sha256"])
    reference_row = manifest["rows"][ref["index"]]
    if (manifest["role"] != "development_style_reference_only"
        or manifest["license"] != ref["rights"]
        or manifest["author"] != ref["author"]
        or manifest["page"] != ref["source_page"]
        or reference_row["path"] != Path(ref["path"]).name
        or reference_row["sha256"] != ref["sha256"]
        or sha(root / ref["path"]) != ref["sha256"]
        or sha(root / ref["page_snapshot"]) != ref["page_snapshot_sha256"]
        or ref["author_reported_capture"] not in (root / ref["page_snapshot"]).read_text(encoding="utf-8")):
        raise ValueError("Reference identity, role, rights or author statement mismatch")
    sources = checked_json(root / cfg["source_manifest"], cfg["source_manifest_sha256"])
    source_parent = (root / cfg["source_manifest"]).parent / "inputs"
    rows = []
    for expected in cfg["sources"]:
        source = sources[expected["index"]]
        path = Path(source["before"])
        if (source["license"] != "CC0/Public Domain"
            or path.resolve().parent != source_parent.resolve()
            or path.stat().st_size != expected["bytes"]
            or sha(path) != expected["sha256"]):
            raise ValueError("Development input bytes, location or rights mismatch")
        with Image.open(path) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.width != 1600 or image.info.get("icc_profile"):
                raise ValueError("Expected existing unprofiled RGB1600 development JPEG")
            shape = list(image.size)
        rows.append({**expected, "path": str(path), "size": shape,
                     "role": "already_consumed_development", "raw_sha256": source["sha256"]})
    model_root = root / "data/ai_models/video_color_grading_v1"
    checks = [(model_root / "source" / name, expected)
              for name, expected in parent["source_hashes"].items()]
    model_manifest = checked_json(model_root / "manifest.json", parent["model_manifest_sha256"])
    checks += [(model_root / row["file"], row["sha256"]) for row in model_manifest["models"]]
    checks += [(model_root / "clip_vit_b32/pytorch_model.bin", parent["clip_sha256"])]
    base = Path.home() / ".cache/huggingface/hub" / parent["sd15_cache_repo"] / "snapshots" / parent["sd15_revision"]
    checks += [(base / name, expected) for name, expected in parent["sd15_hashes"].items()]
    checks += [(root / name, expected) for name, expected in cfg["additional_local_asset_hashes"].items()]
    for path, expected in checks:
        if sha(path) != expected:
            raise ValueError(f"Local model/source asset drift: {path}")
    cache = verify_cache(root, cfg, parent, rows, load_preprocess(root, parent))
    return {"config": cfg, "parent": parent, "sources": rows, "cache": cache,
            "model_and_source_files_hash_verified": len(checks),
            "config_sha256": sha(root / CONFIG),
            "implementation_sha256": {path: sha(root / path) for path in OWNED},
            "model_forwards": 0, "training_updates": 0}


def gpu_snapshot() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True
        )
        rows = list(csv.reader(io.StringIO(result.stdout.strip())))
        if len(rows) != 1 or len(rows[0]) != 7:
            raise ValueError("Expected one local GPU with complete counters")
        index, uuid, name, total, used, free, utilization = [v.strip() for v in rows[0]]
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True
        )
        observed = []
        for row in csv.reader(io.StringIO(apps.stdout.strip())):
            if not row or row[0].strip() != uuid:
                continue
            if len(row) != 4:
                raise ValueError("Unexpected GPU process record")
            name_lower = row[2].strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
            observed.append({"pid": int(row[1].strip()), "process_name": row[2].strip(),
                             "reported_used_memory": row[3].strip(),
                             "known_compute_process": name_lower in {"python", "python.exe", "pythonw.exe", "python3", "python3.exe"}})
        snapshot = {"available": True, "index": int(index), "uuid": uuid, "name": name,
                    "total_mib": int(total), "used_mib": int(used), "free_mib": int(free),
                    "utilization_percent": int(utilization), "process_observations": observed,
                    "known_compute_pids": [row["pid"] for row in observed if row["known_compute_process"]],
                    "process_list_alone_establishes_active_gpu_use": False}
        if any(snapshot[key] < 0 for key in ("total_mib", "used_mib", "free_mib")) or not 0 <= snapshot["utilization_percent"] <= 100:
            raise ValueError("Invalid GPU counters")
        return snapshot
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"available": False, "error": type(error).__name__, "detail": str(error)}


def resource_admission(snapshot: dict, budget: dict, *, own_pid: int | None = None) -> str:
    if not snapshot.get("available"):
        return "DEFERRED_RESOURCE_STATE_UNKNOWN"
    foreign_pids = [pid for pid in snapshot["known_compute_pids"] if pid != own_pid]
    if (foreign_pids or snapshot["free_mib"] < budget["minimum_free_gpu_mib_before_loading"]
        or snapshot["used_mib"] > budget["maximum_existing_gpu_used_mib"]
        or snapshot["utilization_percent"] > budget["maximum_idle_gpu_utilization_percent"]):
        return "DEFERRED_GPU_BUSY"
    return "READY_FOR_EXPLICIT_LOCAL_INFERENCE"


def preflight(root: Path = ROOT) -> dict:
    started = time.monotonic()
    bundle = verify_assets(root)
    cfg = bundle["config"]
    out = root / cfg["output"]
    output_root = (root / "outputs").resolve()
    if out.resolve().parent != output_root or out.resolve().drive.upper() != "P:":
        raise ValueError("Direct P-backed output child required")
    validate_attempt_history(out, bundle["config_sha256"])
    if shutil.disk_usage(output_root).free < cfg["budget"]["minimum_output_free_bytes"]:
        raise ValueError("Insufficient output disk space")
    snapshot = gpu_snapshot()
    return {**bundle, "gpu": snapshot, "status": resource_admission(snapshot, cfg["budget"]),
            "preflight_seconds": time.monotonic() - started, "output_created": False,
            "gpu_model_loaded": False, "independent_assessment": False}


def validate_attempt_history(base: Path, config_sha256: str) -> None:
    if not base.exists():
        return
    scope = json.loads((base / "scope.json").read_text(encoding="utf-8"))
    if scope != {"config_sha256": config_sha256, "round": "ai_single_reference_batch_development_v1"}:
        raise ValueError("Output round ownership mismatch")
    for path in base.iterdir():
        if path.name == "scope.json":
            continue
        if not path.is_dir() or not path.name.startswith("attempt-"):
            raise ValueError("Unknown file in output round")
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        if report["status"] not in {"DEFERRED_GPU_BUSY", "DEFERRED_RESOURCE_STATE_UNKNOWN"} or report["new_forwards"] != 0:
            raise ValueError("Prior attempt ran or failed; no automatic repetition")
        event_path = path / "events.jsonl"
        if event_path.exists() and any(json.loads(line).get("stage") == "FORWARD_START" for line in event_path.read_text(encoding="utf-8").splitlines()):
            raise ValueError("Prior forward event forbids automatic repetition")


def create_attempt(base: Path, config_sha256: str) -> Path:
    validate_attempt_history(base, config_sha256)
    if not base.exists():
        base.mkdir()
        with (base / "scope.json").open("x", encoding="utf-8") as handle:
            json.dump({"config_sha256": config_sha256, "round": "ai_single_reference_batch_development_v1"}, handle)
    out = base / f"attempt-{len(list(base.glob('attempt-*'))) + 1:04d}"
    out.mkdir()
    return out


def check_execution_budget(forwards: int, seconds: float, peak_bytes: int, budget: dict,
                           *, before_new_forward: bool = False) -> None:
    if seconds >= budget["maximum_worker_wall_seconds"]:
        raise TimeoutError("Worker wall-time budget reached")
    if peak_bytes > budget["maximum_cuda_allocated_bytes"]:
        raise MemoryError("CUDA allocated peak exceeded budget")
    if forwards > budget["maximum_new_forwards"] or (
        before_new_forward and forwards >= budget["maximum_new_forwards"]
    ):
        raise RuntimeError("New-forward budget exhausted; no ninth forward allowed")


def record_event(out: Path, event: dict) -> None:
    with (out / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")
        handle.flush()


def run_worker(lock_path: Path) -> int:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    out = lock_path.parent
    started = time.monotonic()
    report = {"status": "STARTING", "config_sha256": lock["config_sha256"],
              "rows": [], "new_forwards": 0, "training_updates": 0,
              "product_promotion": False, "visual_review": "PENDING_ALL_NINE_RESULTS",
              "cached_rows": 0, "independent_assessment": False}
    torch = None
    try:
        bundle = verify_assets()
        if (bundle["config_sha256"] != lock["config_sha256"]
            or bundle["implementation_sha256"] != lock["implementation_sha256"]
            or out.parent.resolve() != (ROOT / bundle["config"]["output"]).resolve()):
            raise ValueError("Execution-lock drift")
        cfg, parent = bundle["config"], bundle["parent"]
        report["config"] = cfg
        report["sources"] = bundle["sources"]
        budget = cfg["budget"]
        snapshot = gpu_snapshot()
        report["gpu_before_initialization"] = snapshot
        admission = resource_admission(snapshot, budget, own_pid=os.getpid())
        if admission != "READY_FOR_EXPLICIT_LOCAL_INFERENCE":
            report["status"] = admission
            return 0
        import torch as torch_module

        from scripts.run_ai_vcg_reference import initialize

        torch = torch_module
        torch.cuda.set_per_process_memory_fraction(
            budget["maximum_cuda_allocated_bytes"] / torch.cuda.get_device_properties(0).total_memory,
            device=0,
        )
        torch.cuda.reset_peak_memory_stats()
        check_execution_budget(0, time.monotonic() - started, 0, budget)
        report["status"] = "LOADING_EXISTING_MODELS"
        record_event(out, {"stage": report["status"], "seconds": time.monotonic() - started})
        predict, preprocess = initialize(parent)
        reference = read_rgb(ROOT / cfg["reference"]["path"])
        style = np.asarray(reference.resize((512, 512), Image.Resampling.BICUBIC))
        for row in bundle["sources"]:
            index = row["index"]
            peak = torch.cuda.max_memory_allocated()
            check_execution_budget(report["new_forwards"], time.monotonic() - started, peak, budget)
            report["status"] = f"RENDERING_SOURCE_{index}"
            record_event(out, {"stage": report["status"], "new_forwards": report["new_forwards"]})
            content = read_rgb(Path(row["path"]))
            if index == cfg["cache"]["source_index"]:
                arms = {arm: read_rgb(Path(meta["path"])) for arm, meta in bundle["cache"]["images"].items()}
                lut = np.load(ROOT / cfg["cache"]["lut"], allow_pickle=False)
                origin = "HASH_VERIFIED_EXISTING_OUTPUT_NO_NEW_FORWARD"
                report["cached_rows"] += 1
            else:
                state = gpu_snapshot()
                if not state.get("available") or any(pid != os.getpid() for pid in state.get("known_compute_pids", [])):
                    raise RuntimeError("GPU ownership became unknown or another compute process appeared")
                corrected, small = preprocess(np.asarray(content)[None], style, 512, False)
                correction = Image.fromarray(corrected[0])
                check_execution_budget(report["new_forwards"], time.monotonic() - started,
                                       torch.cuda.max_memory_allocated(), budget, before_new_forward=True)
                report["new_forwards"] += 1
                record_event(out, {"stage": "FORWARD_START", "source_index": index,
                                   "new_forward_number": report["new_forwards"]})
                lut = predict(small, style)
                torch.cuda.synchronize()
                if lut.shape != (64, 64, 3) or lut.dtype != np.float32 or not np.isfinite(lut).all():
                    raise ValueError("Predicted LUT representation mismatch")
                arms = {"identity": content, "simple": simple_image(content, cfg["simple_control"]),
                        "precorrection": correction,
                        "learned": correction.filter(ImageFilter.Color3DLUT(16, np.clip(lut, 0, 1).flatten()))}
                origin = "NEW_PRETRAINED_MODEL_INFERENCE"
            record = {"source_index": index, "origin": origin, "source_sha256": row["sha256"],
                      "reference_sha256": cfg["reference"]["sha256"], "arms": {},
                      "lut_minimum": float(lut.min()), "lut_maximum": float(lut.max()),
                      "lut_out_of_unit_range_fraction": float(np.mean((lut < 0) | (lut > 1)))}
            with (out / f"{index:02d}_lut.npy").open("xb") as handle:
                np.save(handle, lut, allow_pickle=False)
            record["lut_sha256"] = sha(out / f"{index:02d}_lut.npy")
            sheet = Image.new("RGB", (1600, 374), "white")
            for col, arm in enumerate(cfg["arms"]):
                image = arms[arm]
                if image.mode != "RGB" or image.size != content.size:
                    raise ValueError("Output colour mode or spatial dimensions changed")
                payload = png_bytes(image)
                expected = bundle["cache"]["images"].get(arm) if index == cfg["cache"]["source_index"] else None
                if expected and hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError("Cached output changed during publication")
                path = out / f"{index:02d}_{arm}.png"
                with path.open("xb") as handle:
                    handle.write(payload)
                record["arms"][arm] = {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                                       "size": list(image.size)}
                thumb = image.copy()
                thumb.thumbnail((396, 310))
                sheet.paste(thumb, (col * 400, 25))
                ImageDraw.Draw(sheet).text((col * 400 + 4, 6), f"{index:02d} {arm}", fill="black")
            ImageDraw.Draw(sheet).text((4, 341), f"Reference 18: {cfg['reference']['author']} | CC-BY-4.0 | author-reported Portra 400 | development choice", fill="black")
            ImageDraw.Draw(sheet).text((4, 356), cfg["reference"]["source_page"], fill="black")
            with (out / f"{index:02d}_comparison.png").open("xb") as handle:
                sheet.save(handle, format="PNG")
            report["rows"].append(record)
            record_event(out, {"stage": "SOURCE_COMPLETE", "record": record})
            print("source_complete", index, origin, flush=True)
            check_execution_budget(report["new_forwards"], time.monotonic() - started,
                                   torch.cuda.max_memory_allocated(), budget)
        if len(report["rows"]) != 9 or report["new_forwards"] != 8 or report["cached_rows"] != 1:
            raise ValueError("Incomplete nine-source execution")
        report["status"] = "COMPLETE_DEVELOPMENT_RENDER_PENDING_ALL_NINE_VISUAL_REVIEW"
        return 0
    except Exception as error:  # noqa: BLE001 - Preserve inference failures and return a nonzero worker exit.
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
            handle.write("\n")


def supervise_worker(command: list[str], out: Path, maximum_seconds: float) -> dict:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.monotonic()
    with (out / "worker_stdout.txt").open("xb") as stdout, (out / "worker_stderr.txt").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                   creationflags=flags)
        try:
            code = process.wait(timeout=maximum_seconds)
            return {"status": "WORKER_EXITED", "worker_pid": process.pid, "exit_code": code,
                    "supervisor_seconds": time.monotonic() - started}
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            return {"status": "OWNED_WORKER_TERMINATED_AT_WALL_BUDGET", "worker_pid": process.pid,
                    "supervisor_seconds": time.monotonic() - started,
                    "partial_outputs_preserved": True, "foreign_processes_modified": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true", help="CPU checks only; the default")
    group.add_argument("--run", action="store_true", help="Explicitly request the bounded local inference worker")
    group.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args.worker)
    try:
        result = preflight()
        if not args.run or result["status"] != "READY_FOR_EXPLICIT_LOCAL_INFERENCE":
            print(json.dumps(result, indent=2))
            return 0
        base = ROOT / result["config"]["output"]
        # A second snapshot narrows the race after CPU hashing; no reservation is implied.
        snapshot = gpu_snapshot()
        status = resource_admission(snapshot, result["config"]["budget"])
        if status != "READY_FOR_EXPLICIT_LOCAL_INFERENCE":
            print(json.dumps({"status": status, "gpu": snapshot, "output_created": False}))
            return 0
        out = create_attempt(base, result["config_sha256"])
        lock = {**result, "pid": os.getpid(), "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        lock_path = out / "execution_lock.json"
        with lock_path.open("x", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2)
        outcome = supervise_worker([sys.executable, str(Path(__file__).resolve()), "--worker", str(lock_path)],
                                   out, result["config"]["budget"]["maximum_worker_wall_seconds"])
        with (out / "supervisor.json").open("x", encoding="utf-8") as handle:
            json.dump(outcome, handle, indent=2)
        print(json.dumps(outcome, indent=2))
        return 0 if outcome.get("exit_code") == 0 else 1
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(json.dumps({"status": "BLOCKED_PREFLIGHT", "error": type(error).__name__,
                          "detail": str(error), "gpu_model_loaded": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
