import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import imagecodecs
import numpy as np
import psutil
import tifffile
from PIL import Image, ImageCms
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/tst_crossed_curation_v1.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confined(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative:
        raise ValueError("UNSAFE_RELATIVE_PATH")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("PATH_OUTSIDE_ROOT")
    return target


def manifest(config, root=ROOT):
    for relative, expected in config["pins"].items():
        if digest(root / relative) != expected:
            raise ValueError("PIN_MISMATCH:" + relative)
    acquisition = read(root / config["acquisition_config"])
    sources = read(root / config["source_queue"])
    paths = read(root / config["path_manifest"])
    allowed = {r["path"] for r in paths["head_now"]}
    forbidden = {r["path"] for k in ("original_Y_only_deferred", "real_reserved_no_network") for r in paths[k]}
    files = {r["path"]: r for r in acquisition["files"]}
    if len(files) != config["expected_files"] or files.keys() != allowed or allowed & forbidden:
        raise ValueError("ACQUISITION_ALLOWLIST_MISMATCH")
    if len(sources) != config["expected_sources"] or len({r["path"] for r in sources}) != len(sources):
        raise ValueError("SOURCE_QUEUE_COUNT")
    if not {r["path"] for r in sources} <= allowed:
        raise ValueError("SOURCE_OUTSIDE_ALLOWLIST")
    if any(not all("query_X" in role or "donor_P" in role for role in r["roles"]) for r in sources):
        raise ValueError("SOURCE_ROLE_NOT_X_OR_P")
    for relative in allowed:
        confined(root / config["originals"], relative)
    return acquisition, files, sources


def bind_receipt(config, acquisition, files, root=ROOT):
    receipt_path = root / config["receipt"]
    receipt = read(receipt_path)
    if receipt.get("status") != "ACQUIRED_HASH_VERIFIED_NOT_DECODED":
        raise ValueError("RECEIPT_NOT_COMPLETE")
    if receipt.get("errors") or receipt.get("unattempted") or receipt.get("stop_reasons"):
        raise ValueError("RECEIPT_HAS_FAILURES")
    if receipt["config_sha256"] != digest(root / config["acquisition_config"]):
        raise ValueError("RECEIPT_CONFIG_MISMATCH")
    rows = receipt["files"]
    if len(rows) != len(files) or {r["path"] for r in rows} != files.keys():
        raise ValueError("RECEIPT_FILE_SET_MISMATCH")
    for row in rows:
        expected = files[row["path"]]
        if any(row[k] != expected[k] for k in ("bytes", "sha256")):
            raise ValueError("RECEIPT_FILE_IDENTITY_MISMATCH")
    if receipt["total_verified_bytes"] != acquisition["expected_bytes"]:
        raise ValueError("RECEIPT_TOTAL_BYTES_MISMATCH")
    return {"receipt_sha256": digest(receipt_path), "download_accounting": receipt["accounting"]}


def png_metadata(path):
    chunks = {}
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("PNG_SIGNATURE")
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("PNG_TRUNCATED_CHUNK")
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            if kind in (b"IHDR", b"gAMA", b"cHRM", b"cICP", b"acTL", b"sRGB"):
                if length > 64:
                    raise ValueError("PNG_METADATA_LENGTH")
                chunks[kind.decode()] = stream.read(length).hex()
            else:
                stream.seek(length, 1)
            if len(stream.read(4)) != 4:
                raise ValueError("PNG_TRUNCATED_CRC")
            if kind == b"IEND":
                break
    return chunks


def inspect_header(path, config):
    with Image.open(path) as image:
        profile = image.info.get("icc_profile")
        result = {"format": image.format, "mode": image.mode, "width": image.width,
                  "height": image.height, "orientation": image.getexif().get(274, 1),
                  "frames": getattr(image, "n_frames", 1),
                  "icc_sha256": hashlib.sha256(profile).hexdigest() if profile else None}
    reasons = []
    if result["format"] not in ("TIFF", "PNG", "JPEG"):
        reasons.append("UNSUPPORTED_FORMAT")
    if result["mode"] != "RGB":
        reasons.append("STRICT_RGB_REQUIRED")
    if result["orientation"] not in (1, None):
        reasons.append("NONIDENTITY_ORIENTATION")
    if result["frames"] != 1:
        reasons.append("MULTIFRAME")
    if result["format"] == "TIFF":
        with tifffile.TiffFile(path) as image:
            page = image.pages[0]
            result.update({"bits": page.bitspersample, "pages": len(image.pages),
                           "native_dtype": str(page.dtype), "native_shape": list(page.shape)})
            if len(image.pages) != 1 or page.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
                reasons.append("TIFF_PAGES_OR_DTYPE")
            if len(page.shape) != 3 or page.shape[-1] != 3:
                reasons.append("TIFF_NATIVE_NOT_INTERLEAVED_RGB")
    if result["format"] == "PNG":
        chunks = png_metadata(path)
        result["png_chunks"] = chunks
        ihdr = bytes.fromhex(chunks["IHDR"])
        result["bits"] = ihdr[8]
        if ihdr[8] not in (8, 16) or ihdr[9] != 2:
            reasons.append("PNG_NOT_RGB8_OR_RGB16")
        if "acTL" in chunks or "cICP" in chunks:
            reasons.append("PNG_ANIMATION_OR_CICP_REQUIRES_SEPARATE_POLICY")
        if not profile:
            if "gAMA" in chunks and int(chunks["gAMA"], 16) != 45455:
                reasons.append("UNTAGGED_NON_SRGB_GAMMA")
            if "cHRM" in chunks:
                chromaticities = bytes.fromhex(chunks["cHRM"])
                expected = (31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
                actual = tuple(int.from_bytes(chromaticities[i:i+4], "big") for i in range(0, 32, 4))
                if actual != expected:
                    reasons.append("UNTAGGED_NON_SRGB_CHROMATICITIES")
    estimate = result["width"] * result["height"] * 32 + config["decode_base_bytes"]
    result["conservative_decode_estimate_bytes"] = estimate
    if estimate > config["rss_bytes"]:
        reasons.append("RESOURCE_DEFERRED")
    result.update({"eligible_decode": not reasons, "reasons": reasons,
                   "status": "HEADER_ELIGIBLE" if not reasons else "HEADER_EXCLUDED_OR_DEFERRED"})
    return result


def decode(path, metadata):
    if not metadata["eligible_decode"]:
        raise ValueError("HEADER_NOT_ELIGIBLE")
    with Image.open(path) as image:
        profile = image.info.get("icc_profile")
        if metadata["format"] == "JPEG":
            values = np.asarray(image).copy()
    if metadata["format"] == "PNG":
        values = imagecodecs.png_decode(path.read_bytes())
    elif metadata["format"] == "TIFF":
        values = tifffile.imread(path)
    if values.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or values.shape != (metadata["height"], metadata["width"], 3):
        raise ValueError("DECODE_CONTRACT_MISMATCH")
    if profile:
        srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        converted = imagecodecs.cms_transform(values, profile, srgb, colorspace="RGB", outcolorspace="RGB",
                                              outdtype=np.uint16, intent=1)
        policy = "Embedded ICC to sRGB; LittleCMS relative colorimetric uint16; no BPC"
    else:
        converted = values.astype(np.uint16) * 257 if values.dtype == np.uint8 else values.copy()
        policy = "ASSUMPTION: untagged encoded RGB is sRGB; colorimetry not verified"
    return converted, policy


def canonical_hash(values):
    header = json.dumps({"shape": list(values.shape), "dtype": "uint16-le", "space": "encoded_sRGB"}, sort_keys=True).encode()
    hashed = hashlib.sha256(header)
    for start in range(0, len(values), 64):
        hashed.update(values[start:start+64].astype("<u2", copy=False).tobytes(order="C"))
    return hashed.hexdigest()


def compact(values):
    # Limit float conversion to one row band; preserve native conversion before resize.
    rgb8 = np.empty(values.shape, dtype=np.uint8)
    for start in range(0, len(values), 64):
        rgb8[start:start+64] = np.rint(values[start:start+64].astype(np.float32) / 257).astype(np.uint8)
    image = Image.fromarray(rgb8)
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    difference = np.asarray(gray)[:, 1:] > np.asarray(gray)[:, :-1]
    number = 0
    for bit in difference.ravel():
        number = (number << 1) | int(bit)
    gray64 = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.LANCZOS)).astype(np.float32)
    centered = gray64 - gray64.mean()
    norm = float(np.linalg.norm(centered))
    thumbnail = io.BytesIO()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(thumbnail, format="PNG", icc_profile=profile)
    return thumbnail.getvalue(), {"dhash64": f"{number:016x}", "gray_mean": float(gray64.mean()),
                                  "gray_centered_l2": norm, "aspect_ratio": values.shape[1] / values.shape[0],
                                  "thumbnail_claim": "Source-only candidate screen; not native geometry/content verification"}


def verify_input(path, row):
    if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError("ENCODED_INPUT_IDENTITY_MISMATCH")


def reusable(item, identity, row, directory):
    if item.get("identity") != identity or item.get("encoded_sha256") != row["sha256"]:
        raise ValueError("RESUME_IDENTITY_MISMATCH")
    for artifact in item.get("artifacts", []):
        if digest(confined(directory, artifact["path"])) != artifact["sha256"]:
            raise ValueError("RESUME_ARTIFACT_MISMATCH")
    return True


def duplicate_summary(records):
    grouped = {}
    for item in records:
        if item.get("canonical_pixel_sha256"):
            grouped.setdefault(item["canonical_pixel_sha256"], []).append(item)
    collisions, aliases = [], []
    for sha, group in grouped.items():
        if len(group) < 2:
            continue
        entry = {"sha256": sha, "paths": sorted(r["path"] for r in group), "splits": sorted({r["split"] for r in group})}
        (collisions if len(entry["splits"]) > 1 else aliases).append(entry)
    return {"cross_split_quarantine_components": collisions, "within_split_alias_components": aliases,
            "claim": "Exact canonical duplicate census only; near-duplicate grouping and physical capture independence pending"}


def worker(config_path, stage, attempt):
    config = read(config_path)
    threadpool_limits(config["threads"])
    acquisition, files, sources = manifest(config)
    receipt = bind_receipt(config, acquisition, files)
    output = ROOT / config["output"]
    identity = {"config_sha256": digest(config_path), "entry_sha256": digest(Path(__file__)), **receipt}
    if identity != read(attempt / "identity.json"):
        raise ValueError("WORKER_BINDING_CHANGED")
    queue = [{"path": key} for key in sorted(files)] if stage == "headers" else sources
    directory = output / stage
    records = []
    cached_bytes = 0
    for source in queue:
        row = files[source["path"]]
        key = hashlib.sha256(source["path"].encode()).hexdigest()
        item_path = directory / "items" / (key + ".json")
        path = confined(ROOT / config["originals"], source["path"])
        verify_input(path, row)
        if item_path.exists():
            item = read(item_path)
            reusable(item, identity, row, directory)
            records.append(item)
            cached_bytes += sum(a["bytes"] for a in item.get("artifacts", []))
            continue
        item = {**source, "encoded_sha256": row["sha256"], "identity": identity, "artifacts": []}
        try:
            if stage == "headers":
                item["header"] = inspect_header(path, config)
                item["status"] = item["header"]["status"]
            else:
                header_item = read(output / "headers/items" / (key + ".json"))
                reusable(header_item, identity, row, output / "headers")
                header = header_item.get("header")
                if not header or not header["eligible_decode"]:
                    item.update({"status": "SOURCE_HEADER_EXCLUDED_OR_DEFERRED", "header_record": str(key + ".json")})
                else:
                    values, policy = decode(path, header)
                    thumbnail, descriptor = compact(values)
                    if cached_bytes + len(thumbnail) > config["preview_cache_bytes"]:
                        raise MemoryError("COMPACT_CACHE_LIMIT")
                    destination = directory / "previews" / (key + ".png")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(thumbnail)
                    cached_bytes += len(thumbnail)
                    item.update({"status": "SOURCE_DECODED", "canonical_pixel_sha256": canonical_hash(values),
                                 "shape": list(values.shape), "color_policy": policy, "descriptor": descriptor,
                                 "artifacts": [{"path": "previews/" + key + ".png", "sha256": digest(destination), "bytes": len(thumbnail)}]})
                    del values
        except Exception as exc:  # noqa: BLE001 - per-image boundary outcomes must remain auditable
            item.update({"status": "ITEM_TECHNICAL_FAILURE", "failure": {"type": type(exc).__name__, "message": str(exc)}})
        save(item_path, item)
        records.append(item)
        save(attempt / "progress.json", {"processed": len(records), "expected": len(queue), "worker_cpu_seconds": time.process_time()})
    report = {"status": "COMPLETE_CENSUS_NOT_ADMISSION", "stage": stage, "identity": identity,
              "expected": len(queue), "processed": len(records), "counts": {},
              "worker_cpu_seconds": time.process_time(), "preview_cache_bytes": cached_bytes,
              "claim": "S0/S1 preparation only; no probe admission, near-duplicate clearance, donor fits or learning."}
    for item in records:
        report["counts"][item["status"]] = report["counts"].get(item["status"], 0) + 1
    if stage == "sources":
        report["duplicates"] = duplicate_summary(records)
        report["near_duplicate_comparison_count"] = len(records) * (len(records) - 1) // 2
        report["near_duplicate_status"] = "DESCRIPTORS_READY_COMPARISON_AND_REVIEW_NOT_RUN"
        report["probe_selection_status"] = "NOT_RUN_SOURCE_ONLY_FIRST64_PER_SPLIT_NEXT"
    save(attempt / "worker_report.json", report)


def preflight(config_path, stage):
    config = read(config_path)
    acquisition, files, _ = manifest(config)
    receipt_path = ROOT / config["receipt"]
    if not receipt_path.exists():
        return config, {"status": "WAIT_COMPLETED_RECEIPT", "photo_reads": 0}
    try:
        receipt = bind_receipt(config, acquisition, files)
    except ValueError as exc:
        return config, {"status": "WAIT_COMPLETED_RECEIPT", "reason": str(exc), "photo_reads": 0}
    identity = {"config_sha256": digest(config_path), "entry_sha256": digest(Path(__file__)), **receipt}
    if stage == "sources":
        header_report = ROOT / config["output"] / "headers/report.json"
        if not header_report.exists() or read(header_report).get("identity") != identity or read(header_report).get("status") != "COMPLETE_CENSUS_NOT_ADMISSION":
            return config, {"status": "WAIT_COMPLETE_HEADERS", "photo_reads": 0}
    return config, {"status": "READY", "identity": identity, "photo_reads": 0}


def sample_tree(owned, registry):
    try:
        processes = [owned, *owned.children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []
    for process in processes:
        try:
            created = process.create_time()
            key = (process.pid, created)
            entry = registry.setdefault(key, {"process": process, "cpu_seconds": 0., "peak_rss_bytes": 0})
            times = process.cpu_times()
            entry["cpu_seconds"] = max(entry["cpu_seconds"], times.user + times.system)
            entry["peak_rss_bytes"] = max(entry["peak_rss_bytes"], process.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
    return sum(r["cpu_seconds"] for r in registry.values()), sum(r["peak_rss_bytes"] for r in registry.values())


def stop_tree(registry):
    for (pid, created), entry in reversed(list(registry.items())):
        try:
            process = entry["process"]
            if process.pid == pid and process.create_time() == created and process.is_running():
                process.kill()
        except psutil.NoSuchProcess:
            pass


def launch(config_path, stage):
    config, checks = preflight(config_path, stage)
    if checks["status"] != "READY":
        raise ValueError(checks)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "running.lock"
    with lock_path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    started, cpu_started = time.monotonic(), time.process_time()
    process = None
    attempt = None
    measured_cpu, peak, reason = 0., 0, None
    registry = {}
    try:
        previous = [read(p) for p in output.glob("*/attempts/*/accounting.json")]
        same_stage = [r for r in previous if r["stage"] == stage]
        spent_cpu = sum(r["aggregate_cpu_seconds"] for r in same_stage)
        spent_wall = sum(r["active_wall_seconds"] for r in same_stage)
        limits = config["stages"][stage]
        download = checks["identity"]["download_accounting"]
        all_wall = sum(r["active_wall_seconds"] for r in previous) + config["prior_reserve_wall_seconds"] + download["wall_seconds"]
        if spent_cpu >= limits["cpu_seconds"] or spent_wall >= limits["wall_seconds"] or all_wall >= config["study_wall_seconds"]:
            raise ValueError("CUMULATIVE_BUDGET_EXHAUSTED_NO_RESET")
        number = len(list((output / stage / "attempts").glob("*"))) + 1
        attempt = output / stage / "attempts" / f"{number:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        save(attempt / "identity.json", checks["identity"])
        with (attempt / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), "--config", str(config_path), "--stage", stage,
                                        "--worker", str(attempt)], stdout=log, stderr=subprocess.STDOUT)
            owned = psutil.Process(process.pid)
            while process.poll() is None:
                measured_cpu, peak = sample_tree(owned, registry)
                elapsed = time.monotonic() - started
                parent_cpu = time.process_time() - cpu_started
                if spent_cpu + measured_cpu + parent_cpu > limits["cpu_seconds"]:
                    reason = "CPU_LIMIT"
                elif spent_wall + elapsed > limits["wall_seconds"] or all_wall + elapsed > config["study_wall_seconds"]:
                    reason = "WALL_LIMIT"
                elif peak > config["rss_bytes"]:
                    reason = "RSS_LIMIT"
                if reason:
                    stop_tree(registry)
                    break
                time.sleep(.1)
            code = process.wait()
        progress_path = attempt / "progress.json"
        if progress_path.exists():
            measured_cpu = max(measured_cpu, read(progress_path)["worker_cpu_seconds"])
        report_path = attempt / "worker_report.json"
        if report_path.exists():
            measured_cpu = max(measured_cpu, read(report_path)["worker_cpu_seconds"])
        accounting = {"stage": stage, "exit_code": code, "failure": reason,
                      "aggregate_cpu_seconds": measured_cpu + time.process_time() - cpu_started + .2,
                      "active_wall_seconds": time.monotonic() - started, "peak_worker_rss_bytes": peak,
                      "prior_reserve_cpu_seconds": config["prior_reserve_cpu_seconds"],
                      "prior_reserve_wall_seconds": config["prior_reserve_wall_seconds"], "download_accounting": download,
                      "owned_processes": [{"pid": pid, "create_time": created, "cpu_seconds": row["cpu_seconds"]} for (pid, created), row in registry.items()],
                      "claim": "Includes failed/resumed attempts and conservative0.2CPU second allowance for final sampling interval; historical negative study ledger unchanged."}
        save(attempt / "accounting.json", accounting)
        if code == 0 and reason is None and report_path.exists():
            report = read(report_path)
            report["last_attempt_accounting"] = accounting
            save(output / stage / "report.json", report)
        else:
            save(attempt / "failure.json", {"status": "WORKER_INCOMPLETE", **accounting})
        return code if code else int(reason is not None)
    finally:
        stop_tree(registry)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if attempt is not None and not (attempt / "accounting.json").exists():
            save(attempt / "accounting.json", {"stage": stage, "failure": "INTERRUPTED_OR_LAUNCH_EXCEPTION",
                "aggregate_cpu_seconds": measured_cpu + time.process_time() - cpu_started + .2,
                "active_wall_seconds": time.monotonic() - started, "peak_worker_rss_bytes": peak})
        lock_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("headers", "sources"), default="headers")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.config, args.stage, args.worker)
        finally:
            progress_path = args.worker / "progress.json"
            progress = read(progress_path) if progress_path.exists() else {}
            progress["worker_cpu_seconds"] = time.process_time()
            save(progress_path, progress)
    elif args.run:
        raise SystemExit(launch(args.config, args.stage))
    else:
        print(json.dumps(preflight(args.config, args.stage)[1], indent=2))
