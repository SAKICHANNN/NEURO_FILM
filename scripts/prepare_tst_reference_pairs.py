import argparse
import hashlib
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
from PIL import Image, ImageCms, ImageDraw
from scipy.ndimage import sobel
from threadpoolctl import threadpool_limits

CONFIG = ROOT/"configs/tst_reference_pairs_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_hash(values):
    assert values.dtype == np.uint16 and values.ndim == 3 and values.shape[-1] == 3
    header = json.dumps({"shape": list(values.shape), "dtype": "uint16-le", "space": "encoded_sRGB"}, sort_keys=True).encode()
    return hashlib.sha256(header+values.astype("<u2", copy=False).tobytes(order="C")).hexdigest()


def geometry(left, right):
    if left.shape != right.shape:
        return {"status": "DIMENSION_MISMATCH_NO_PIXEL_PAIRING", "regions": []}
    h, w = left.shape[:2]
    gradients = []
    for image in (left, right):
        luma = image.astype(np.float32)@np.array([.2126, .7152, .0722], dtype=np.float32)/65535
        gradients.append(np.hypot(sobel(luma, axis=0), sobel(luma, axis=1)))
    regions = []
    for y0, y1, x0, x1 in ((0, h, 0, w), (0, h//2, 0, w//2), (0, h//2, w//2, w), (h//2, h, 0, w//2), (h//2, h, w//2, w)):
        y0, y1, x0, x1 = y0+4, y1-4, x0+4, x1-4
        if y1 <= y0 or x1 <= x0:
            regions.append({"best": None, "zero": None, "reason": "region_too_small"})
            continue
        source = gradients[0][y0:y1:4, x0:x1:4].astype(float).ravel()
        source -= source.mean()
        norm = np.linalg.norm(source)
        scores = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                target = gradients[1][y0+dy:y1+dy:4, x0+dx:x1+dx:4].astype(float).ravel()
                target -= target.mean()
                denominator = norm*np.linalg.norm(target)
                scores.append({"dy": dy, "dx": dx, "ncc": float(source@target/denominator) if denominator else None})
        valid = [s for s in scores if s["ncc"] is not None]
        best = max(valid, key=lambda s: (s["ncc"], -(abs(s["dx"])+abs(s["dy"])))) if valid else None
        regions.append({"best": best, "zero": scores[12]})
    return {"status": "SAME_NATIVE_DIMENSIONS_DIAGNOSTIC_ONLY", "regions": regions,
            "claim": "Low NCC/nonzero best shift is a review flag, not proof of wrong mapping. No subpixel proof or resampling."}


def cross_split_collisions(endpoints, field):
    grouped = {}
    for endpoint in endpoints:
        if endpoint.get(field):
            grouped.setdefault(endpoint[field], []).append(endpoint)
    return [{"hash": h, "endpoints": [{k: v for k, v in e.items() if k in ("path", "split", "role")} for e in entries]}
            for h, entries in grouped.items() if len({e["split"] for e in entries}) > 1]


def preflight():
    cfg = json.loads(CONFIG.read_text())
    assert digest(Path(__file__)) == cfg["entry_sha256"]
    for path, expected in cfg["pins"].items():
        assert digest(ROOT/path) == expected, path
    assert len(cfg["pairs"]) == 59 and len(cfg["existing_endpoints"]) == 160
    assert len({p["path"] for p in cfg["pairs"]}) == 59
    for endpoint in cfg["existing_endpoints"]:
        assert digest(ROOT/endpoint["original"]) == endpoint["encoded_sha256"]
        assert digest(ROOT/endpoint["decoded"]) == endpoint["decoded_file_sha256"]
    for pair in cfg["pairs"]:
        p = ROOT/pair["original"]
        assert p.stat().st_size == pair["bytes"] and digest(p) == pair["encoded_sha256"]
    legacy = load(ROOT/"scripts/run_tst_reference_check.py", "supervisor")
    return cfg, {"status": "READY", "config_sha256": digest(CONFIG), "remaining_cpu_seconds": {"learning": cfg["cpu_seconds"]}}, legacy


def worker(directory):
    cfg, checks, legacy = preflight()
    assert checks == json.loads((directory/"lock.json").read_text())
    threadpool_limits(2)
    legacy.save(directory/"report.json", {"status": "RUNNING", "checks": checks})
    decoder = load(ROOT/cfg["decoder"], "frozen_decoder")
    decoder.OUT = directory
    endpoints = []
    for endpoint in cfg["existing_endpoints"]:
        endpoints.append({**endpoint, "canonical_pixel_sha256": canonical_hash(tifffile.imread(ROOT/endpoint["decoded"]))})
    pairs = []
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    for record in cfg["pairs"]:
        key = f'{record["group"]:02d}_R{record["target"]}'
        row = {**record, "manual_same_content": "PENDING_VISUAL_REVIEW"}
        try:
            preview, metadata = decoder.display_decode(ROOT/record["original"], directory/"display_srgb16"/(key+"_P.tif"))
            left = tifffile.imread(directory/metadata["srgb16_path"])
            right = tifffile.imread(ROOT/record["reference_decoded"])
            row.update({"before_metadata": metadata, "before_shape": list(left.shape), "reference_shape": list(right.shape), "geometry": geometry(left, right)})
            endpoints.append({"path": record["path"], "split": record["split"], "role": "P", "encoded_sha256": record["encoded_sha256"], "canonical_pixel_sha256": canonical_hash(left)})
            board = Image.new("RGB", (1200, 650), (238, 238, 238))
            draw = ImageDraw.Draw(board)
            for i, image in enumerate((preview, Image.fromarray(np.rint(right.astype(np.float32)/257).astype(np.uint8)))):
                image.thumbnail((590, 590), Image.Resampling.LANCZOS)
                board.paste(image, (600*i+(600-image.width)//2, 35+(590-image.height)//2))
            draw.text((8, 8), key+" | "+record["split"]+" | LEFT mapped before P; RIGHT reference after R", fill="black")
            draw.text((8, 630), "Native shapes "+str(left.shape)+" / "+str(right.shape)+"; display thumbnails only, no registration", fill="black")
            destination = directory/"review"/record["split"]/(key+".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            board.save(destination, icc_profile=srgb)
            row["board"] = {"path": str(destination.relative_to(directory)), "sha256": digest(destination)}
        except Exception as exc:  # noqa: BLE001 - retain per-file boundary decode failures without replacing data
            row["decode_or_pair_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        pairs.append(row)
        legacy.save(directory/"partial_pairs.json", {"pairs": pairs, "review": "PENDING"})
        print(key, row.get("geometry", {}).get("status", "DECODE_FAILURE"), flush=True)
    collisions = {field: cross_split_collisions(endpoints, field) for field in ("encoded_sha256", "canonical_pixel_sha256")}
    legacy.save(directory/"endpoints.json", {"endpoints": endpoints, "cross_split_collisions": collisions})
    artifact = [{"path": str(p.relative_to(directory)), "sha256": digest(p)} for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in ("report.json", "worker.log")]
    legacy.save(directory/"report.json", {"status": "COMPLETE_PHASE", "checks": checks, "pairs": pairs,
        "pairs_expected": 59, "existing_endpoints": 160, "endpoints_audited": len(endpoints), "cross_split_collisions": collisions,
        "decode_failure_count": sum("decode_or_pair_failure" in r for r in pairs),
        "dimension_mismatch_count": sum(r.get("geometry", {}).get("status") == "DIMENSION_MISMATCH_NO_PIXEL_PAIRING" for r in pairs),
        "training_admission": "PENDING_ALL59_VISUAL_AND_PAIR_COMPATIBILITY_REVIEW", "new_feature_forwards": 0, "training_updates": 0,
        "precision": "59 originals unchanged; pinned ICC-aware uint16 decoder; untagged sRGB assumption recorded perfile. Canonical hashes compare exact decoded uint16 content, notperceptual/captureidentity.",
        "artifacts": artifact, "worker_pid": os.getpid(), "worker_cpu_seconds": time.process_time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="prepare", choices=["prepare"])
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        cfg, checks, legacy = preflight()
        if args.run:
            legacy.__file__ = __file__
            legacy.CONFIG = CONFIG
            legacy.category = lambda stage: "learning"
            code = legacy.launch("prepare", cfg, checks)
            p = ROOT/cfg["output"]/"prepare/report.json"
            report = json.loads(p.read_text())
            report["new_study_budget_ledger"] = {"limit_cpu_seconds": 1800, "processing_cpu_seconds": report["accounted_cpu_seconds"],
                "remaining_feature_training_scoring_cpu_seconds": max(0, 1800-report["accounted_cpu_seconds"]),
                "remaining_active_wall_seconds": max(0, 1800-report["supervisor"]["wall_seconds"]),
                "rule": "Future newfeature/training/render/scoring phases must subtractthiscostandinterveningprocessing;noreset."}
            legacy.save(p, report)
            raise SystemExit(code)
        print(json.dumps(checks, indent=2))
