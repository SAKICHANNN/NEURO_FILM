#!/usr/bin/env python3
"""Acquire exact REPID9 fit/calibration originals into a CLI-provided root."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acquire_u5_r2repid3_fit_calibration import _download, _file_sha256


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(
    *,
    config_path: Path,
    output_root: Path,
    report_path: Path,
    reverse: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent = config["parent"]
    manifest_path = ROOT / parent["manifest_path"]
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != parent["manifest_sha256"]:
        raise ValueError("REPID9A manifest drift")
    manifest = json.loads(manifest_bytes)
    evidence = json.loads((ROOT / parent["evidence_path"]).read_text(encoding="utf-8"))
    if evidence["decision"] != parent["required_decision"]:
        raise ValueError("REPID9A decision drift")
    if manifest["member_identity_sha256"] != parent["member_identity_sha256"]:
        raise ValueError("REPID9A member identity drift")
    members = list(manifest["members"])
    if reverse:
        members.reverse()
    tasks = []
    for row in members:
        local = output_root / str(row["role"]) / "original" / str(row["scene_id"])
        remote = str(row["path"])
        url = f"{config['acquisition']['resolve_base']}/{urllib.parse.quote(remote, safe='/')}"
        tasks.append((url, local, int(row["size"]), str(row["sha256"])))
    with ThreadPoolExecutor(max_workers=int(config["acquisition"]["maximum_workers"])) as pool:
        futures = [
            pool.submit(
                _download,
                url,
                local,
                size,
                sha,
                str(config["acquisition"]["temporary_suffix"]),
            )
            for url, local, size, sha in tasks
        ]
        for future in futures:
            future.result()
    records = []
    for row in manifest["members"]:
        local = output_root / str(row["role"]) / "original" / str(row["scene_id"])
        with Image.open(local) as image:
            image.load()
            if image.format != config["decode_preflight"]["required_container"]:
                raise ValueError("container drift")
            if image.mode != config["decode_preflight"]["required_mode"]:
                raise ValueError("mode drift")
            if min(image.size) < int(config["decode_preflight"]["minimum_dimension"]):
                raise ValueError("geometry below minimum")
            icc = image.info.get("icc_profile")
            record = {
                "scene_id": row["scene_id"],
                "role": row["role"],
                "logical_path": f"{row['role']}/original/{row['scene_id']}",
                "size": local.stat().st_size,
                "sha256": _file_sha256(local),
                "width": image.width,
                "height": image.height,
                "icc_sha256": _sha256(icc) if icc else None,
            }
        if record["size"] != int(row["size"]) or record["sha256"] != row["sha256"]:
            raise ValueError("download identity drift")
        records.append(record)
    records.sort(key=lambda row: (str(row["role"]), str(row["scene_id"])))
    spec = config["gates"]
    role_counts = Counter(str(row["role"]) for row in records)
    total_bytes = sum(int(row["size"]) for row in records)
    gates = {
        "scene_count": len({row["scene_id"] for row in records}) == int(spec["required_scene_count"]),
        "member_count": len(records) == int(spec["required_member_count"]),
        "total_bytes": total_bytes == int(spec["required_total_bytes"]),
        "role_counts": role_counts == Counter({"fit": 80, "calibration": 24}),
        "sealed_requests_zero": int(spec["sealed_requests"]) == 0,
        "operator_fits_zero": int(spec["operator_fits"]) == 0,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "manifest_sha256": _sha256(manifest_bytes),
        "member_identity_sha256": manifest["member_identity_sha256"],
        "role_counts": dict(sorted(role_counts.items())),
        "scene_count": len(records),
        "member_count": len(records),
        "total_bytes": total_bytes,
        "records": records,
        "sealed_requests": 0,
        "operator_fits": 0,
        "gates": gates,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"] if passed else "invalidate_repid9b_acquisition",
        "claim_ceiling": config["claim_ceiling"],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["stable_evidence_id"] = f"sha256:{_sha256(canonical)}"
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(encoded)
    return {"report": report, "sha256": _sha256(encoded)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/u5_r2repid9b_original_acquisition_v1.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = run(
        config_path=config_path,
        output_root=args.output_root.resolve(),
        report_path=args.report.resolve(),
        reverse=args.reverse,
    )
    print(json.dumps({"decision": result["report"]["decision"], "sha256": result["sha256"]}, sort_keys=True))
    return 0 if result["report"]["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
