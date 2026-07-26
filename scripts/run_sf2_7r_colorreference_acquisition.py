#!/usr/bin/env python
"""Acquire and audit the frozen SF2.7R ColorReference nuisance lane."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, destination: Path, expected_bytes: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        return "reused"
    temporary = destination.with_name(destination.name + ".part")
    offset = temporary.stat().st_size if temporary.is_file() else 0
    if offset > expected_bytes:
        raise ValueError(f"partial file exceeds expected bytes: {destination}")
    headers = {"User-Agent": "K-MCFM-SF2.7R/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        status = int(getattr(response, "status", response.getcode()))
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset and status == 206 else "wb"
        with temporary.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                if handle.tell() > expected_bytes:
                    raise ValueError(f"download exceeds frozen bytes: {destination}")
    if temporary.stat().st_size != expected_bytes:
        raise ValueError(
            f"download byte mismatch for {destination}: "
            f"{temporary.stat().st_size} != {expected_bytes}"
        )
    temporary.replace(destination)
    return "downloaded"


def _image_metadata(payload: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
        }


def _slide_id(name: str) -> str | None:
    stem = Path(name).stem.lower()
    matches = re.findall(
        r"(?:slide|scan|test|img|image)?[_ -]*([1-5])(?:[_ -]?scaled)?$",
        stem,
    )
    return matches[-1] if matches else None


def audit_asset(path: Path, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
        "archive_crc_clean": None,
        "members": [],
    }
    if path.suffix.lower() in IMAGE_SUFFIXES:
        payload = path.read_bytes()
        record["image"] = _image_metadata(payload)
        record["image"]["sha256"] = hashlib.sha256(payload).hexdigest()
        return record
    if path.suffix.lower() != ".zip":
        raise ValueError(f"unsupported frozen asset: {path}")
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        record["archive_crc_clean"] = bad_member is None
        if bad_member is not None:
            raise ValueError(f"archive CRC failed: {path}/{bad_member}")
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            payload = archive.read(info)
            member: dict[str, Any] = {
                "name": info.filename,
                "bytes": len(payload),
                "crc32": f"{info.CRC:08x}",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            if Path(info.filename).suffix.lower() in IMAGE_SUFFIXES:
                member["image"] = _image_metadata(payload)
                member["slide_id"] = _slide_id(info.filename)
            record["members"].append(member)
    return record


def build_report(
    config: dict[str, Any],
    config_sha256: str,
    asset_records: list[dict[str, Any]],
) -> dict[str, Any]:
    scan_records = [
        record for record in asset_records if str(record["role"]).startswith("nikon_")
    ]
    complete_pipelines = 0
    image_hashes_by_pipeline: dict[str, dict[str, str]] = {}
    for record in scan_records:
        images = [
            member for member in record["members"] if "image" in member
        ]
        slide_map = {
            str(member["slide_id"]): str(member["sha256"])
            for member in images
            if member.get("slide_id") is not None
        }
        record["image_member_count"] = len(images)
        record["slide_ids"] = sorted(slide_map)
        if set(slide_map) == {"1", "2", "3", "4", "5"}:
            complete_pipelines += 1
        image_hashes_by_pipeline[str(record["role"])] = slide_map
    duplicate_pairs = []
    roles = sorted(image_hashes_by_pipeline)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            for slide_id in sorted(
                set(image_hashes_by_pipeline[left])
                & set(image_hashes_by_pipeline[right])
            ):
                if (
                    image_hashes_by_pipeline[left][slide_id]
                    == image_hashes_by_pipeline[right][slide_id]
                ):
                    duplicate_pairs.append(
                        {"left": left, "right": right, "slide_id": slide_id}
                    )
    gates = config["gates"]
    checks = {
        "all_expected_bytes": all(
            record["bytes"] == record["expected_bytes"] for record in asset_records
        ),
        "all_archives_crc_clean": all(
            record["archive_crc_clean"] in (None, True) for record in asset_records
        ),
        "all_images_decode": all(
            (
                "image" in record
                or all(
                    "image" not in member or member["image"]["width"] > 0
                    for member in record["members"]
                )
            )
            for record in asset_records
        ),
        "complete_scan_pipelines": complete_pipelines
        >= int(gates["minimum_complete_scan_pipelines"]),
        "complete_slide_ids": all(
            len(record.get("slide_ids", []))
            >= int(gates["minimum_complete_slide_ids_per_pipeline"])
            for record in scan_records
        ),
        "zero_cross_pipeline_exact_duplicates": len(duplicate_pairs) == 0,
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": config_sha256,
        "expected_total_bytes": config["expected_total_bytes"],
        "observed_total_bytes": sum(record["bytes"] for record in asset_records),
        "asset_count": len(asset_records),
        "complete_scan_pipelines": complete_pipelines,
        "cross_pipeline_exact_duplicate_pairs": duplicate_pairs,
        "assets": asset_records,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf2_7r_colorreference_scanner_nuisance_v1.json",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    data_root = ROOT / str(config["data_root"])
    records = []
    for asset in config["assets"]:
        path = data_root / str(asset["path"])
        transfer = "skipped"
        if not args.skip_download:
            transfer = download_resumable(
                str(asset["url"]), path, int(asset["expected_bytes"])
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        record = {
            "path": str(asset["path"]),
            "url": str(asset["url"]),
            "etag": str(asset["etag"]),
            "expected_bytes": int(asset["expected_bytes"]),
            "transfer": transfer,
            **audit_asset(path, str(asset["role"])),
        }
        records.append(record)
    report = build_report(
        config, hashlib.sha256(config_bytes).hexdigest(), records
    )
    output = args.output or ROOT / str(config["output_root"]) / "audit.json"
    if not output.is_absolute():
        output = ROOT / output
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "asset_count": len(records),
                "total_bytes": report["observed_total_bytes"],
                "all_checks_passed": report["all_checks_passed"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
