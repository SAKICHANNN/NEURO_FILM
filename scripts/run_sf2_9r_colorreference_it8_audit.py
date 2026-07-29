#!/usr/bin/env python
"""Acquire and audit the frozen SF2.9R multi-family IT8 references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.it8_reference import parse_cgats_spectral, parse_it8


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path, expected_bytes: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == expected_bytes:
        return "reused"
    temporary = path.with_name(path.name + ".part")
    request = Request(url, headers={"User-Agent": "K-MCFM-SF2.9R/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            if output.tell() > expected_bytes:
                raise ValueError(f"download exceeds frozen bytes: {path}")
    if temporary.stat().st_size != expected_bytes:
        raise ValueError(
            f"download byte mismatch: {temporary.stat().st_size} != {expected_bytes}"
        )
    temporary.replace(path)
    return "downloaded"


def _member_by_suffix(archive: ZipFile, suffix: str) -> str:
    names = sorted(
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and info.filename.lower().endswith(suffix)
    )
    if len(names) != 1:
        raise ValueError(f"expected one {suffix} member, found {names}")
    return names[0]


def _it8_member(archive: ZipFile, archive_stem: str) -> str:
    """Find the exact charge table across the historical .txt/.it8 naming."""

    names = sorted(
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and Path(info.filename).stem.casefold() == archive_stem.casefold()
        and Path(info.filename).suffix.casefold() in {".it8", ".txt"}
        and "extras" not in {
            part.casefold() for part in Path(info.filename).parts[:-1]
        }
    )
    if len(names) != 1:
        raise ValueError(
            f"expected one exact charge IT8/.txt member, found {names}"
        )
    return names[0]


def audit_archive(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"archive CRC failed: {path}/{bad_member}")
        inventory = [
            {
                "name": info.filename,
                "bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
            }
            for info in sorted(archive.infolist(), key=lambda item: item.filename)
            if not info.is_dir()
        ]
        it8_name = _it8_member(archive, path.stem)
        cgt_name = _member_by_suffix(archive, ".cgt")
        it8_payload = archive.read(it8_name)
        cgt_payload = archive.read(cgt_name)
    reference = parse_it8(it8_payload)
    spectral = parse_cgats_spectral(cgt_payload)
    if reference.sample_ids != spectral.sample_ids:
        raise ValueError(f"IT8/CGATS sample order mismatch: {path}")
    labs = {
        sample_id: lab
        for sample_id, lab in zip(spectral.sample_ids, spectral.lab, strict=True)
    }
    mean_de = [
        float(row["MEAN_DE"])
        for row in reference.rows
        if "MEAN_DE" in row
    ]
    if len(mean_de) != len(reference.rows):
        raise ValueError(f"missing MEAN_DE measurements: {path}")
    return {
        "path": asset["path"],
        "declared_family": asset["declared_family"],
        "configured_exact_stock_id": asset["exact_stock_id"],
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "archive_crc_clean": True,
        "members": inventory,
        "it8_member": it8_name,
        "it8_sha256": _sha256(it8_payload),
        "cgats_member": cgt_name,
        "cgats_sha256": _sha256(cgt_payload),
        "header": {
            key: reference.header.get(key, "unknown")
            for key in (
                "DESCRIPTOR",
                "CREATED",
                "PROD_DATE",
                "DIFFUSE_GEOMETRY",
                "SERIAL",
                "MATERIAL",
            )
        },
        "sample_count": len(reference.rows),
        "sample_ids": list(reference.sample_ids),
        "field_count": len(reference.fields),
        "spectral_sample_count": len(spectral.sample_ids),
        "spectral_wavelengths_nm": list(spectral.wavelengths_nm),
        "mean_batch_delta_e76_median": statistics.median(mean_de),
        "mean_batch_delta_e76_p90": _percentile(mean_de, 0.9),
        "_labs": labs,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def build_report(
    config: dict[str, Any],
    config_sha256: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    common_ids = set(records[0]["sample_ids"])
    for record in records[1:]:
        common_ids &= set(record["sample_ids"])
    pairwise = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            distances = [
                _distance(left["_labs"][sample_id], right["_labs"][sample_id])
                for sample_id in sorted(common_ids)
            ]
            pairwise.append(
                {
                    "left": left["path"],
                    "right": right["path"],
                    "common_samples": len(distances),
                    "delta_e76_median": statistics.median(distances),
                    "delta_e76_p90": _percentile(distances, 0.9),
                    "delta_e76_maximum": max(distances),
                }
            )
    for record in records:
        del record["_labs"]
    common_fraction = len(common_ids) / max(
        len(record["sample_ids"]) for record in records
    )
    gates = config["gates"]
    checks = {
        "expected_archive_count": len(records) == len(config["assets"]),
        "all_expected_bytes": all(
            record["bytes"] == asset["expected_bytes"]
            for record, asset in zip(records, config["assets"], strict=True)
        ),
        "expected_total_bytes": sum(record["bytes"] for record in records)
        == int(config["expected_total_bytes"]),
        "all_archives_crc_clean": all(
            record["archive_crc_clean"] for record in records
        ),
        "all_reference_files_parse": all(
            record["sample_count"] == record["spectral_sample_count"] == 288
            for record in records
        ),
        "common_patch_id_fraction": common_fraction
        >= float(gates["minimum_common_patch_id_fraction"]),
        "spectral_family_support": sum(
            bool(record["spectral_wavelengths_nm"]) for record in records
        )
        >= int(gates["minimum_families_with_spectral_measurements"]),
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": config_sha256,
        "expected_total_bytes": int(config["expected_total_bytes"]),
        "observed_total_bytes": sum(record["bytes"] for record in records),
        "archive_count": len(records),
        "common_patch_count": len(common_ids),
        "common_patch_fraction": common_fraction,
        "pairwise_target_lab_delta_e76": pairwise,
        "records": records,
        "checks": checks,
        "automatic_source_gate_passed": all(checks.values()),
        "identifiability": {
            "source_purpose": "scanner calibration target references",
            "common_uncalibrated_camera_scene_input_identified": False,
            "film_recorder_input_or_aim_adjustment_lineage_present": False,
            "independent_camera_exposure_pairs_present": False,
            "target_batch_average_measurements": True,
            "stock_operator_fitting_allowed": False,
            "reason": (
                "Direct spectra are valuable physical/nuisance evidence, but the "
                "targets were manufactured for IT8 scanner calibration and no "
                "uncalibrated common recorder input or camera-scene exposure is "
                "lineaged. Cross-family residuals are not a stock appearance operator."
            ),
        },
        "decision": "physical_spectral_and_target_manufacturing_prior_only",
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf2_9r_colorreference_multifamily_it8_v1.json",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    data_root = ROOT / config["data_root"]
    records = []
    for asset in config["assets"]:
        path = data_root / asset["path"]
        transfer = "skipped"
        if not args.skip_download:
            transfer = _download(asset["url"], path, int(asset["expected_bytes"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        record = audit_archive(path, asset)
        record["transfer"] = transfer
        records.append(record)
    report = build_report(config, _sha256(config_bytes), records)
    output = args.output or ROOT / config["output_root"] / "audit.json"
    if not output.is_absolute():
        output = ROOT / output
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(encoded),
                "automatic_source_gate_passed": report[
                    "automatic_source_gate_passed"
                ],
                "decision": report["decision"],
                "common_patch_count": report["common_patch_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_source_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
