#!/usr/bin/env python
"""Acquire and audit the frozen SF2.9B multi-charge IT8 source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sf2_9r_colorreference_it8_audit import audit_archive


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assets(config: dict[str, Any]) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for family_code, family in sorted(config["families"].items()):
        for name in family["archive_names"]:
            assets.append(
                {
                    "path": name,
                    "url": f"https://www.colorreference.de/targets/{name}",
                    "family_code": family_code,
                    "declared_family": family["declared_family"],
                    "exact_stock_id": family["exact_stock_id"],
                }
            )
    names = [asset["path"] for asset in assets]
    if len(names) != len(set(names)):
        raise ValueError("duplicate archive name")
    expected = int(config["source_gates"]["expected_archive_count"])
    if len(assets) != expected:
        raise ValueError(f"archive count mismatch: {len(assets)} != {expected}")
    return assets


def _download(asset: dict[str, str], path: Path, maximum_bytes: int) -> str:
    if path.is_file():
        if path.stat().st_size > maximum_bytes:
            raise ValueError(f"existing archive exceeds bound: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    request = Request(
        asset["url"], headers={"User-Agent": "K-MCFM-SF2.9B/1.0"}
    )
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: {asset['url']}")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                if output.tell() > maximum_bytes:
                    raise ValueError(f"archive exceeds bound: {asset['path']}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def _year_from_name(name: str) -> int:
    two_digit = int(name[1:3])
    return 2000 + two_digit


def build_report(
    config: dict[str, Any],
    config_sha256: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = config["source_gates"]
    expected_wavelengths = list(
        range(
            int(gates["required_spectral_wavelengths_nm"]["start"]),
            int(gates["required_spectral_wavelengths_nm"]["stop"]) + 1,
            int(gates["required_spectral_wavelengths_nm"]["step"]),
        )
    )
    family_counts: dict[str, int] = {}
    family_years: dict[str, set[int]] = {}
    common_ids = set(records[0]["sample_ids"])
    for record in records:
        code = record["family_code"]
        family_counts[code] = family_counts.get(code, 0) + 1
        family_years.setdefault(code, set()).add(record["charge_year"])
        common_ids &= set(record["sample_ids"])
    shared_years = set.intersection(*family_years.values())
    total_bytes = sum(int(record["bytes"]) for record in records)
    checks = {
        "expected_archive_count": len(records)
        == int(gates["expected_archive_count"]),
        "expected_total_bytes": total_bytes == int(config["expected_total_bytes"]),
        "maximum_download_bytes": total_bytes
        <= int(config["maximum_download_bytes"]),
        "minimum_charges_per_family": min(family_counts.values())
        >= int(gates["minimum_charges_per_family"]),
        "minimum_shared_calendar_years": len(shared_years)
        >= int(gates["minimum_shared_calendar_years"]),
        "all_archives_crc_clean": all(
            bool(record["archive_crc_clean"]) for record in records
        ),
        "all_reference_files_parse": all(
            int(record["sample_count"]) == int(gates["required_sample_count"])
            and int(record["spectral_sample_count"])
            == int(gates["required_sample_count"])
            for record in records
        ),
        "spectral_grid_exact": all(
            record["spectral_wavelengths_nm"] == expected_wavelengths
            for record in records
        ),
        "common_patch_fraction": len(common_ids)
        / max(len(record["sample_ids"]) for record in records)
        >= float(gates["minimum_common_patch_fraction"]),
    }
    stable_evidence = {
        "config_sha256": config_sha256,
        "observed_total_bytes": total_bytes,
        "family_counts": family_counts,
        "family_years": {
            code: sorted(years) for code, years in sorted(family_years.items())
        },
        "shared_calendar_years": sorted(shared_years),
        "common_patch_count": len(common_ids),
        "records": [
            {
                key: record[key]
                for key in (
                    "path",
                    "family_code",
                    "charge_year",
                    "bytes",
                    "sha256",
                    "it8_sha256",
                    "cgats_sha256",
                    "header",
                    "sample_count",
                    "spectral_wavelengths_nm",
                    "mean_batch_delta_e76_median",
                    "mean_batch_delta_e76_p90",
                )
            }
            for record in records
        ],
        "checks": checks,
    }
    stable_bytes = (
        json.dumps(stable_evidence, sort_keys=True, separators=(",", ":")).encode()
    )
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": config_sha256,
        "stable_evidence_id": _sha256(stable_bytes),
        "observed_total_bytes": total_bytes,
        "archive_count": len(records),
        "family_counts": family_counts,
        "shared_calendar_years": sorted(shared_years),
        "common_patch_count": len(common_ids),
        "records": records,
        "checks": checks,
        "automatic_source_gate_passed": all(checks.values()),
        "decision": (
            "open_batch_held_out_family_identifiability_pilot"
            if all(checks.values())
            else "close_source_before_identifiability"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sf2_9b_colorreference_batch_family_v1.json",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    assets = _assets(config)
    data_root = ROOT / config["data_root"]
    records: list[dict[str, Any]] = []
    for asset in assets:
        path = data_root / asset["path"]
        transfer = "skipped"
        if not args.skip_download:
            transfer = _download(
                asset, path, int(config["maximum_archive_bytes"])
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        record = audit_archive(path, asset)
        record["family_code"] = asset["family_code"]
        record["charge_year"] = _year_from_name(asset["path"])
        record["transfer"] = transfer
        records.append(record)
        if sum(item["bytes"] for item in records) > int(
            config["maximum_download_bytes"]
        ):
            raise ValueError("aggregate download exceeds frozen bound")
    report = build_report(config, _sha256(config_bytes), records)
    output = args.output or ROOT / config["output_root"] / "source_audit.json"
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
                "stable_evidence_id": report["stable_evidence_id"],
                "automatic_source_gate_passed": report[
                    "automatic_source_gate_passed"
                ],
                "decision": report["decision"],
                "archive_count": report["archive_count"],
                "observed_total_bytes": report["observed_total_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_source_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
