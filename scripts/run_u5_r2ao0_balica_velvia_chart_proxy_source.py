#!/usr/bin/env python
"""Acquire and validate the frozen U5.R2AO0 real-film chart proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.velvia_chart_proxy import (  # noqa: E402
    download_exact_asset,
    extract_chart_proxy,
)


CONFIG_SHA256 = "522f292c786135116f40949c816c5406661620306d1a278226370d4fb802b2b8"
REPORT_SCHEMA = "neuro-film.u5.r2ao0.balica-velvia-chart-source-report.v1"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AO0 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO0 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2ao0-balica-velvia-chart-proxy-source-v1"
        or config["article"]["license"] != "CC BY 4.0"
        or config["asset"]["expected_sha256"]
        != "cca9de4255dfd15afb325e6d845ab782e6550429a1d1ab18a39d150b358aca35"
        or int(config["download_count_maximum"]) != 2
        or config["production_integration_allowed"]
        or config["calibrated_reference_claim_allowed"]
        or config["stock_response_claim_allowed"]
    ):
        raise ValueError("AO0 frozen contract mismatch")
    return config


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset = config["asset"]
    maximum_per_download = int(config["download_bytes_maximum"]) // 2
    paths = [output_dir / "figure_8_a.jpg", output_dir / "figure_8_b.jpg"]
    downloads = [
        download_exact_asset(
            str(asset["official_cdn_url"]),
            path,
            maximum_bytes=maximum_per_download,
        )
        for path in paths
    ]
    extractions = [extract_chart_proxy(path, config) for path in paths]
    download_bytes_identical = paths[0].read_bytes() == paths[1].read_bytes()
    extraction_identical = _canonical_json(extractions[0]) == _canonical_json(
        extractions[1]
    )
    checks = [
        {
            "name": "two_downloads_byte_identical",
            "passed": download_bytes_identical,
        },
        {
            "name": "two_extractions_identical",
            "passed": extraction_identical,
        },
        {
            "name": "download_count_and_budget",
            "passed": len(paths) <= int(config["download_count_maximum"])
            and sum(int(value["bytes"]) for value in downloads)
            <= int(config["download_bytes_maximum"]),
        },
        {
            "name": "both_extractions_pass",
            "passed": all(bool(value["automatic_pass"]) for value in extractions),
        },
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "source_url": asset["official_cdn_url"],
        "source_license": config["article"]["license"],
        "downloads": downloads,
        "extraction": extractions[0],
        "automatic_checks": checks,
        "automatic_pass": all(bool(check["passed"]) for check in checks),
        "decision": (
            "open_grouped_display_proxy_explainability"
            if all(bool(check["passed"]) for check in checks)
            else "close_real_velvia_chart_proxy_source"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    _atomic_write(output_dir / "report.json", _canonical_json(report))
    _atomic_write(
        output_dir / "paired_patches.json",
        _canonical_json(
            {
                "schema": "neuro-film.u5.r2ao0.paired-patches.v1",
                "source_asset_sha256": asset["expected_sha256"],
                "paired_patch_channel_order": [
                    "film_rgb",
                    "reference_rgb",
                ],
                "paired_patch_u8_sha256": extractions[0][
                    "paired_patch_u8_sha256"
                ],
                "reference_patch_rgb_u8": extractions[0][
                    "reference_patch_rgb_u8"
                ],
                "film_patch_rgb_u8": extractions[0]["film_patch_rgb_u8"],
                "claim_ceiling": config["claim_ceiling"],
            }
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ao0_balica_velvia_chart_proxy_source_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(
        args.config, expected_sha256=args.expected_config_sha256
    )
    report = run(config, args.output_dir)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": _sha256(_canonical_json(report)),
                "paired_patch_u8_sha256": report["extraction"][
                    "paired_patch_u8_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
