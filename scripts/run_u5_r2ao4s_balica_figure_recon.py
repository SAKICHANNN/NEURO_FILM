#!/usr/bin/env python
"""Run the frozen AO4S additional official-figure reconnaissance."""

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

from src.real_film.figure_recon import (  # noqa: E402
    acquire_official_figure,
    build_vertical_contact_sheet,
)


CONFIG_SHA256 = "e77c1b440570159c94cfb3015bca03f3bfbd67c9c75024d147e2e228a46d9e34"
REPORT_SCHEMA = "neuro-film.u5.r2ao4s.balica-figure-recon-report.v1"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AO4S requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO4S config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2ao4s-balica-additional-figure-recon-v1"
        or len(config["assets"]) != 2
        or [int(row["figure"]) for row in config["assets"]] != [6, 11]
        or not config["acquisition"]["official_cdn_only"]
        or config["operator_fitting_allowed"]
        or config["training_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AO4S frozen contract mismatch")
    return config


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    acquisition = config["acquisition"]
    allowed_types = tuple(str(v) for v in acquisition["allowed_media_types"])
    records: list[dict[str, Any]] = []
    sheet_inputs: list[tuple[int, Path]] = []
    for asset in config["assets"]:
        number = int(asset["figure"])
        path = output_dir / f"figure_{number}.jpg"
        record = acquire_official_figure(
            str(asset["official_cdn_url"]),
            path,
            maximum_bytes=int(acquisition["maximum_bytes_per_asset"]),
            allowed_media_types=allowed_types,
        )
        records.append(
            {
                "figure": number,
                "source_url": asset["official_cdn_url"],
                **record,
            }
        )
        sheet_inputs.append((number, path))
    total_bytes = sum(int(row["bytes"]) for row in records)
    checks = [
        {
            "name": "exact_asset_count",
            "passed": len(records) == int(acquisition["maximum_assets"]),
        },
        {
            "name": "total_download_budget",
            "passed": total_bytes <= int(acquisition["maximum_total_bytes"]),
        },
        {
            "name": "all_rgb_jpeg",
            "passed": all(
                row["format"] == "JPEG" and row["mode"] == "RGB"
                for row in records
            ),
        },
    ]
    sheet = build_vertical_contact_sheet(
        sheet_inputs, output_dir / "contact_sheet.png"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "source_license": config["article"]["license"],
        "assets": records,
        "total_download_bytes": total_bytes,
        "contact_sheet": sheet,
        "automatic_checks": checks,
        "automatic_pass": all(bool(check["passed"]) for check in checks),
        "decision": "manual_semantic_audit_required",
        "claim_ceiling": config["claim_ceiling"],
    }
    _atomic_write(output_dir / "report.json", _canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ao4s_balica_additional_figure_recon_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, expected_sha256=args.expected_config_sha256)
    report = run(config, args.output_dir)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": _sha256(_canonical_json(report)),
                "total_download_bytes": report["total_download_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
