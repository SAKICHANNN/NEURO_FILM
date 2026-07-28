#!/usr/bin/env python
"""Acquire and extract the frozen AO4P paired palette proxy."""

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

from src.real_film.figure_recon import acquire_official_figure  # noqa: E402
from src.real_film.paired_palette_proxy import (  # noqa: E402
    extract_palette_groups,
)


CONFIG_SHA256 = "36e249cb21c72f837cf8236e18817f2269570f78850c3cdf674aa3278f656425"
REPORT_SCHEMA = "neuro-film.u5.r2ao4p.paired-palette-proxy-report.v1"


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
            raise RuntimeError("AO4P requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO4P config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2ao4p-balica-paired-palette-proxy-v1"
        or [int(group["expected_pair_count"]) for group in config["groups"]]
        != [47, 56]
        or config["operator_fitting_allowed"]
        or config["training_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AO4P frozen contract mismatch")
    return config


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    asset = config["asset"]
    path = output_dir / "figure_11_fresh.jpg"
    download = acquire_official_figure(
        str(asset["official_cdn_url"]),
        path,
        maximum_bytes=int(config["download_bytes_maximum"]),
        allowed_media_types=("image/jpeg",),
    )
    extraction = extract_palette_groups(path, config)
    checks = [
        {
            "name": "download_identity_exact",
            "passed": download["bytes"] == int(asset["expected_bytes"])
            and download["sha256"] == asset["expected_sha256"],
        },
        {
            "name": "extraction_exact",
            "passed": extraction["automatic_pass"],
        },
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "source_license": config["article"]["license"],
        "download": download,
        "extraction": extraction,
        "checks": checks,
        "automatic_pass": all(bool(check["passed"]) for check in checks),
        "decision": (
            "open_cross_domain_velvia_validation"
            if all(bool(check["passed"]) for check in checks)
            else "close_paired_palette_proxy"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    _atomic_write(output_dir / "paired_palettes.json", _canonical_json(extraction))
    _atomic_write(output_dir / "report.json", _canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ao4p_balica_paired_palette_proxy_v1.json",
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
                "pair_counts": [
                    group["pair_count"]
                    for group in report["extraction"]["groups"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
