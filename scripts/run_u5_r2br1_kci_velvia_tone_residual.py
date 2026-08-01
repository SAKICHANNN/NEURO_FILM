#!/usr/bin/env python
"""Run the frozen U5.R2BR1 controlled Velvia neutral-tone experiment."""

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

from scripts.run_u5_r2br0_kci_velvia_simulation_residual import (  # noqa: E402
    CONFIG_SHA256 as PARENT_CONFIG_SHA256,
    load_config as load_parent_config,
)
from src.real_film.kci_velvia_simulation_residual import (  # noqa: E402
    evaluate_tone_residual,
    extract_exact_pair,
)


CONFIG_SHA256 = "2a0876d3ffee597e3d4005f39c45fb18c4084a060cbe632a6dff7ef941841ea5"
REPORT_SCHEMA = "neuro-film.u5.r2br1.kci-velvia-tone-residual-report.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("BR1 config hash mismatch")
    config = json.loads(raw)
    if (
        config.get("experiment_id") != "u5.r2br1-kci-velvia-tone-residual-v1"
        or config["parent"]["config_sha256"] != PARENT_CONFIG_SHA256
        or config["production_integration_allowed"]
        or config["stock_response_claim_allowed"]
        or config["calibrated_reference_claim_allowed"]
    ):
        raise ValueError("BR1 frozen contract mismatch")
    return config


def _require_clean_tracked_worktree() -> None:
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("BR1 requires a clean tracked worktree")


def run(
    config: dict[str, Any],
    *,
    parent_config_path: Path,
    digital_plot: Path,
    film_plot: Path,
    output: Path,
) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    parent = load_parent_config(
        parent_config_path, expected_sha256=PARENT_CONFIG_SHA256
    )
    extracted = extract_exact_pair(digital_plot, film_plot, parent)
    result = evaluate_tone_residual(
        extracted.pop("digital_rgb_u8"), extracted.pop("film_rgb_u8"), config
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "parent": config["parent"],
        "extraction": extracted,
        **result,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = _canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2br1_kci_velvia_tone_residual_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=ROOT / "configs/u5_r2br0_kci_velvia_simulation_residual_v1.json",
    )
    parser.add_argument("--digital-plot", type=Path, required=True)
    parser.add_argument("--film-plot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, expected_sha256=args.expected_config_sha256)
    report = run(
        config,
        parent_config_path=args.parent_config,
        digital_plot=args.digital_plot,
        film_plot=args.film_plot,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "selected_model": report["selected_model"],
                "identity_error": report["aggregate"]["identity"]["mean_confirmation_absolute_lstar_error"],
                "gamma_error": report["aggregate"]["gamma"]["mean_confirmation_absolute_lstar_error"],
                "affine_error": report["aggregate"]["affine"]["mean_confirmation_absolute_lstar_error"],
                "pchip_error": report["aggregate"]["monotone_pchip"]["mean_confirmation_absolute_lstar_error"],
                "report_sha256": _sha256(_canonical_json(report)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
