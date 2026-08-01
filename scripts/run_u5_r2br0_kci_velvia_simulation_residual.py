#!/usr/bin/env python
"""Run the frozen U5.R2BR0 controlled Velvia chart residual experiment."""

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

from src.real_film.kci_velvia_simulation_residual import (  # noqa: E402
    evaluate_residual,
    extract_exact_pair,
)


CONFIG_SHA256 = "bd7edf5c86b3f1d50d639a8891cef6e8c52437216e8932f183da659dfe9556c3"
REPORT_SCHEMA = "neuro-film.u5.r2br0.kci-velvia-simulation-residual-report.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _require_clean_tracked_worktree() -> None:
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("BR0 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("BR0 config hash mismatch")
    config = json.loads(raw)
    if (
        config.get("experiment_id")
        != "u5.r2br0-kci-velvia-simulation-residual-v1"
        or config["split"]["formal_status"]
        != "independent_controlled_display-chain_development_not_stock_confirmation"
        or config["production_integration_allowed"]
        or config["stock_response_claim_allowed"]
        or config["calibrated_reference_claim_allowed"]
    ):
        raise ValueError("BR0 frozen contract mismatch")
    return config


def run(
    config: dict[str, Any],
    *,
    digital_plot: Path,
    film_plot: Path,
    output: Path,
) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    extracted = extract_exact_pair(digital_plot, film_plot, config)
    result = evaluate_residual(
        extracted.pop("digital_rgb_u8"), extracted.pop("film_rgb_u8"), config
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "source": config["source"],
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
        default=ROOT / "configs/u5_r2br0_kci_velvia_simulation_residual_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--digital-plot", type=Path, required=True)
    parser.add_argument("--film-plot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, expected_sha256=args.expected_config_sha256)
    report = run(
        config,
        digital_plot=args.digital_plot,
        film_plot=args.film_plot,
        output=args.output,
    )
    encoded = _canonical_json(report)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "selected_model": report["selected_model"],
                "identity_rmse": report["aggregate"]["identity"]["mean_confirmation_chroma_rmse"],
                "global_chroma_rmse": report["aggregate"]["global_chroma"]["mean_confirmation_chroma_rmse"],
                "full_matrix_rmse": report["aggregate"]["full_matrix"]["mean_confirmation_chroma_rmse"],
                "full_matrix_gain_over_global_chroma": report["aggregate"]["full_matrix"]["gain_over_global_chroma"],
                "report_sha256": _sha256(encoded),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
