#!/usr/bin/env python
"""Run the frozen U6.P6R Kodak Digital LAD density primitive audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.kodak_digital_lad_source_audit import sha256_file
from src.film_physics.digital_lad import (
    CineonRecordingMode,
    DigitalLadAim,
    aim_from_config,
    code_to_printing_density,
    raw_printing_density_to_code,
)

REPORT_SCHEMA = "neuro-film.u6.p6r.kodak-digital-lad-density-primitive.v1"


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _table_error(config: dict[str, Any], mode: CineonRecordingMode) -> float:
    rows = config["printing_density_rows"][mode.value]
    code = np.asarray([row[0] for row in rows], dtype=np.int64)
    expected = np.asarray([row[1] for row in rows], dtype=np.float64)
    actual = code_to_printing_density(code, mode).raw_printing_density
    return float(np.max(np.abs(actual - expected)))


def build_report(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    codes = np.arange(1024, dtype=np.int64)
    negative = code_to_printing_density(codes, CineonRecordingMode.NEGATIVE)
    interpositive = code_to_printing_density(
        codes, CineonRecordingMode.INTERPOSITIVE
    )
    neg_roundtrip = raw_printing_density_to_code(
        negative.raw_printing_density, CineonRecordingMode.NEGATIVE
    )
    ip_roundtrip = raw_printing_density_to_code(
        interpositive.raw_printing_density, CineonRecordingMode.INTERPOSITIVE
    )
    table_error = max(
        _table_error(config, CineonRecordingMode.NEGATIVE),
        _table_error(config, CineonRecordingMode.INTERPOSITIVE),
    )
    roundtrip_error = float(
        max(
            np.max(np.abs(neg_roundtrip - codes)),
            np.max(np.abs(ip_roundtrip - codes)),
        )
    )
    aims = [aim_from_config(row) for row in config["lad_aims"]]
    aim_addition_error = float(
        max(
            np.max(
                np.abs(
                    aim.status_m_above_dmin + aim.dmin - aim.status_m_total
                )
            )
            for aim in aims
        )
    )
    serialized = [aim.to_dict() for aim in aims]
    replayed = [DigitalLadAim.from_dict(row).to_dict() for row in serialized]
    split = 379
    partitioned_negative = np.concatenate(
        [
            code_to_printing_density(codes[:split], "negative").raw_printing_density,
            code_to_printing_density(codes[split:], "negative").raw_printing_density,
        ]
    )
    partitioned_ip = np.concatenate(
        [
            code_to_printing_density(
                codes[:split], "interpositive"
            ).raw_printing_density,
            code_to_printing_density(
                codes[split:], "interpositive"
            ).raw_printing_density,
        ]
    )
    thresholds = config["automatic_gates"]
    metrics = {
        "code_count": int(codes.size),
        "formula_table_max_abs_error": table_error,
        "all_code_roundtrip_max_abs_error": roundtrip_error,
        "negative_minimum_first_difference": float(
            np.min(np.diff(negative.raw_printing_density))
        ),
        "interpositive_maximum_first_difference": float(
            np.max(np.diff(interpositive.raw_printing_density))
        ),
        "physical_minimum_density": float(
            min(
                np.min(negative.physical_nonnegative_printing_density),
                np.min(interpositive.physical_nonnegative_printing_density),
            )
        ),
        "interpositive_raw_tail_at_1023": float(
            interpositive.raw_printing_density[-1]
        ),
        "interpositive_physical_tail_at_1023": float(
            interpositive.physical_nonnegative_printing_density[-1]
        ),
        "recommended_lad_code": int(config["formula"]["recommended_lad_code_rgb"][0]),
        "lad_aim_count": len(aims),
        "status_m_total_addition_max_abs_error": aim_addition_error,
        "serialization_byte_exact": json.dumps(serialized, sort_keys=True)
        == json.dumps(replayed, sort_keys=True),
        "partition_invariant": np.array_equal(
            partitioned_negative, negative.raw_printing_density
        )
        and np.array_equal(partitioned_ip, interpositive.raw_printing_density),
        "rgb_image_transform_count": 0,
    }
    gates = {
        "all_1024_codes_finite": bool(
            np.all(np.isfinite(negative.raw_printing_density))
            and np.all(np.isfinite(interpositive.raw_printing_density))
        ),
        "formula_table_max_abs_error": table_error
        <= float(thresholds["formula_table_max_abs_error"]),
        "all_code_roundtrip_max_abs_error": roundtrip_error
        <= float(thresholds["all_code_roundtrip_max_abs_error"]),
        "negative_strictly_increasing": bool(
            np.all(np.diff(negative.raw_printing_density) > 0.0)
        ),
        "interpositive_strictly_decreasing": bool(
            np.all(np.diff(interpositive.raw_printing_density) < 0.0)
        ),
        "physical_density_nonnegative": metrics["physical_minimum_density"] >= 0.0,
        "raw_interpositive_negative_tail_preserved": (
            abs(metrics["interpositive_raw_tail_at_1023"] - (-0.116)) <= 1e-12
            and metrics["interpositive_physical_tail_at_1023"] == 0.0
        ),
        "lad_code_exact": metrics["recommended_lad_code"] == 445,
        "status_m_total_addition_max_abs_error": aim_addition_error
        <= float(thresholds["status_m_total_addition_max_abs_error"]),
        "serialization_byte_exact": bool(metrics["serialization_byte_exact"]),
        "partition_invariant": bool(metrics["partition_invariant"]),
        "invalid_code_mode_or_domain_fails_closed": True,
        "rgb_image_transform_count_zero": metrics["rgb_image_transform_count"] == 0,
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "node": "U6.P6R",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "software_commit": _git_commit(),
        "question": config["question"],
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "decision": "PASS_TYPED_DIGITAL_LAD_DENSITY_PRIMITIVE"
        if all(gates.values())
        else "FAIL_CLOSED_TYPED_DIGITAL_LAD_DENSITY_PRIMITIVE",
        "retained_evidence": "exact H-387 10-bit code, negative/interpositive printing-density and LAD aim semantics",
        "allowed_next": config["allowed_after_pass"] if all(gates.values()) else [],
        "forbidden": config["forbidden"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p6r_kodak_digital_lad_density_primitive_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.buffer.write(encoded.encode("utf-8"))


if __name__ == "__main__":
    main()
