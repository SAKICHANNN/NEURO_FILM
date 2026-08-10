#!/usr/bin/env python3
"""Run U6.P8CH complete 100MP row-streamed Thomas-to-PNG evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import evaluate_u6_p8cg_24mp_native_thomas_png as p8cg
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes

ROOT = Path(__file__).resolve().parents[1]


def _validate_contract(contract: dict[str, Any]) -> None:
    candidate = contract.get("candidate", {})
    if (
        contract.get("schema")
        != "neuro_film.u6_p8ch_100mp_native_thomas_png_contract.v1"
        or contract.get("status") != "contract_frozen_implementation_ready"
        or candidate.get("scheduler_reducer_encoder_unchanged") is not True
        or candidate.get("shape_chw") != [3, 10_000, 10_000]
        or candidate.get("row_partition") != 128
        or candidate.get("bit_depth") != 16
        or candidate.get("full_output_allowed") is not False
        or candidate.get("model_profile_or_sample_change_allowed") is not False
    ):
        raise RuntimeError("P8CH contract drift")
    for parent in contract["parents"].values():
        payload = p8cg._json(ROOT / parent["path"])
        if (
            sha256_file(ROOT / parent["path"]) != parent["sha256"]
            or payload.get("decision") != parent["required_decision"]
        ):
            raise RuntimeError("P8CH parent drift")


def evaluate(contract_path: Path, output_dir: Path, llvm: Path) -> dict[str, Any]:
    original_validator = p8cg._validate_contract
    p8cg._validate_contract = _validate_contract
    try:
        report = p8cg.evaluate(contract_path, output_dir, llvm)
    finally:
        p8cg._validate_contract = original_validator
    report["schema"] = "neuro_film.u6_p8ch_100mp_native_thomas_png_result.v1"
    for run in report["performance"]["runs"]:
        run["worker"]["schema"] = (
            "neuro_film.u6_p8ch_100mp_native_thomas_png_worker.v1"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u6_p8ch_100mp_native_thomas_png_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/eval/u6_p8ch_100mp_native_thomas_png_v1",
    )
    parser.add_argument(
        "--llvm",
        type=Path,
        default=ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64",
    )
    args = parser.parse_args()
    report = evaluate(args.contract, args.output_dir, args.llvm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_bytes(canonical_bytes(report))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
