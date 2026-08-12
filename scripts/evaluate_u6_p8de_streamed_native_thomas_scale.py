#!/usr/bin/env python3
"""Confirm the 12MP streamed FFT spatial input to native Thomas runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_u6_p8ct_thomas_atomic_publication_scale import _monitored
from scripts.evaluate_u6_p8dd_native_thomas_fft_spatial_scale import _decode
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rgb16_png_conformance import build_msvc
from src.preprocess.output_encode import srgb_icc_profile

SCHEMA = "neuro_film.u6_p8de_streamed_native_thomas_scale_contract.v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_exact(parent: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / parent["path"]
    if sha256_file(path) != parent["sha256"]:
        raise ValueError(f"P8DE parent hash drift: {parent['path']}")
    return _json(path)


def _validate(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _json(contract_path)
    fixture = contract.get("fixture", {})
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "contract_frozen_implementation_ready"
        or fixture.get("height") != 3000
        or fixture.get("width") != 4000
        or fixture.get("tile_rows") != 512
        or fixture.get("baseline_runs") != 1
        or fixture.get("streamed_runs") != 2
    ):
        raise ValueError("P8DE contract drift")
    parents = contract["parents"]
    p8dd_contract = _load_exact(parents["p8dd_contract"])
    evidence = _load_exact(parents["p8dd_evidence"])
    if evidence.get("decision") != parents["p8dd_evidence"]["required_decision"]:
        raise ValueError("P8DE P8DD decision drift")
    p3k = _load_exact(parents["p3k_decision"])
    if p3k.get("decision") != parents["p3k_decision"]["required_decision"]:
        raise ValueError("P8DE P3K decision drift")
    return contract, p8dd_contract


def evaluate(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract, _p8dd_contract = _validate(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    build = build_msvc(ROOT, output_dir / "build")
    plans = [("fft", 0), ("streamed", 0), ("streamed", 1)]
    runs: list[dict[str, Any]] = []
    decoded: list[tuple[np.ndarray, str, str]] = []
    worker_script = ROOT / "scripts/evaluate_u6_p8dd_native_thomas_fft_spatial_scale.py"
    p8dd_path = ROOT / contract["parents"]["p8dd_contract"]["path"]
    for mode, index in plans:
        stem = f"{mode}_{index}"
        destination = output_dir / f"{stem}.png"
        result = output_dir / f"{stem}.json"
        command = [
            sys.executable,
            str(worker_script),
            "--worker",
            "--mode",
            mode,
            "--contract",
            str(p8dd_path),
            "--dll",
            str(Path(build["dll_path"])),
            "--destination",
            str(destination),
            "--result",
            str(result),
        ]
        runs.append(_monitored(command, result, timeout=90.0))
        decoded.append(_decode(destination))

    baseline = runs[0]
    candidates = runs[1:]
    baseline_rgb, baseline_decoded_sha, baseline_icc = decoded[0]
    candidate_decoded = decoded[1:]
    errors = [
        np.abs(baseline_rgb.astype(np.int64) - row[0].astype(np.int64))
        for row in candidate_decoded
    ]
    changed = [float(np.count_nonzero(error) / error.size) for error in errors]
    candidate_walls = [row["worker"]["wall_seconds"] for row in candidates]
    candidate_peaks = [row["peak_process_tree_rss_bytes"] for row in candidates]
    parent = contract["parents"]["p8dd_evidence"]
    baseline_peak = parent["baseline_maximum_process_tree_rss_bytes"]
    baseline_wall = parent["baseline_maximum_wall_seconds"]
    gates_cfg = contract["gates"]
    maximum_peak = max(candidate_peaks)
    maximum_wall = max(candidate_walls)
    gates = {
        "p8dd_baseline_output": baseline["worker"]["output_sha256"] == parent["expected_fft_output_sha256"],
        "p8dd_baseline_decoded": baseline_decoded_sha == parent["expected_fft_decoded_rgb16_sha256"],
        "input_exact": len({row["worker"]["input_sha256"] for row in runs}) == 1,
        "streamed_input_exact": len({row["worker"]["streamed_input_sha256"] for row in candidates}) == 1,
        "streamed_output_exact": len({row["worker"]["output_sha256"] for row in candidates}) == 1,
        "streamed_decoded_exact": len({row[1] for row in candidate_decoded}) == 1,
        "icc_exact": len({row[2] for row in decoded}) == 1 and baseline_icc == hashlib.sha256(srgb_icc_profile()).hexdigest(),
        "rss_reduction": baseline_peak - maximum_peak >= gates_cfg["minimum_rss_reduction_bytes"],
        "rss_ratio": maximum_peak / baseline_peak <= gates_cfg["maximum_rss_ratio_to_p8dd"],
        "wall_ratio": maximum_wall / baseline_wall <= gates_cfg["maximum_wall_ratio_to_p8dd"],
        "maximum_wall": maximum_wall <= gates_cfg["maximum_streamed_wall_seconds"],
        "wall_repeat": max(candidate_walls) / min(candidate_walls) <= gates_cfg["maximum_repeat_wall_ratio"],
        "rss_repeat": max(candidate_peaks) / min(candidate_peaks) <= gates_cfg["maximum_repeat_rss_ratio"],
        "decoded_code_error": max(int(np.max(error)) for error in errors) <= gates_cfg["maximum_decoded_code_error"],
        "decoded_changed_fraction": max(changed) <= gates_cfg["maximum_changed_decoded_fraction"],
        "worker_cleanup": all(row["stderr_empty"] and row["surviving_process_count"] == 0 for row in runs),
        "no_mapped_input_residue": not list(output_dir.glob("*.f32")),
        "no_stage_residue": not list(output_dir.glob("*.stage-*")),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "p8dd_contract_sha256": sha256_file(ROOT / contract["parents"]["p8dd_contract"]["path"]),
        "input_sha256": baseline["worker"]["input_sha256"],
        "streamed_input_sha256": candidates[0]["worker"]["streamed_input_sha256"],
        "baseline_output_sha256": baseline["worker"]["output_sha256"],
        "streamed_output_sha256": candidates[0]["worker"]["output_sha256"],
        "baseline_decoded_rgb16_sha256": baseline_decoded_sha,
        "streamed_decoded_rgb16_sha256": candidate_decoded[0][1],
        "icc_sha256": baseline_icc,
        "maximum_decoded_code_error": max(int(np.max(error)) for error in errors),
        "maximum_changed_decoded_fraction": max(changed),
        "gates": gates,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("_contract", "_report"),
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest(),
        "stable": stable,
        "performance": {
            "p8dd_baseline_maximum_process_tree_rss_bytes": baseline_peak,
            "maximum_streamed_process_tree_rss_bytes": maximum_peak,
            "rss_reduction_bytes": baseline_peak - maximum_peak,
            "rss_ratio_to_p8dd": maximum_peak / baseline_peak,
            "p8dd_baseline_maximum_wall_seconds": baseline_wall,
            "maximum_streamed_wall_seconds": maximum_wall,
            "wall_ratio_to_p8dd": maximum_wall / baseline_wall,
            "streamed_wall_repeat_ratio": max(candidate_walls) / min(candidate_walls),
            "streamed_rss_repeat_ratio": max(candidate_peaks) / min(candidate_peaks),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/u6_p8de_streamed_native_thomas_scale_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/eval/u6_p8de_streamed_native_thomas_scale_v1")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(args.contract, args.output_dir)
    report_path = args.report or args.output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "automatic_pass": report["automatic_pass"],
        "decision": report["stable"]["decision"],
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "stable_evidence_id": report["stable_evidence_id"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
