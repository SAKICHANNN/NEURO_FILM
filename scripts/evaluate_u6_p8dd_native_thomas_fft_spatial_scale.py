#!/usr/bin/env python3
"""Compare direct and FFT backing return through the 12MP native package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import _parent_payloads
from scripts.evaluate_u6_p8ck_native_thomas_rgb16_png_program import _icc_payload
from scripts.evaluate_u6_p8ct_thomas_atomic_publication_scale import _monitored
from scripts.evaluate_u6_p8db_native_thomas_spatial_scale import _fixture
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rgb16_png_conformance import build_msvc
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_thomas_package import resolve_native_thomas_package
from src.film_physics.native_thomas_runtime import NativeThomasExportRuntime
from src.film_physics.native_thomas_spatial_chain import (
    apply_native_thomas_spatial_chain,
    apply_native_thomas_spatial_chain_fft,
    compile_native_thomas_spatial_chain,
)
from src.preprocess.output_encode import srgb_icc_profile

SCHEMA = "neuro_film.u6_p8dd_native_thomas_fft_spatial_scale_contract.v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_exact(parent: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / parent["path"]
    if sha256_file(path) != parent["sha256"]:
        raise ValueError(f"P8DD parent hash drift: {parent['path']}")
    return _json(path)


def _validate(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _json(contract_path)
    fixture = contract.get("fixture", {})
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != "contract_frozen_implementation_ready"
        or fixture.get("height") != 3000
        or fixture.get("width") != 4000
        or fixture.get("fft_runs") != 2
        or fixture.get("direct_reference_runs") != 1
    ):
        raise ValueError("P8DD contract drift")
    parents = contract["parents"]
    p8dc = _load_exact(parents["p8dc_evidence"])
    if p8dc.get("decision") != parents["p8dc_evidence"]["required_decision"]:
        raise ValueError("P8DD P8DC decision drift")
    _load_exact(parents["p3h_contract"])
    p3h = _load_exact(parents["p3h_decision"])
    if p3h.get("automatic_pass") is not parents["p3h_decision"]["required_automatic_pass"]:
        raise ValueError("P8DD P3H decision drift")
    return (
        contract,
        _load_exact(parents["p1_contract"]),
        _load_exact(parents["p3d_contract"]),
        _load_exact(parents["package"]),
    )


def _worker(
    contract_path: Path,
    dll_path: Path,
    destination: Path,
    result_path: Path,
    mode: str,
) -> None:
    contract, p1, p3d, package = _validate(contract_path)
    fixture = contract["fixture"]
    prior = ManufacturerCharacteristicPrior.from_dict(_parent_payloads()[1]["prior"])
    exposure = _fixture(prior, fixture["height"], fixture["width"])
    input_sha256 = hashlib.sha256(exposure.values.tobytes()).hexdigest()
    chain = compile_native_thomas_spatial_chain(p1, p3d)
    resolved = resolve_native_thomas_package(
        package,
        profile_path=ROOT / contract["parents"]["package"]["profile_path"],
        library_path=dll_path,
    )
    runtime = NativeThomasExportRuntime(package=package, resolved=resolved)
    started = time.perf_counter()
    if mode == "direct":
        receipt = runtime.publish_spatial_layer_exposure(
            exposure, chain, destination=destination
        )
    elif mode == "fft":
        receipt = runtime.publish_fft_spatial_layer_exposure(
            exposure, chain, destination=destination
        )
    else:
        raise ValueError("P8DD worker mode drift")
    wall = time.perf_counter() - started
    if hashlib.sha256(exposure.values.tobytes()).hexdigest() != input_sha256:
        raise RuntimeError("P8DD input mutated")
    result_path.write_bytes(
        canonical_bytes(
            {
                "mode": mode,
                "input_sha256": input_sha256,
                "output_sha256": receipt["output"]["sha256"],
                "output_bytes": receipt["output"]["bytes"],
                "wall_seconds": wall,
            }
        )
    )


def _decode(path: Path) -> tuple[np.ndarray, str, str]:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.shape != (3000, 4000, 3):
        raise RuntimeError("P8DD decoded PNG drift")
    rgb = np.ascontiguousarray(decoded[..., ::-1])
    return (
        rgb,
        hashlib.sha256(rgb.tobytes()).hexdigest(),
        hashlib.sha256(_icc_payload(path.read_bytes())).hexdigest(),
    )


def _numerical_probe(
    contract: dict[str, Any], p1: dict[str, Any], p3d: dict[str, Any]
) -> dict[str, float]:
    fixture = contract["fixture"]
    prior = ManufacturerCharacteristicPrior.from_dict(_parent_payloads()[1]["prior"])
    exposure = _fixture(
        prior,
        fixture["numerical_probe_height"],
        fixture["numerical_probe_width"],
    )
    chain = compile_native_thomas_spatial_chain(p1, p3d)
    direct = apply_native_thomas_spatial_chain(exposure, chain).values
    fft = apply_native_thomas_spatial_chain_fft(exposure, chain).values
    error = np.abs(direct.astype(np.float64) - fft.astype(np.float64))
    relative = error / np.maximum(np.abs(direct.astype(np.float64)), 1e-30)
    spacing = np.spacing(direct).astype(np.float64)
    ulp = np.divide(error, spacing, out=np.zeros_like(error), where=spacing != 0)
    return {
        "maximum_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error)),
        "maximum_relative_error": float(np.max(relative)),
        "p99_ulp_error": float(np.quantile(ulp, 0.99)),
        "maximum_ulp_error": float(np.max(ulp)),
    }


def evaluate(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract, p1, p3d, _package = _validate(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    build = build_msvc(ROOT, output_dir / "build")
    plans = [("direct", 0), ("fft", 0), ("fft", 1)]
    monitored_runs: list[dict[str, Any]] = []
    decoded_rows: list[tuple[np.ndarray, str, str]] = []
    for mode, index in plans:
        stem = f"{mode}_{index}"
        destination = output_dir / f"{stem}.png"
        result = output_dir / f"{stem}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--mode",
            mode,
            "--contract",
            str(contract_path.resolve()),
            "--dll",
            str(Path(build["dll_path"])),
            "--destination",
            str(destination),
            "--result",
            str(result),
        ]
        monitored_runs.append(_monitored(command, result, timeout=60.0))
        decoded_rows.append(_decode(destination))
    numerical = _numerical_probe(contract, p1, p3d)
    direct_run = monitored_runs[0]
    fft_runs = monitored_runs[1:]
    direct_worker = direct_run["worker"]
    fft_workers = [row["worker"] for row in fft_runs]
    direct_rgb, direct_decoded_sha, direct_icc = decoded_rows[0]
    fft_decoded = decoded_rows[1:]
    code_errors = [
        np.abs(direct_rgb.astype(np.int64) - row[0].astype(np.int64))
        for row in fft_decoded
    ]
    changed_fractions = [float(np.count_nonzero(row) / row.size) for row in code_errors]
    fft_walls = [row["wall_seconds"] for row in fft_workers]
    fft_peaks = [row["peak_process_tree_rss_bytes"] for row in fft_runs]
    parent = contract["parents"]["p8dc_evidence"]
    gates = {
        "direct_parent_output": direct_worker["output_sha256"] == parent["expected_output_sha256"],
        "direct_parent_decoded": direct_decoded_sha == parent["expected_decoded_rgb16_sha256"],
        "input_exact": len({row["worker"]["input_sha256"] for row in monitored_runs}) == 1,
        "fft_output_exact": len({row["output_sha256"] for row in fft_workers}) == 1,
        "fft_decoded_exact": len({row[1] for row in fft_decoded}) == 1,
        "icc_exact": len({row[2] for row in decoded_rows}) == 1 and direct_icc == hashlib.sha256(srgb_icc_profile()).hexdigest(),
        "relative_exposure_error": numerical["maximum_relative_error"] <= contract["gates"]["maximum_relative_exposure_error"],
        "p99_exposure_ulp_error": numerical["p99_ulp_error"] <= contract["gates"]["maximum_p99_exposure_ulp_error"],
        "maximum_exposure_ulp_error": numerical["maximum_ulp_error"] <= contract["gates"]["maximum_exposure_ulp_error"],
        "decoded_code_error": max(int(np.max(row)) for row in code_errors) <= contract["gates"]["maximum_decoded_code_error"],
        "decoded_changed_fraction": max(changed_fractions) <= contract["gates"]["maximum_changed_decoded_fraction"],
        "wall_speedup": direct_worker["wall_seconds"] / max(fft_walls) >= contract["gates"]["minimum_wall_speedup_over_direct"],
        "maximum_fft_wall": max(fft_walls) <= contract["gates"]["maximum_fft_wall_seconds"],
        "maximum_rss": max(fft_peaks) <= contract["gates"]["maximum_process_tree_rss_bytes"],
        "wall_repeat": max(fft_walls) / min(fft_walls) <= contract["gates"]["maximum_repeat_wall_ratio"],
        "rss_repeat": max(fft_peaks) / min(fft_peaks) <= contract["gates"]["maximum_repeat_rss_ratio"],
        "worker_cleanup": all(row["stderr_empty"] and row["surviving_process_count"] == 0 for row in monitored_runs),
        "no_stage_residue": not list(output_dir.glob("*.stage-*")),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": direct_worker["input_sha256"],
        "direct_output_sha256": direct_worker["output_sha256"],
        "fft_output_sha256": fft_workers[0]["output_sha256"],
        "direct_decoded_rgb16_sha256": direct_decoded_sha,
        "fft_decoded_rgb16_sha256": fft_decoded[0][1],
        "icc_sha256": direct_icc,
        "numerical_probe": numerical,
        "maximum_decoded_code_error": max(int(np.max(row)) for row in code_errors),
        "maximum_changed_decoded_fraction": max(changed_fractions),
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
            "direct_wall_seconds": direct_worker["wall_seconds"],
            "maximum_fft_wall_seconds": max(fft_walls),
            "wall_speedup_over_direct": direct_worker["wall_seconds"] / max(fft_walls),
            "maximum_fft_process_tree_rss_bytes": max(fft_peaks),
            "fft_wall_repeat_ratio": max(fft_walls) / min(fft_walls),
            "fft_rss_repeat_ratio": max(fft_peaks) / min(fft_peaks),
        },
        "runs": monitored_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/u6_p8dd_native_thomas_fft_spatial_scale_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/eval/u6_p8dd_native_thomas_fft_spatial_scale_v1")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("direct", "fft"))
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.mode is None or args.dll is None or args.destination is None or args.result is None:
            parser.error("--worker requires --mode, --dll, --destination and --result")
        _worker(args.contract, args.dll, args.destination, args.result, args.mode)
        return
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
