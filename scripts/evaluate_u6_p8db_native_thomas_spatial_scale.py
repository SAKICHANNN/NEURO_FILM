#!/usr/bin/env python3
"""Measure retained exposure-domain spatial physics through native Thomas."""

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
from src.eval.native_msvc import sha256_file
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.eval.native_thomas_rgb16_png_conformance import build_msvc
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_thomas_package import resolve_native_thomas_package
from src.film_physics.native_thomas_runtime import NativeThomasExportRuntime
from src.film_physics.native_thomas_spatial_chain import (
    compile_native_thomas_spatial_chain,
)
from src.preprocess.output_encode import srgb_icc_profile

SCHEMAS = {
    "neuro_film.u6_p8db_native_thomas_spatial_scale_contract.v1",
    "neuro_film.u6_p8dc_native_thomas_fullframe_spatial_scale_contract.v1",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_exact(parent: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / parent["path"]
    if sha256_file(path) != parent["sha256"]:
        raise ValueError(f"P8DB parent hash drift: {parent['path']}")
    return _json(path)


def _validate(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _json(contract_path)
    fixture = contract.get("fixture", {})
    if (
        contract.get("schema") not in SCHEMAS
        or contract.get("status") != "contract_frozen_implementation_ready"
        or fixture.get("height") != 3000
        or fixture.get("width") != 4000
        or fixture.get("pixel_pitch_um") != 8.0
        or fixture.get("runs") != 2
        or fixture.get("log_domain_fraction_low") != 0.2
        or fixture.get("log_domain_fraction_high") != 0.6
    ):
        raise ValueError("P8DB contract drift")
    expected_tile_rows = 128 if "p8db_" in contract["schema"] else None
    if fixture.get("tile_rows") != expected_tile_rows:
        raise ValueError("P8DB spatial execution drift")
    parents = contract["parents"]
    p8da = _load_exact(parents["p8da_evidence"])
    if p8da.get("decision") != parents["p8da_evidence"]["required_decision"]:
        raise ValueError("P8DB P8DA decision drift")
    if "p8db_evidence" in parents:
        p8db = _load_exact(parents["p8db_evidence"])
        if p8db.get("decision") != parents["p8db_evidence"]["required_decision"]:
            raise ValueError("P8DC P8DB decision drift")
    p1 = _load_exact(parents["p1_contract"])
    p3d = _load_exact(parents["p3d_contract"])
    package = _load_exact(parents["package"])
    return contract, p1, p3d, package


def _fixture(prior: ManufacturerCharacteristicPrior, height: int, width: int) -> PhysicalDomainArray:
    columns = np.arange(width, dtype=np.uint64)
    values = np.empty((height, width, 3), dtype=np.float32)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        span = upper - lower
        for row in range(height):
            rank = ((columns * np.uint64(17) + np.uint64(row * 13 + channel * 29)) % 4093).astype(np.float64) / 4092.0
            log_exposure = lower + (0.2 + 0.4 * rank) * span
            np.power(10.0, log_exposure, out=values[row, :, channel], dtype=np.float32)
    return PhysicalDomainArray.adopt(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(8.0),
    )


def _worker(contract_path: Path, dll_path: Path, destination: Path, result_path: Path) -> None:
    contract, p1, p3d, package = _validate(contract_path)
    prior = ManufacturerCharacteristicPrior.from_dict(_parent_payloads()[1]["prior"])
    fixture = contract["fixture"]
    exposure = _fixture(prior, fixture["height"], fixture["width"])
    input_sha256 = hashlib.sha256(exposure.values.tobytes()).hexdigest()
    chain = compile_native_thomas_spatial_chain(p1, p3d)
    resolved = resolve_native_thomas_package(
        package,
        profile_path=ROOT / contract["parents"]["package"]["profile_path"],
        library_path=dll_path,
    )
    started = time.perf_counter()
    receipt = NativeThomasExportRuntime(package=package, resolved=resolved).publish_spatial_layer_exposure(
        exposure,
        chain,
        destination=destination,
        tile_rows=fixture["tile_rows"],
    )
    wall = time.perf_counter() - started
    if hashlib.sha256(exposure.values.tobytes()).hexdigest() != input_sha256:
        raise RuntimeError("P8DB input mutated")
    result_path.write_bytes(canonical_bytes({
        "input_sha256": input_sha256,
        "output_sha256": receipt["output"]["sha256"],
        "output_bytes": receipt["output"]["bytes"],
        "wall_seconds": wall,
    }))


def evaluate(contract_path: Path, output_dir: Path) -> dict[str, Any]:
    contract, _p1, _p3d, _package = _validate(contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    build = build_msvc(ROOT, output_dir / "build")
    runs = []
    decoded_hashes = []
    icc_hashes = []
    for index in range(contract["fixture"]["runs"]):
        destination = output_dir / f"run_{index}.png"
        result = output_dir / f"run_{index}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--contract",
            str(contract_path.resolve()),
            "--dll",
            str(Path(build["dll_path"])),
            "--destination",
            str(destination),
            "--result",
            str(result),
        ]
        monitored = _monitored(command, result, timeout=60.0)
        decoded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.dtype != np.uint16 or decoded.shape != (3000, 4000, 3):
            raise RuntimeError("P8DB decoded PNG drift")
        decoded_hashes.append(hashlib.sha256(decoded[..., ::-1].tobytes()).hexdigest())
        icc_hashes.append(
            hashlib.sha256(_icc_payload(destination.read_bytes())).hexdigest()
        )
        runs.append(monitored)
    workers = [row["worker"] for row in runs]
    walls = [row["wall_seconds"] for row in workers]
    peaks = [row["peak_process_tree_rss_bytes"] for row in runs]
    gates = {
        "input_exact": len({row["input_sha256"] for row in workers}) == 1,
        "output_exact": len({row["output_sha256"] for row in workers}) == 1,
        "decoded_exact": len(set(decoded_hashes)) == 1,
        "icc_exact": len(set(icc_hashes)) == 1 and icc_hashes[0] == hashlib.sha256(srgb_icc_profile()).hexdigest(),
        "maximum_wall": max(walls) <= contract["gates"]["maximum_wall_seconds"],
        "maximum_rss": max(peaks) <= contract["gates"]["maximum_process_tree_rss_bytes"],
        "wall_repeat": max(walls) / min(walls) <= contract["gates"]["maximum_repeat_wall_ratio"],
        "rss_repeat": max(peaks) / min(peaks) <= contract["gates"]["maximum_repeat_rss_ratio"],
        "worker_cleanup": all(row["stderr_empty"] and row["surviving_process_count"] == 0 for row in runs),
        "no_stage_residue": not list(output_dir.glob("*.stage-*")),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": workers[0]["input_sha256"],
        "output_sha256": workers[0]["output_sha256"],
        "output_bytes": workers[0]["output_bytes"],
        "decoded_rgb16_sha256": decoded_hashes[0],
        "icc_sha256": icc_hashes[0],
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
            "maximum_wall_seconds": max(walls),
            "maximum_peak_process_tree_rss_bytes": max(peaks),
            "wall_repeat_ratio": max(walls) / min(walls),
            "rss_repeat_ratio": max(peaks) / min(peaks),
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/u6_p8db_native_thomas_spatial_scale_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/eval/u6_p8db_native_thomas_spatial_scale_v1")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.dll is None or args.destination is None or args.result is None:
            parser.error("--worker requires --dll, --destination and --result")
        _worker(args.contract, args.dll, args.destination, args.result)
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
