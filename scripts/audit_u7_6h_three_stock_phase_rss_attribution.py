#!/usr/bin/env python3
"""Attribute the unchanged 24MP three-stock product RSS peak by phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import normalized_icc_profile_sha256

CONFIG = ROOT / "configs/u7_6h_three_stock_phase_rss_attribution_v1.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/eval/u7_6h_three_stock_phase_rss_attribution_result.json"
)
SCRATCH = ROOT / "outputs/eval/u7_6h_three_stock_phase_rss_attribution_audit"
STYLES = ("velvia_50", "portra_400", "ektar_100")
STOCK_IDS = ("fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decoded_sha256(path: Path) -> str:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype.name != "uint16" or decoded.shape[2] != 3:
        raise RuntimeError("output is not a decoded uint16 RGB PNG")
    return hashlib.sha256(decoded[..., ::-1].tobytes(order="C")).hexdigest()


def _icc_fingerprint(path: Path) -> str:
    with Image.open(path) as image:
        profile = image.info.get("icc_profile")
        if not isinstance(profile, bytes):
            raise TypeError("output has no byte-valued ICC profile")
    return normalized_icc_profile_sha256(profile)


def _normalized_recipe_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["output"]["path"] = "OUTPUT"
    payload["output"]["sha256"] = "OUTPUT_SHA"
    payload["software"]["commit"] = "SOFTWARE_COMMIT"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class PhaseSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.phase = "input_decode_and_orchestration"
        self.peaks: dict[str, int] = {}
        self.samples: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @contextmanager
    def measure(self, phase: str):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        root = psutil.Process()
        while not self._stop.is_set():
            phase = self.phase
            try:
                processes = [root, *root.children(recursive=True)]
                rss = sum(
                    process.memory_info().rss
                    for process in processes
                    if process.is_running()
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            self.peaks[phase] = max(self.peaks.get(phase, 0), rss)
            self.samples[phase] = self.samples.get(phase, 0) + 1
            self._stop.wait(self.interval)


def run_worker(run_directory: Path) -> dict[str, Any]:
    import src.inference.three_stock_batch as batch_module
    import src.inference.three_stock_look as look_module

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    execution = config["execution"]
    sampler = PhaseSampler(execution["sample_interval_seconds"])
    original_context = look_module.build_safe_lab_source_context
    original_render = look_module.render_resolved_safe_lab_rgb
    original_save = batch_module.save_srgb16_png

    def measured_context(source):
        with sampler.measure("source_context"):
            return original_context(source)

    def measured_render(*args, **kwargs):
        style = str(kwargs["style"])
        with sampler.measure(f"render_{style}"):
            return original_render(*args, **kwargs)

    def measured_save(*args, **kwargs):
        with sampler.measure("encode_png16"):
            return original_save(*args, **kwargs)

    look_module.build_safe_lab_source_context = measured_context
    look_module.render_resolved_safe_lab_rgb = measured_render
    batch_module.save_srgb16_png = measured_save
    output_directory = run_directory / "output"
    sampler.start()
    started = time.perf_counter()
    try:
        manifest = batch_module.render_three_stock_batch_to_directory(
            ROOT / config["input"]["path"],
            output_directory,
            root=ROOT,
            profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
            statistics_path=ROOT / "configs/film_color_stats.json",
            guardrails_path=ROOT / "configs/color_guardrails.json",
            look_amount=execution["look_amount"],
            seed=execution["render_seed"],
            tile_size=execution["tile_size"],
            tile_workers=execution["tile_workers"],
            png_compression=execution["png_compression"],
        )
    finally:
        sampler.stop()
    rows = []
    for style in STYLES:
        output = output_directory / f"{style}.png"
        rows.append(
            {
                "style_id": style,
                "output_sha256": _sha256(output),
                "decoded_rgb16_sha256": _decoded_sha256(output),
                "icc_fingerprint_sha256": _icc_fingerprint(output),
                "normalized_recipe_sha256": _normalized_recipe_sha256(
                    output_directory / f"{style}.recipe.json"
                ),
            }
        )
    dominant_phase = max(sampler.peaks, key=sampler.peaks.__getitem__)
    return {
        "wall_seconds": time.perf_counter() - started,
        "phase_peak_process_tree_rss_bytes": sampler.peaks,
        "phase_sample_counts": sampler.samples,
        "dominant_phase": dominant_phase,
        "peak_process_tree_rss_bytes": sampler.peaks[dominant_phase],
        "rows": rows,
        "manifest_stock_ids": [row["film_stock_id"] for row in manifest["rows"]],
    }


def _run_fresh(index: int) -> dict[str, Any]:
    run_directory = SCRATCH / f"run_{index}"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--run-directory",
            str(run_directory),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        if process.returncode != 0:
            raise RuntimeError(f"worker failed: {process.stderr.strip()}")
        row = json.loads(process.stdout)
        row["stderr"] = process.stderr
        return row
    finally:
        shutil.rmtree(run_directory, ignore_errors=True)


def evaluate_rows(rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    def identities(key: str) -> set[tuple[str, ...]]:
        return {tuple(item[key] for item in row["rows"]) for row in rows}

    peaks = [row["peak_process_tree_rss_bytes"] for row in rows]
    peak_ratio = max(peaks) / min(peaks)
    residue = sum(1 for path in SCRATCH.rglob("*") if path.is_file())
    gate_results = {
        "three_encoded_outputs_repeat_exact": len(identities("output_sha256")) == 1,
        "three_decoded_sample_arrays_repeat_exact": len(
            identities("decoded_rgb16_sha256")
        )
        == 1,
        "three_icc_fingerprints_repeat_exact": len(
            identities("icc_fingerprint_sha256")
        )
        == 1,
        "three_normalized_recipe_semantics_repeat_exact": len(
            identities("normalized_recipe_sha256")
        )
        == 1,
        "ordered_stock_ids_exact": all(
            row["manifest_stock_ids"] == list(STOCK_IDS) for row in rows
        ),
        "dominant_phase_repeat_exact": len(
            {row["dominant_phase"] for row in rows}
        )
        == 1,
        "peak_process_tree_rss_repeat_ratio": peak_ratio
        <= gates["peak_process_tree_rss_repeat_ratio_max"],
        "owned_residue_count": residue == gates["owned_residue_count"],
    }
    return {
        "decision": "PASS" if all(gate_results.values()) else "FAIL_CLOSED",
        "gate_results": gate_results,
        "dominant_phase": rows[0]["dominant_phase"],
        "peak_process_tree_rss_repeat_ratio": peak_ratio,
        "owned_residue_count": residue,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.run_directory is None:
            parser.error("--worker requires --run-directory")
        print(json.dumps(run_worker(args.run_directory), sort_keys=True))
        return 0
    if args.run_directory is not None:
        parser.error("--run-directory requires --worker")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    decoded = cv2.imread(str(ROOT / config["input"]["path"]), cv2.IMREAD_UNCHANGED)
    if decoded is None or list(decoded.shape[:2]) != config["input"][
        "expected_dimensions"
    ]:
        raise RuntimeError("frozen input dimensions drifted")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    try:
        rows = [
            _run_fresh(index)
            for index in range(config["execution"]["fresh_process_count"])
        ]
        evaluation = evaluate_rows(rows, config["gates"])
        identity_payload = {
            "decision": evaluation["decision"],
            "gate_results": evaluation["gate_results"],
            "dominant_phase": evaluation["dominant_phase"],
            "rows": [
                {
                    "phase_peak_process_tree_rss_bytes": row[
                        "phase_peak_process_tree_rss_bytes"
                    ],
                    "dominant_phase": row["dominant_phase"],
                    "rows": row["rows"],
                    "manifest_stock_ids": row["manifest_stock_ids"],
                }
                for row in rows
            ],
        }
        identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report = {
            "schema_version": "neuro-film.u7-6h-three-stock-phase-rss-attribution-result.v1",
            "node_id": "U7.6H",
            "config_sha256": _sha256(CONFIG),
            "execution_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "runs": rows,
            **evaluation,
            "scientific_identity": identity,
            "claim_ceiling": config["claim_ceiling"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"decision": report["decision"], "scientific_identity": identity}))
        return 0 if report["decision"] == "PASS" else 1
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
