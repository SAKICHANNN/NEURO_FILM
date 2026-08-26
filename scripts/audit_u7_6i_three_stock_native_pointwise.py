#!/usr/bin/env python3
"""Audit the unchanged native safe-Lab pointwise subset on three looks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_guardrail_config
from src.color_engine.safe_lab import (
    apply_chroma_curve,
    apply_color_guardrails,
    safe_lab_context_from_lab,
)
from src.eval.native_safe_lab_pointwise_v2 import (
    apply_native_safe_lab_pointwise_v2,
    build_native_safe_lab_pointwise_v2,
    load_native_safe_lab_pointwise_v2,
)
from src.inference.render_contract import atomic_write_json, load_render_profile
from src.inference.three_stock_look import resolve_three_stock_look_parameters
from src.preprocess import load_working_image, working_image_to_srgb_float

CONTRACT_SCHEMA = "neuro-film.u7-6i-three-stock-native-pointwise-contract.v1"
REPORT_SCHEMA = "neuro-film.u7-6i-three-stock-native-pointwise-result.v1"
WORKER_SCHEMA = "neuro-film.u7-6i-three-stock-native-pointwise-worker.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("U7.6I contract schema drift")
    for binding in payload["bindings"]:
        if _sha256(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"U7.6I binding drift: {binding['path']}")
    source = payload["source"]
    if _sha256(ROOT / source["path"]) != source["sha256"]:
        raise ValueError("U7.6I source identity drift")
    return payload, hashlib.sha256(raw).hexdigest()


def _python_pointwise(
    source: np.ndarray,
    *,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    destination_mean: np.ndarray,
    destination_std: np.ndarray,
    style: dict[str, Any],
    guard: dict[str, Any],
) -> np.ndarray:
    transferred = (
        (source - source_mean) / source_std * destination_std + destination_mean
    )
    output = source.copy()
    output[..., 0] = source[..., 0] + float(style["luma_strength"]) * float(
        style["strength"]
    ) * (transferred[..., 0] - source[..., 0])
    output[..., 1:] = source[..., 1:] + float(style["strength"]) * (
        transferred[..., 1:] - source[..., 1:]
    )
    output = apply_chroma_curve(
        source, output, float(style["chroma_curve_strength"])
    )
    return apply_color_guardrails(
        source,
        output,
        neutral_protect=float(guard["neutral_protect"]),
        skin_protect=float(guard["skin_protect"]),
        max_chroma_gain=guard.get("max_chroma_gain"),
        max_chroma_boost=guard.get("max_chroma_boost"),
        max_chroma_absolute=guard.get("max_chroma_absolute"),
    )


def _prepare_source(contract: dict[str, Any]) -> np.ndarray:
    source = contract["source"]
    working = load_working_image(ROOT / source["path"])
    if (working.pixels.shape[1], working.pixels.shape[0]) != (
        source["source_width"],
        source["source_height"],
    ):
        raise ValueError("U7.6I decoded source geometry drift")
    resized = np.ascontiguousarray(
        cv2.resize(
            working.pixels,
            (source["preview_width"], source["preview_height"]),
            interpolation=cv2.INTER_AREA,
        ),
        dtype=np.float32,
    )
    working = replace(working, pixels=resized)
    encoded = working_image_to_srgb_float(working)
    return np.ascontiguousarray(rgb2lab(encoded), dtype=np.float32)


def _timed(callable_: Any, repetitions: int) -> tuple[np.ndarray, list[float]]:
    callable_()
    values: list[float] = []
    output: np.ndarray | None = None
    for _ in range(repetitions):
        started = time.perf_counter()
        output = callable_()
        values.append(time.perf_counter() - started)
    assert output is not None
    return output, values


def run_worker(contract_path: Path, output_path: Path) -> dict[str, Any]:
    contract, contract_sha256 = _load_contract(contract_path)
    execution = contract["execution"]
    source_lab = _prepare_source(contract)
    if source_lab.shape != (
        contract["source"]["preview_height"],
        contract["source"]["preview_width"],
        3,
    ):
        raise ValueError("U7.6I prepared source geometry drift")
    context = safe_lab_context_from_lab(source_lab)
    source_mean = np.asarray(context.lab_mean, dtype=np.float32)
    source_std = np.asarray(context.lab_std, dtype=np.float32)
    profile = load_render_profile(
        ROOT / "configs/render_profiles/safe_rich_v1.json", root=ROOT
    )
    stats = json.loads((ROOT / "configs/film_color_stats.json").read_text("utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_dir = output_path.parent / f"{output_path.stem}.build"
    build = build_native_safe_lab_pointwise_v2(root=ROOT, output_dir=build_dir)
    library = load_native_safe_lab_pointwise_v2(Path(build["dll_path"]))
    rows: list[dict[str, Any]] = []
    for style_id in contract["styles"]:
        stock_id = {
            "velvia_50": "fujifilm_velvia_50",
            "portra_400": "kodak_portra_400",
            "ektar_100": "kodak_ektar_100",
        }[style_id]
        resolved_style, parameters = resolve_three_stock_look_parameters(
            profile,
            film_stock_id=stock_id,
            look_amount=execution["look_amount"],
        )
        if resolved_style != style_id:
            raise ValueError("U7.6I stock/style mapping drift")
        destination_mean = np.asarray(
            stats["styles"][style_id]["mean"], dtype=np.float32
        )
        destination_std = np.asarray(
            stats["styles"][style_id]["std"], dtype=np.float32
        )
        guard = load_guardrail_config(
            ROOT / "configs/color_guardrails.json", style_id
        )

        python_call = partial(
            _python_pointwise,
            source_lab,
            source_mean=source_mean,
            source_std=source_std,
            destination_mean=destination_mean,
            destination_std=destination_std,
            style=parameters,
            guard=guard,
        )

        native_buffer = np.empty_like(source_lab)

        native_call = partial(
            apply_native_safe_lab_pointwise_v2,
            library,
            source_lab,
            source_context=context,
            destination_mean=destination_mean,
            destination_std=destination_std,
            strength=float(parameters["strength"]),
            luma_strength=float(parameters["luma_strength"]),
            chroma_curve_strength=float(parameters["chroma_curve_strength"]),
            neutral_protect=float(guard["neutral_protect"]),
            skin_protect=float(guard["skin_protect"]),
            max_chroma_gain=float(guard["max_chroma_gain"]),
            max_chroma_boost=float(guard["max_chroma_boost"]),
            max_chroma_absolute=guard.get("max_chroma_absolute"),
            thread_count=int(execution["thread_count"]),
            output=native_buffer,
        )

        oracle, python_times = _timed(
            python_call, int(execution["timed_repetitions_per_style"])
        )
        native, native_times = _timed(
            native_call, int(execution["timed_repetitions_per_style"])
        )
        rows.append(
            {
                "style_id": style_id,
                "output_sha256": _array_sha256(native),
                "oracle_sha256": _array_sha256(oracle),
                "maximum_lab_absolute_error": float(
                    np.max(np.abs(native - oracle))
                ),
                "finite": bool(np.isfinite(native).all()),
                "python_wall_seconds": python_times,
                "native_wall_seconds": native_times,
                "python_median_wall_seconds": float(np.median(python_times)),
                "native_median_wall_seconds": float(np.median(native_times)),
            }
        )
    report = {
        "schema": WORKER_SCHEMA,
        "contract_sha256": contract_sha256,
        "source_lab_sha256": _array_sha256(source_lab),
        "source_context": {
            "lab_mean": list(context.lab_mean),
            "lab_std": list(context.lab_std),
        },
        "build": {
            "source_sha256": build["source_sha256"],
            "header_sha256": build["header_sha256"],
            "dll_sha256": build["dll_sha256"],
            "toolchain": build["toolchain"],
        },
        "rows": rows,
    }
    atomic_write_json(output_path, report)
    return report


def _stable_id(report: dict[str, Any]) -> str:
    stable = {
        "schema": report["schema"],
        "experiment_id": report["experiment_id"],
        "contract_sha256": report["contract_sha256"],
        "source": report["source"],
        "rows": [
            {
                key: row[key]
                for key in (
                    "style_id",
                    "output_sha256",
                    "oracle_sha256",
                    "maximum_lab_absolute_error",
                    "finite",
                )
            }
            for row in report["workers"][0]["rows"]
        ],
        "gates": {
            key: value
            for key, value in report["gates"].items()
            if key not in {
                "three_style_native_wall",
                "native_to_python_wall_ratio",
            }
        },
        "decision": report["decision"],
        "claim_ceiling": report["claim_ceiling"],
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate(contract_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError("U7.6I output must be create-only")
    contract, contract_sha256 = _load_contract(contract_path)
    scratch = Path(tempfile.mkdtemp(prefix="u7_6i_", dir=ROOT / "tmp"))
    try:
        workers: list[dict[str, Any]] = []
        for index in range(int(contract["execution"]["formal_processes"])):
            worker_path = scratch / f"worker_{index}.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(contract_path),
                    "--worker-output",
                    str(worker_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stderr:
                raise RuntimeError("U7.6I worker wrote stderr")
            workers.append(json.loads(worker_path.read_text("utf-8")))

        output_identities = [
            tuple((row["style_id"], row["output_sha256"]) for row in worker["rows"])
            for worker in workers
        ]
        maximum_error = max(
            row["maximum_lab_absolute_error"]
            for worker in workers
            for row in worker["rows"]
        )
        all_finite = all(
            row["finite"] for worker in workers for row in worker["rows"]
        )
        native_totals = [
            sum(row["native_median_wall_seconds"] for row in worker["rows"])
            for worker in workers
        ]
        python_totals = [
            sum(row["python_median_wall_seconds"] for row in worker["rows"])
            for worker in workers
        ]
        conservative_ratio = max(native_totals) / min(python_totals)
        gates_config = contract["gates"]
        gates = {
            "maximum_lab_absolute_error": maximum_error
            <= gates_config["maximum_lab_absolute_error"],
            "finite_output": all_finite,
            "cross_process_output_identity": len(set(output_identities)) == 1,
            "three_distinct_style_outputs": len(
                {row["output_sha256"] for row in workers[0]["rows"]}
            )
            == len(contract["styles"]),
            "three_style_native_wall": max(native_totals)
            <= gates_config["maximum_three_style_native_pointwise_wall_seconds"],
            "native_to_python_wall_ratio": conservative_ratio
            <= gates_config["maximum_native_to_python_pointwise_wall_ratio"],
        }
        automatic_pass = all(gates.values())
        report = {
            "schema": REPORT_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "status": "PASS" if automatic_pass else "FAIL_CLOSED",
            "contract_sha256": contract_sha256,
            "source": contract["source"],
            "workers": workers,
            "observations": {
                "maximum_lab_absolute_error": maximum_error,
                "native_three_style_wall_seconds": native_totals,
                "python_three_style_wall_seconds": python_totals,
                "conservative_native_to_python_wall_ratio": conservative_ratio,
                "absolute_times_excluded_from_stable_identity": True,
            },
            "gates": gates,
            "decision": contract[
                "decision_if_pass" if automatic_pass else "decision_if_fail"
            ],
            "claim_ceiling": contract["claim_ceiling"],
            "production_default_changed": False,
        }
        report["stable_evidence_id"] = _stable_id(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path, report)
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_6i_three_stock_native_pointwise_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.worker_output is None):
        parser.error("choose exactly one of --output or --worker-output")
    if args.worker_output is not None:
        run_worker(args.config, args.worker_output)
        return 0
    report = evaluate(args.config, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
