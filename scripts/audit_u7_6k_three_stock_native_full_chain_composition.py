#!/usr/bin/env python3
"""Audit exact three-stock RGB composition with the U7.6J native kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_u7_6i_three_stock_native_pointwise import (
    _array_sha256,
    _prepare_source,
    _sha256,
    _timed,
)
from scripts.audit_u7_6j_three_stock_native_pointwise_exact import _unload
from scripts.pipeline_color_baseline import (
    apply_output_margin,
    compress_to_srgb_gamut_parallel,
    lab_to_rgb_no_clip,
    load_guardrail_config,
)
from src.color_engine.safe_lab import (
    apply_safe_lab_transform,
    apply_tone_rolloff,
    preserve_luma_detail,
    safe_lab_context_from_lab,
)
from src.eval.native_safe_lab_pointwise_v3 import (
    apply_native_safe_lab_pointwise_v3,
    build_native_safe_lab_pointwise_v3,
    load_native_safe_lab_pointwise_v3,
)
from src.inference.render_contract import atomic_write_json, load_render_profile
from src.inference.three_stock_look import resolve_three_stock_look_parameters

CONTRACT_SCHEMA = (
    "neuro-film.u7-6k-three-stock-native-full-chain-composition-contract.v1"
)
WORKER_SCHEMA = (
    "neuro-film.u7-6k-three-stock-native-full-chain-composition-worker.v1"
)
REPORT_SCHEMA = (
    "neuro-film.u7-6k-three-stock-native-full-chain-composition-result.v1"
)


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("U7.6K contract schema drift")
    for binding in payload["bindings"]:
        if _sha256(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"U7.6K binding drift: {binding['path']}")
    parent = payload["parent_positive"]
    if _sha256(ROOT / parent["evidence_path"]) != parent["evidence_sha256"]:
        raise ValueError("U7.6K parent evidence drift")
    source = payload["source"]
    if _sha256(ROOT / source["path"]) != source["sha256"]:
        raise ValueError("U7.6K source identity drift")
    return payload, hashlib.sha256(raw).hexdigest()


def _lab_to_output(
    source_lab: np.ndarray,
    styled_lab: np.ndarray,
    *,
    dither: float,
    output_margin: int,
    seed: int,
    gamut_workers: int,
) -> np.ndarray:
    compressed = compress_to_srgb_gamut_parallel(
        source_lab, styled_lab, workers=gamut_workers
    )
    result = lab_to_rgb_no_clip(compressed)
    if dither > 0.0:
        rng = np.random.default_rng(seed + 1009)
        noise = rng.uniform(-0.5, 0.5, size=result.shape).astype(np.float32)
        result = np.clip(result + noise * (dither / 255.0), 0.0, 1.0)
    result = apply_output_margin(result, output_margin)
    return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def _oracle_chain(
    source_lab: np.ndarray,
    *,
    context,
    destination_mean: np.ndarray,
    destination_std: np.ndarray,
    style_id: str,
    parameters: dict[str, Any],
    guard: dict[str, Any],
    seed: int,
    gamut_workers: int,
) -> np.ndarray:
    styled = apply_safe_lab_transform(
        source_lab,
        source_context=context,
        destination_mean=destination_mean,
        destination_std=destination_std,
        style=style_id,
        strength=float(parameters["strength"]),
        luma_strength=float(parameters["luma_strength"]),
        tone_rolloff=float(parameters["tone_rolloff"]),
        shadow_floor_l=float(parameters["shadow_floor_l"]),
        highlight_ceiling_l=float(parameters["highlight_ceiling_l"]),
        preserve_luma_detail_strength=float(parameters["preserve_luma_detail"]),
        chroma_curve_strength=float(parameters["chroma_curve_strength"]),
        neutral_protect=float(guard["neutral_protect"]),
        skin_protect=float(guard["skin_protect"]),
        max_chroma_gain=guard.get("max_chroma_gain"),
        max_chroma_boost=guard.get("max_chroma_boost"),
        max_chroma_absolute=guard.get("max_chroma_absolute"),
    )
    return _lab_to_output(
        source_lab,
        styled,
        dither=float(parameters["dither"]),
        output_margin=int(parameters["output_margin"]),
        seed=seed,
        gamut_workers=gamut_workers,
    )


def _native_chain(
    source_lab: np.ndarray,
    *,
    library,
    context,
    destination_mean: np.ndarray,
    destination_std: np.ndarray,
    parameters: dict[str, Any],
    guard: dict[str, Any],
    seed: int,
    thread_count: int,
    gamut_workers: int,
) -> np.ndarray:
    pointwise = apply_native_safe_lab_pointwise_v3(
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
        thread_count=thread_count,
    )
    detailed = preserve_luma_detail(
        source_lab, pointwise, float(parameters["preserve_luma_detail"])
    )
    styled = apply_tone_rolloff(
        detailed,
        float(parameters["tone_rolloff"]),
        float(parameters["shadow_floor_l"]),
        float(parameters["highlight_ceiling_l"]),
    )
    return _lab_to_output(
        source_lab,
        styled,
        dither=float(parameters["dither"]),
        output_margin=int(parameters["output_margin"]),
        seed=seed,
        gamut_workers=gamut_workers,
    )


def run_worker(contract_path: Path, output_path: Path) -> dict[str, Any]:
    contract, contract_sha256 = _load_contract(contract_path)
    source_lab = _prepare_source(contract)
    context = safe_lab_context_from_lab(source_lab)
    execution = contract["execution"]
    profile = load_render_profile(
        ROOT / "configs/render_profiles/safe_rich_v1.json", root=ROOT
    )
    stats = json.loads((ROOT / "configs/film_color_stats.json").read_text("utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build = build_native_safe_lab_pointwise_v3(
        root=ROOT, output_dir=output_path.parent / f"{output_path.stem}.build"
    )
    library = load_native_safe_lab_pointwise_v3(Path(build["dll_path"]))
    try:
        rows: list[dict[str, Any]] = []
        for style_id in contract["styles"]:
            stock_id = {
                "velvia_50": "fujifilm_velvia_50",
                "portra_400": "kodak_portra_400",
                "ektar_100": "kodak_ektar_100",
            }[style_id]
            resolved_style, parameters = resolve_three_stock_look_parameters(
                profile, film_stock_id=stock_id, look_amount=execution["look_amount"]
            )
            if resolved_style != style_id:
                raise ValueError("U7.6K stock/style mapping drift")
            destination_mean = np.asarray(
                stats["styles"][style_id]["mean"], dtype=np.float32
            )
            destination_std = np.asarray(
                stats["styles"][style_id]["std"], dtype=np.float32
            )
            guard = load_guardrail_config(
                ROOT / "configs/color_guardrails.json", style_id
            )
            common = {
                "context": context,
                "destination_mean": destination_mean,
                "destination_std": destination_std,
                "parameters": parameters,
                "guard": guard,
                "seed": int(execution["seed"]),
                "gamut_workers": int(execution["gamut_workers"]),
            }
            oracle, python_times = _timed(
                partial(_oracle_chain, source_lab, style_id=style_id, **common),
                int(execution["timed_repetitions_per_style"]),
            )
            native, native_times = _timed(
                partial(
                    _native_chain,
                    source_lab,
                    library=library,
                    thread_count=int(execution["thread_count"]),
                    **common,
                ),
                int(execution["timed_repetitions_per_style"]),
            )
            rows.append(
                {
                    "style_id": style_id,
                    "output_sha256": _array_sha256(native),
                    "oracle_sha256": _array_sha256(oracle),
                    "maximum_rgb_absolute_error": float(
                        np.max(np.abs(native - oracle))
                    ),
                    "finite_bounded": bool(
                        np.isfinite(native).all()
                        and np.all((native >= 0.0) & (native <= 1.0))
                    ),
                    "python_median_wall_seconds": float(np.median(python_times)),
                    "native_median_wall_seconds": float(np.median(native_times)),
                }
            )
    finally:
        _unload(library)
    worker = {
        "schema": WORKER_SCHEMA,
        "contract_sha256": contract_sha256,
        "source_lab_sha256": _array_sha256(source_lab),
        "build": {
            key: build[key]
            for key in ("source_sha256", "header_sha256", "dll_sha256", "toolchain")
        },
        "rows": rows,
    }
    atomic_write_json(output_path, worker)
    return worker


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
                    "maximum_rgb_absolute_error",
                    "finite_bounded",
                )
            }
            for row in report["workers"][0]["rows"]
        ],
        "gates": {
            key: value
            for key, value in report["gates"].items()
            if key not in {
                "three_style_native_chain_wall",
                "native_to_python_chain_wall_ratio",
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
        raise ValueError("U7.6K output must be create-only")
    contract, contract_sha256 = _load_contract(contract_path)
    scratch = Path(tempfile.mkdtemp(prefix="u7_6k_", dir=ROOT / "tmp"))
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
                capture_output=True,
                text=True,
                check=True,
            )
            if completed.stderr:
                raise RuntimeError("U7.6K worker wrote stderr")
            workers.append(json.loads(worker_path.read_text("utf-8")))
        maximum_error = max(
            row["maximum_rgb_absolute_error"]
            for worker in workers
            for row in worker["rows"]
        )
        output_identities = [
            tuple((row["style_id"], row["output_sha256"]) for row in worker["rows"])
            for worker in workers
        ]
        native_totals = [
            sum(row["native_median_wall_seconds"] for row in worker["rows"])
            for worker in workers
        ]
        python_totals = [
            sum(row["python_median_wall_seconds"] for row in worker["rows"])
            for worker in workers
        ]
        ratio = max(native_totals) / min(python_totals)
        gate_config = contract["gates"]
        gates = {
            "maximum_rgb_absolute_error": maximum_error
            <= gate_config["maximum_rgb_absolute_error"],
            "exact_output_sha256_per_style": all(
                row["output_sha256"] == row["oracle_sha256"]
                for worker in workers
                for row in worker["rows"]
            ),
            "finite_bounded_output": all(
                row["finite_bounded"] for worker in workers for row in worker["rows"]
            ),
            "cross_process_output_identity": len(set(output_identities)) == 1,
            "reproducible_dll_identity": len(
                {worker["build"]["dll_sha256"] for worker in workers}
            )
            == 1,
            "three_distinct_style_outputs": len(
                {row["output_sha256"] for row in workers[0]["rows"]}
            )
            == len(contract["styles"]),
            "three_style_native_chain_wall": max(native_totals)
            <= gate_config["maximum_three_style_native_chain_wall_seconds"],
            "native_to_python_chain_wall_ratio": ratio
            <= gate_config["maximum_native_to_python_chain_wall_ratio"],
        }
        passed = all(gates.values())
        report = {
            "schema": REPORT_SCHEMA,
            "experiment_id": contract["experiment_id"],
            "status": "PASS" if passed else "FAIL_CLOSED",
            "contract_sha256": contract_sha256,
            "source": contract["source"],
            "workers": workers,
            "observations": {
                "maximum_rgb_absolute_error": maximum_error,
                "native_three_style_wall_seconds": native_totals,
                "python_three_style_wall_seconds": python_totals,
                "conservative_native_to_python_chain_wall_ratio": ratio,
                "absolute_times_excluded_from_stable_identity": True,
            },
            "gates": gates,
            "decision": contract["decision_if_pass" if passed else "decision_if_fail"],
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
        default=(
            ROOT / "configs/u7_6k_three_stock_native_full_chain_composition_v1.json"
        ),
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
