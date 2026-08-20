#!/usr/bin/env python3
"""Run the frozen 24MP native pointwise safe-Lab feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.safe_lab import (
    apply_chroma_curve,
    apply_color_guardrails,
    safe_lab_context_from_lab,
)
from src.eval.native_rec2020_oklab_interior import (
    apply_native_rec2020_interior,
    build_native_rec2020_interior,
    load_native_rec2020_interior,
)
from src.eval.native_safe_lab_pointwise_v2 import (
    apply_native_safe_lab_pointwise_v2,
    build_native_safe_lab_pointwise_v2,
    load_native_safe_lab_pointwise_v2,
)
from src.inference import atomic_write_json
from src.inference.romm_rec2020_velvia import load_profile
from src.preprocess.prophoto_icc import decode_prophoto_rgb16_to_linear_rec2020


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_memmap(array: np.memmap, row_chunk: int = 128) -> str:
    digest = hashlib.sha256()
    for y0 in range(0, array.shape[0], row_chunk):
        y1 = min(array.shape[0], y0 + row_chunk)
        digest.update(np.ascontiguousarray(array[y0:y1]).tobytes())
    return digest.hexdigest()


def _close_memmap(array: np.memmap) -> None:
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _load_contract(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != "neuro-film.u1-4c27-native-safe-lab-pointwise.v1":
        raise ValueError("C27 contract schema is invalid")
    for binding in payload["bindings"]:
        bound = root / binding["path"]
        if _sha256(bound) != binding["sha256"]:
            raise ValueError(f"C27 binding drift: {binding['path']}")
    return payload, hashlib.sha256(raw).hexdigest()


def _stable_id(payload: dict[str, Any]) -> str:
    stable = {
        "schema": payload["schema"],
        "experiment_id": payload["experiment_id"],
        "contract_sha256": payload["contract_sha256"],
        "input": payload["input"],
        "build": payload["build"],
        "context": payload["context"],
        "correctness": payload["correctness"],
        "replay": payload["replay"],
        "automatic_pass": payload["automatic_pass"],
        "decision": payload["decision"],
        "claim_ceiling": payload["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def evaluate(
    contract_path: Path, output_dir: Path, *, root: Path = ROOT
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("C27 output directory must be create-only")
    contract, contract_sha256 = _load_contract(contract_path, root=root)
    source = contract["source"]
    input_path = root / source["path"]
    if _sha256(input_path) != source["sha256"]:
        raise ValueError("C27 input identity drift")
    profile_path = root / contract["profile"]["path"]
    if _sha256(profile_path) != contract["profile"]["sha256"]:
        raise ValueError("C27 profile identity drift")
    output_dir.mkdir(parents=True)
    ingress_build = build_native_rec2020_interior(
        root=root, output_dir=output_dir / "build_ingress"
    )
    ingress_library = load_native_rec2020_interior(Path(ingress_build["dll_path"]))
    pointwise_build = build_native_safe_lab_pointwise_v2(
        root=root, output_dir=output_dir / "build_pointwise"
    )
    pointwise_library = load_native_safe_lab_pointwise_v2(
        Path(pointwise_build["dll_path"])
    )
    profile, profile_sha256 = load_profile(profile_path, root=root)
    execution = contract["execution"]
    gates = contract["gates"]
    row_chunk = int(execution["row_chunk"])

    with tempfile.TemporaryDirectory(prefix="u1_4c27_", dir=output_dir) as temporary:
        scratch = Path(temporary)
        with tifffile.TiffFile(input_path) as document:
            page = document.pages[0]
            profile_tag = page.tags.get(34675)
            embedded_profile = bytes(profile_tag.value) if profile_tag is not None else b""
            embedded_sha256 = hashlib.sha256(embedded_profile).hexdigest()
            shape = tuple(int(value) for value in page.shape)
            if (
                len(document.pages) != 1
                or embedded_sha256 != source["embedded_icc_sha256"]
                or shape != tuple(source["shape"])
                or page.dtype != np.dtype(np.uint16)
            ):
                raise ValueError("C27 decoded source contract drift")
            encoded = np.memmap(
                scratch / "encoded.u16", mode="w+", dtype=np.uint16, shape=shape
            )
            page.asarray(out=encoded)

        source_lab = np.memmap(
            scratch / "source_lab.f32", mode="w+", dtype=np.float32, shape=shape
        )
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            decoded = decode_prophoto_rgb16_to_linear_rec2020(
                np.asarray(encoded[y0:y1]), embedded_profile
            )
            mapped = np.empty_like(decoded)
            mapped_scale = np.empty(decoded.shape[:2], dtype=np.float32)
            apply_native_rec2020_interior(
                ingress_library,
                decoded,
                softness=float(execution["ingress_softness"]),
                margin=float(execution["ingress_margin"]),
                iterations=int(execution["ingress_iterations"]),
                thread_count=int(execution["thread_count"]),
                output=mapped,
                chroma_scale=mapped_scale,
            )
            source_lab[y0:y1] = linear_rgb_to_lab(
                mapped, working_space="linear_rec2020"
            )
        source_lab.flush()
        _close_memmap(encoded)
        context = safe_lab_context_from_lab(source_lab)
        source_mean = np.asarray(context.lab_mean, dtype=np.float32)
        source_std = np.asarray(context.lab_std, dtype=np.float32)
        assets = {binding["role"]: root / binding["path"] for binding in profile["assets"]}
        stats = json.loads(assets["style_statistics"].read_text(encoding="utf-8"))
        guard_payload = json.loads(assets["color_guardrails"].read_text(encoding="utf-8"))
        style = profile["style"]
        style_id = style["id"]
        destination_mean = np.asarray(
            stats["styles"][style_id]["mean"], dtype=np.float32
        )
        destination_std = np.asarray(
            stats["styles"][style_id]["std"], dtype=np.float32
        )
        guard = dict(guard_payload["defaults"])
        guard.update(guard_payload["styles"].get(style_id, {}))

        oracle = np.memmap(
            scratch / "oracle.f32", mode="w+", dtype=np.float32, shape=shape
        )
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            oracle[y0:y1] = _python_pointwise(
                np.asarray(source_lab[y0:y1]),
                source_mean=source_mean,
                source_std=source_std,
                destination_mean=destination_mean,
                destination_std=destination_std,
                style=style,
                guard=guard,
            )
        oracle.flush()

        run_facts: list[dict[str, Any]] = []
        native_outputs: list[np.memmap] = []
        for run_index in range(int(execution["formal_runs"])):
            output = np.memmap(
                scratch / f"native_{run_index}.f32",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            started = time.perf_counter()
            apply_native_safe_lab_pointwise_v2(
                pointwise_library,
                source_lab,
                source_context=context,
                destination_mean=destination_mean,
                destination_std=destination_std,
                strength=float(style["strength"]),
                luma_strength=float(style["luma_strength"]),
                chroma_curve_strength=float(style["chroma_curve_strength"]),
                neutral_protect=float(guard["neutral_protect"]),
                skin_protect=float(guard["skin_protect"]),
                max_chroma_gain=float(guard["max_chroma_gain"]),
                max_chroma_boost=float(guard["max_chroma_boost"]),
                max_chroma_absolute=guard.get("max_chroma_absolute"),
                thread_count=int(execution["thread_count"]),
                output=output,
            )
            wall = time.perf_counter() - started
            run_facts.append(
                {"wall_seconds": wall, "output_sha256": _hash_memmap(output)}
            )
            native_outputs.append(output)

        maximum_error = 0.0
        finite = True
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            native_tile = np.asarray(native_outputs[0][y0:y1])
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(native_tile - np.asarray(oracle[y0:y1])))),
            )
            finite = finite and bool(np.isfinite(native_tile).all())
        output_hashes = {item["output_sha256"] for item in run_facts}
        replay = {
            "formal_runs": len(run_facts),
            "output_exact": len(output_hashes) == 1,
            "output_sha256": next(iter(output_hashes))
            if len(output_hashes) == 1
            else None,
        }
        correctness = {"maximum_lab_absolute_error": maximum_error, "finite": finite}
        del decoded, mapped, mapped_scale, native_tile, output
        for array in (source_lab, oracle, *native_outputs):
            _close_memmap(array)
        native_outputs.clear()
        automatic_pass = bool(
            maximum_error <= gates["maximum_lab_absolute_error"]
            and finite
            and replay["output_exact"]
            and max(item["wall_seconds"] for item in run_facts)
            <= gates["maximum_native_kernel_wall_seconds"]
        )
        result: dict[str, Any] = {
            "schema": "neuro-film.u1-4c27-native-safe-lab-pointwise-result.v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "input": {
                "sha256": source["sha256"],
                "shape": list(shape),
                "pixel_count": shape[0] * shape[1],
                "embedded_icc_sha256": embedded_sha256,
                "profile_sha256": profile_sha256,
            },
            "build": {
                "ingress_dll_sha256": ingress_build["dll_sha256"],
                "pointwise_source_sha256": pointwise_build["source_sha256"],
                "pointwise_header_sha256": pointwise_build["header_sha256"],
                "pointwise_dll_sha256": pointwise_build["dll_sha256"],
                "toolchain": pointwise_build["toolchain"],
            },
            "context": {
                "lab_mean": list(context.lab_mean),
                "lab_std": list(context.lab_std),
                "destination_mean": destination_mean.tolist(),
                "destination_std": destination_std.tolist(),
                "style_id": style_id,
            },
            "correctness": correctness,
            "replay": replay,
            "resources": {
                "native_kernel_wall_seconds": [item["wall_seconds"] for item in run_facts],
                "thread_count": execution["thread_count"],
                "absolute_times_excluded_from_stable_id": True,
            },
            "automatic_pass": automatic_pass,
            "decision": contract[
                "decision_if_pass" if automatic_pass else "decision_if_fail"
            ],
            "claim_ceiling": contract["claim_ceiling"],
            "production_default_changed": False,
        }
        result["stable_evidence_id"] = _stable_id(result)
        atomic_write_json(output_dir / "report.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.contract, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
