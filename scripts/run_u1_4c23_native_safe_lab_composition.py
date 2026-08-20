#!/usr/bin/env python3
"""Run the frozen 24MP native-pointwise safe-Lab composition gate."""

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

from scripts.run_u1_4c22_native_safe_lab_pointwise import (
    _close_memmap,
    _hash_memmap,
    _sha256,
)
from src.color_engine.lab import linear_rgb_to_lab
from src.color_engine.safe_lab import (
    apply_safe_lab_transform,
    apply_tone_rolloff,
    preserve_luma_detail,
    safe_lab_context_from_lab,
)
from src.eval.native_rec2020_oklab_interior import (
    apply_native_rec2020_interior,
    build_native_rec2020_interior,
    load_native_rec2020_interior,
)
from src.eval.native_safe_lab_pointwise import (
    apply_native_safe_lab_pointwise,
    build_native_safe_lab_pointwise,
    load_native_safe_lab_pointwise,
)
from src.inference import atomic_write_json
from src.inference.romm_rec2020_velvia import load_profile
from src.preprocess.prophoto_icc import decode_prophoto_rgb16_to_linear_rec2020


def _load_contract(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != "neuro-film.u1-4c23-native-safe-lab-composition.v1":
        raise ValueError("C23 contract schema is invalid")
    for binding in payload["bindings"]:
        bound = root / binding["path"]
        if _sha256(bound) != binding["sha256"]:
            raise ValueError(f"C23 binding drift: {binding['path']}")
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


def evaluate(
    contract_path: Path, output_dir: Path, *, root: Path = ROOT
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("C23 output directory must be create-only")
    contract, contract_sha256 = _load_contract(contract_path, root=root)
    source = contract["source"]
    input_path = root / source["path"]
    profile_path = root / contract["profile"]["path"]
    if _sha256(input_path) != source["sha256"]:
        raise ValueError("C23 input identity drift")
    if _sha256(profile_path) != contract["profile"]["sha256"]:
        raise ValueError("C23 profile identity drift")
    output_dir.mkdir(parents=True)
    ingress_build = build_native_rec2020_interior(
        root=root, output_dir=output_dir / "build_ingress"
    )
    ingress_library = load_native_rec2020_interior(Path(ingress_build["dll_path"]))
    pointwise_build = build_native_safe_lab_pointwise(
        root=root, output_dir=output_dir / "build_pointwise"
    )
    pointwise_library = load_native_safe_lab_pointwise(
        Path(pointwise_build["dll_path"])
    )
    profile, profile_sha256 = load_profile(profile_path, root=root)
    execution = contract["execution"]
    gates = contract["gates"]
    row_chunk = int(execution["row_chunk"])
    halo = int(execution["style_halo_rows"])

    with tempfile.TemporaryDirectory(prefix="u1_4c23_", dir=output_dir) as temporary:
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
                raise ValueError("C23 decoded source contract drift")
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
            expanded_y0 = max(0, y0 - halo)
            expanded_y1 = min(shape[0], y1 + halo)
            source_tile = np.asarray(source_lab[expanded_y0:expanded_y1])
            styled = apply_safe_lab_transform(
                source_tile,
                source_context=context,
                destination_mean=destination_mean,
                destination_std=destination_std,
                style=style_id,
                strength=float(style["strength"]),
                luma_strength=float(style["luma_strength"]),
                tone_rolloff=float(style["tone_rolloff"]),
                shadow_floor_l=float(style["shadow_floor_l"]),
                highlight_ceiling_l=float(style["highlight_ceiling_l"]),
                preserve_luma_detail_strength=float(
                    style["preserve_luma_detail_strength"]
                ),
                chroma_curve_strength=float(style["chroma_curve_strength"]),
                neutral_protect=float(guard["neutral_protect"]),
                skin_protect=float(guard["skin_protect"]),
                max_chroma_gain=guard.get("max_chroma_gain"),
                max_chroma_boost=guard.get("max_chroma_boost"),
                max_chroma_absolute=guard.get("max_chroma_absolute"),
            )
            crop0 = y0 - expanded_y0
            oracle[y0:y1] = styled[crop0 : crop0 + y1 - y0]
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
            for y0 in range(0, shape[0], row_chunk):
                y1 = min(shape[0], y0 + row_chunk)
                expanded_y0 = max(0, y0 - halo)
                expanded_y1 = min(shape[0], y1 + halo)
                source_tile = np.ascontiguousarray(
                    source_lab[expanded_y0:expanded_y1]
                )
                pointwise = apply_native_safe_lab_pointwise(
                    pointwise_library,
                    source_tile,
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
                )
                detailed = preserve_luma_detail(
                    source_tile,
                    pointwise,
                    float(style["preserve_luma_detail_strength"]),
                )
                toned = apply_tone_rolloff(
                    detailed,
                    float(style["tone_rolloff"]),
                    float(style["shadow_floor_l"]),
                    float(style["highlight_ceiling_l"]),
                )
                crop0 = y0 - expanded_y0
                output[y0:y1] = toned[crop0 : crop0 + y1 - y0]
            wall = time.perf_counter() - started
            output.flush()
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
        del (
            decoded,
            mapped,
            mapped_scale,
            source_tile,
            styled,
            pointwise,
            detailed,
            toned,
            native_tile,
            output,
        )
        for array in (source_lab, oracle, *native_outputs):
            _close_memmap(array)
        native_outputs.clear()
        automatic_pass = bool(
            maximum_error <= gates["maximum_lab_absolute_error"]
            and finite
            and replay["output_exact"]
            and max(item["wall_seconds"] for item in run_facts)
            <= gates["maximum_composed_wall_seconds"]
        )
        result: dict[str, Any] = {
            "schema": "neuro-film.u1-4c23-native-safe-lab-composition-result.v1",
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
                "pointwise_dll_sha256": pointwise_build["dll_sha256"],
                "toolchain": pointwise_build["toolchain"],
            },
            "context": {
                "lab_mean": list(context.lab_mean),
                "lab_std": list(context.lab_std),
                "style_id": style_id,
                "spatial_detail": "retained-python-gaussian-sigma-1p1",
                "tone_rolloff": "retained-python-safe-lab-v1",
            },
            "correctness": correctness,
            "replay": replay,
            "resources": {
                "composed_wall_seconds": [item["wall_seconds"] for item in run_facts],
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
