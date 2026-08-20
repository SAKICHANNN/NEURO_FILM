#!/usr/bin/env python3
"""Run the frozen 24MP native Rec.2020 Lab source-compression gate."""

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

from src.color_engine.gamut import compress_source_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.safe_lab import (
    apply_safe_lab_transform,
    safe_lab_context_from_lab,
)
from src.eval.native_rec2020_lab_source_compress_v2 import (
    apply_native_rec2020_lab_compress_v2,
    build_native_rec2020_lab_compress_v2,
    load_native_rec2020_lab_compress_v2,
)
from src.eval.native_rec2020_oklab_interior import (
    apply_native_rec2020_interior,
    build_native_rec2020_interior,
    load_native_rec2020_interior,
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
    if payload.get("schema") != "neuro-film.u1-4c26-native-rec2020-lab-source-compress.v1":
        raise ValueError("C26 contract schema is invalid")
    for binding in payload["bindings"]:
        bound = root / binding["path"]
        if _sha256(bound) != binding["sha256"]:
            raise ValueError(f"C26 binding drift: {binding['path']}")
    return payload, hashlib.sha256(raw).hexdigest()


def _stable_id(payload: dict[str, Any]) -> str:
    stable = {
        "schema": payload["schema"],
        "experiment_id": payload["experiment_id"],
        "contract_sha256": payload["contract_sha256"],
        "input": payload["input"],
        "build": payload["build"],
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
        raise ValueError("C26 output directory must be create-only")
    contract, contract_sha256 = _load_contract(contract_path, root=root)
    source = contract["source"]
    input_path = root / source["path"]
    if _sha256(input_path) != source["sha256"]:
        raise ValueError("C26 input identity drift")
    profile_path = root / contract["profile"]["path"]
    if _sha256(profile_path) != contract["profile"]["sha256"]:
        raise ValueError("C26 profile identity drift")
    output_dir.mkdir(parents=True)
    ingress_build = build_native_rec2020_interior(
        root=root, output_dir=output_dir / "build_ingress"
    )
    ingress_library = load_native_rec2020_interior(Path(ingress_build["dll_path"]))
    compress_build = build_native_rec2020_lab_compress_v2(
        root=root, output_dir=output_dir / "build_compress"
    )
    compress_library = load_native_rec2020_lab_compress_v2(
        Path(compress_build["dll_path"])
    )
    profile, profile_sha256 = load_profile(profile_path, root=root)
    execution = contract["execution"]
    gates = contract["gates"]
    row_chunk = int(execution["row_chunk"])

    with tempfile.TemporaryDirectory(prefix="u1_4c26_", dir=output_dir) as temporary:
        scratch = Path(temporary)
        with tifffile.TiffFile(input_path) as document:
            if len(document.pages) != 1:
                raise ValueError("C26 input must contain one TIFF page")
            page = document.pages[0]
            profile_tag = page.tags.get(34675)
            embedded_profile = bytes(profile_tag.value) if profile_tag is not None else b""
            embedded_sha256 = hashlib.sha256(embedded_profile).hexdigest()
            shape = tuple(int(value) for value in page.shape)
            if (
                embedded_sha256 != source["embedded_icc_sha256"]
                or shape != tuple(source["shape"])
                or page.dtype != np.dtype(np.uint16)
            ):
                raise ValueError("C26 decoded source contract drift")
            encoded = np.memmap(
                scratch / "encoded.u16", mode="w+", dtype=np.uint16, shape=shape
            )
            page.asarray(out=encoded)

        mapped = np.memmap(
            scratch / "mapped.f32", mode="w+", dtype=np.float32, shape=shape
        )
        source_lab = np.memmap(
            scratch / "source_lab.f32", mode="w+", dtype=np.float32, shape=shape
        )
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            decoded_tile = decode_prophoto_rgb16_to_linear_rec2020(
                np.asarray(encoded[y0:y1]), embedded_profile
            )
            scale_tile = np.empty(decoded_tile.shape[:2], dtype=np.float32)
            apply_native_rec2020_interior(
                ingress_library,
                decoded_tile,
                softness=float(execution["ingress_softness"]),
                margin=float(execution["ingress_margin"]),
                iterations=int(execution["ingress_iterations"]),
                thread_count=int(execution["thread_count"]),
                output=mapped[y0:y1],
                chroma_scale=scale_tile,
            )
            source_lab[y0:y1] = linear_rgb_to_lab(
                np.asarray(mapped[y0:y1]), working_space="linear_rec2020"
            )
        mapped.flush()
        source_lab.flush()
        _close_memmap(encoded)

        context = safe_lab_context_from_lab(source_lab)
        assets = {binding["role"]: root / binding["path"] for binding in profile["assets"]}
        stats = json.loads(assets["style_statistics"].read_text(encoding="utf-8"))
        guard_payload = json.loads(assets["color_guardrails"].read_text(encoding="utf-8"))
        style = profile["style"]
        style_id = style["id"]
        guard = dict(guard_payload["defaults"])
        guard.update(guard_payload["styles"].get(style_id, {}))
        target_lab = np.memmap(
            scratch / "target_lab.f32", mode="w+", dtype=np.float32, shape=shape
        )
        halo = int(execution["style_halo_rows"])
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            expanded_y0 = max(0, y0 - halo)
            expanded_y1 = min(shape[0], y1 + halo)
            lab_tile = np.asarray(source_lab[expanded_y0:expanded_y1])
            styled = apply_safe_lab_transform(
                lab_tile,
                source_context=context,
                destination_mean=np.asarray(stats["styles"][style_id]["mean"], dtype=np.float32),
                destination_std=np.asarray(stats["styles"][style_id]["std"], dtype=np.float32),
                style=style_id,
                strength=float(style["strength"]),
                luma_strength=float(style["luma_strength"]),
                tone_rolloff=float(style["tone_rolloff"]),
                shadow_floor_l=float(style["shadow_floor_l"]),
                highlight_ceiling_l=float(style["highlight_ceiling_l"]),
                preserve_luma_detail_strength=float(style["preserve_luma_detail_strength"]),
                chroma_curve_strength=float(style["chroma_curve_strength"]),
                neutral_protect=float(guard["neutral_protect"]),
                skin_protect=float(guard["skin_protect"]),
                max_chroma_gain=guard.get("max_chroma_gain"),
                max_chroma_boost=guard.get("max_chroma_boost"),
                max_chroma_absolute=guard.get("max_chroma_absolute"),
            )
            crop0 = y0 - expanded_y0
            target_lab[y0:y1] = styled[crop0 : crop0 + y1 - y0]
        target_lab.flush()

        oracle_lab = np.memmap(
            scratch / "oracle_lab.f32", mode="w+", dtype=np.float32, shape=shape
        )
        oracle_rgb = np.memmap(
            scratch / "oracle_rgb.f32", mode="w+", dtype=np.float32, shape=shape
        )
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            oracle_tile = compress_source_to_working_gamut(
                np.asarray(source_lab[y0:y1]),
                np.asarray(target_lab[y0:y1]),
                working_space="linear_rec2020",
                iterations=int(execution["compression_iterations"]),
                tolerance=float(execution["tolerance"]),
            )
            oracle_lab[y0:y1] = oracle_tile
            oracle_rgb[y0:y1] = lab_to_linear_rgb(
                oracle_tile, working_space="linear_rec2020"
            )
        oracle_lab.flush()
        oracle_rgb.flush()

        run_facts: list[dict[str, Any]] = []
        native_labs: list[np.memmap] = []
        native_rgbs: list[np.memmap] = []
        native_scales: list[np.memmap] = []
        for run_index in range(int(execution["formal_runs"])):
            output_lab = np.memmap(
                scratch / f"native_lab_{run_index}.f32",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            output_rgb = np.memmap(
                scratch / f"native_rgb_{run_index}.f32",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            output_scale = np.memmap(
                scratch / f"native_scale_{run_index}.f32",
                mode="w+",
                dtype=np.float32,
                shape=shape[:2],
            )
            started = time.perf_counter()
            apply_native_rec2020_lab_compress_v2(
                compress_library,
                source_lab,
                target_lab,
                iterations=int(execution["compression_iterations"]),
                tolerance=float(execution["tolerance"]),
                thread_count=int(execution["thread_count"]),
                output_lab=output_lab,
                output_rgb=output_rgb,
                scale=output_scale,
            )
            wall = time.perf_counter() - started
            run_facts.append(
                {
                    "wall_seconds": wall,
                    "lab_sha256": _hash_memmap(output_lab),
                    "rgb_sha256": _hash_memmap(output_rgb),
                    "scale_sha256": _hash_memmap(output_scale),
                }
            )
            native_labs.append(output_lab)
            native_rgbs.append(output_rgb)
            native_scales.append(output_scale)

        maximum_lab_error = 0.0
        maximum_rgb_error = 0.0
        finite = True
        bounded = True
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            lab_tile = np.asarray(native_labs[0][y0:y1])
            rgb_tile = np.asarray(native_rgbs[0][y0:y1])
            maximum_lab_error = max(
                maximum_lab_error,
                float(np.max(np.abs(lab_tile - np.asarray(oracle_lab[y0:y1])))),
            )
            maximum_rgb_error = max(
                maximum_rgb_error,
                float(np.max(np.abs(rgb_tile - np.asarray(oracle_rgb[y0:y1])))),
            )
            finite = finite and bool(np.isfinite(lab_tile).all() and np.isfinite(rgb_tile).all())
            bounded = bounded and bool(
                np.all(rgb_tile >= -float(execution["tolerance"]))
                and np.all(rgb_tile <= 1.0 + float(execution["tolerance"]))
            )

        lab_hashes = {item["lab_sha256"] for item in run_facts}
        rgb_hashes = {item["rgb_sha256"] for item in run_facts}
        scale_hashes = {item["scale_sha256"] for item in run_facts}
        correctness = {
            "maximum_lab_absolute_error": maximum_lab_error,
            "maximum_rgb_absolute_error": maximum_rgb_error,
            "finite": finite,
            "bounded": bounded,
        }
        replay = {
            "formal_runs": len(run_facts),
            "lab_exact": len(lab_hashes) == 1,
            "rgb_exact": len(rgb_hashes) == 1,
            "scale_exact": len(scale_hashes) == 1,
            "lab_sha256": next(iter(lab_hashes)) if len(lab_hashes) == 1 else None,
            "rgb_sha256": next(iter(rgb_hashes)) if len(rgb_hashes) == 1 else None,
            "scale_sha256": next(iter(scale_hashes)) if len(scale_hashes) == 1 else None,
        }
        del decoded_tile, scale_tile, lab_tile, styled, oracle_tile, rgb_tile
        for array in (
            mapped,
            source_lab,
            target_lab,
            oracle_lab,
            oracle_rgb,
            *native_labs,
            *native_rgbs,
            *native_scales,
        ):
            _close_memmap(array)
        native_labs.clear()
        native_rgbs.clear()
        native_scales.clear()
        automatic_pass = bool(
            maximum_lab_error <= gates["maximum_lab_absolute_error"]
            and maximum_rgb_error <= gates["maximum_rgb_absolute_error"]
            and finite
            and bounded
            and replay["lab_exact"]
            and replay["rgb_exact"]
            and replay["scale_exact"]
            and max(item["wall_seconds"] for item in run_facts)
            <= gates["maximum_native_kernel_wall_seconds"]
        )
        result: dict[str, Any] = {
            "schema": "neuro-film.u1-4c26-native-rec2020-lab-source-compress-result.v1",
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
                "compress_source_sha256": compress_build["source_sha256"],
                "compress_header_sha256": compress_build["header_sha256"],
                "compress_dll_sha256": compress_build["dll_sha256"],
                "toolchain": compress_build["toolchain"],
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
