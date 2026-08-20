#!/usr/bin/env python3
"""Run the frozen 24MP native Rec.2020 OKLab ingress feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from src.inference.atomic_json import atomic_write_json

from src.color_engine.oklab_analytical_interior import (
    analytical_oklab_interior_rec2020,
)
from src.eval.native_rec2020_oklab_interior import (
    apply_native_rec2020_interior,
    build_native_rec2020_interior,
    load_native_rec2020_interior,
)
from src.preprocess.prophoto_icc import decode_prophoto_rgb16_to_linear_rec2020

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != "neuro-film.u1-4c20-native-rec2020-oklab-ingress.v1":
        raise ValueError("C20 contract schema drift")
    for binding in payload["bindings"]:
        bound = root / binding["path"]
        if not bound.is_file() or _sha256(bound) != binding["sha256"]:
            raise ValueError(f"C20 binding drift: {binding['path']}")
    return payload, hashlib.sha256(raw).hexdigest()


def _hash_memmap(value: np.memmap) -> str:
    value.flush()
    path = Path(value.filename)
    return _sha256(path)


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
        raise ValueError("C20 output directory must be create-only")
    contract, contract_sha256 = _load_contract(contract_path, root=root)
    source = contract["source"]
    input_path = root / source["path"]
    if _sha256(input_path) != source["sha256"]:
        raise ValueError("C20 input identity drift")
    output_dir.mkdir(parents=True)
    build = build_native_rec2020_interior(root=root, output_dir=output_dir / "build")
    library = load_native_rec2020_interior(Path(build["dll_path"]))
    execution = contract["execution"]
    gates = contract["gates"]
    row_chunk = int(execution["oracle_row_chunk"])

    with tempfile.TemporaryDirectory(prefix="u1_4c20_", dir=output_dir) as temporary:
        scratch = Path(temporary)
        with tifffile.TiffFile(input_path) as document:
            if len(document.pages) != 1:
                raise ValueError("C20 input must contain one TIFF page")
            page = document.pages[0]
            profile_tag = page.tags.get(34675)
            profile = bytes(profile_tag.value) if profile_tag is not None else b""
            profile_sha256 = hashlib.sha256(profile).hexdigest()
            shape = tuple(int(value) for value in page.shape)
            if (
                profile_sha256 != source["embedded_icc_sha256"]
                or shape != tuple(source["shape"])
                or page.dtype != np.dtype(np.uint16)
            ):
                raise ValueError("C20 decoded source contract drift")
            encoded = np.memmap(
                scratch / "encoded.u16", mode="w+", dtype=np.uint16, shape=shape
            )
            page.asarray(out=encoded)

        decoded = np.memmap(
            scratch / "decoded.f32", mode="w+", dtype=np.float32, shape=shape
        )
        oracle = np.memmap(
            scratch / "oracle.f32", mode="w+", dtype=np.float32, shape=shape
        )
        oracle_scale = np.memmap(
            scratch / "oracle_scale.f32",
            mode="w+",
            dtype=np.float32,
            shape=shape[:2],
        )
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            tile = decode_prophoto_rgb16_to_linear_rec2020(
                np.asarray(encoded[y0:y1]), profile
            )
            decoded[y0:y1] = tile
            mapped, scale = analytical_oklab_interior_rec2020(tile)
            oracle[y0:y1] = mapped
            oracle_scale[y0:y1] = scale
        for value in (decoded, oracle, oracle_scale):
            value.flush()
        del encoded

        run_facts: list[dict[str, Any]] = []
        native_outputs: list[np.memmap] = []
        native_scales: list[np.memmap] = []
        for run_index in range(int(execution["formal_runs"])):
            output = np.memmap(
                scratch / f"native_{run_index}.f32",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            scale = np.memmap(
                scratch / f"native_scale_{run_index}.f32",
                mode="w+",
                dtype=np.float32,
                shape=shape[:2],
            )
            started = time.perf_counter()
            apply_native_rec2020_interior(
                library,
                decoded,
                softness=float(execution["softness"]),
                margin=float(execution["margin"]),
                iterations=int(execution["iterations"]),
                thread_count=int(execution["thread_count"]),
                output=output,
                chroma_scale=scale,
            )
            wall = time.perf_counter() - started
            run_facts.append(
                {
                    "wall_seconds": wall,
                    "output_sha256": _hash_memmap(output),
                    "scale_sha256": _hash_memmap(scale),
                }
            )
            native_outputs.append(output)
            native_scales.append(scale)

        maximum_output_error = 0.0
        maximum_scale_error = 0.0
        in_gamut_exact = True
        finite = True
        for y0 in range(0, shape[0], row_chunk):
            y1 = min(shape[0], y0 + row_chunk)
            source_tile = np.asarray(decoded[y0:y1])
            output_tile = np.asarray(native_outputs[0][y0:y1])
            scale_tile = np.asarray(native_scales[0][y0:y1])
            maximum_output_error = max(
                maximum_output_error,
                float(np.max(np.abs(output_tile - np.asarray(oracle[y0:y1])))),
            )
            maximum_scale_error = max(
                maximum_scale_error,
                float(np.max(np.abs(scale_tile - np.asarray(oracle_scale[y0:y1])))),
            )
            in_gamut = np.all((source_tile >= 0.0) & (source_tile <= 1.0), axis=2)
            in_gamut_exact = in_gamut_exact and np.array_equal(
                output_tile[in_gamut], source_tile[in_gamut]
            )
            finite = finite and bool(np.isfinite(output_tile).all())

        output_hashes = {item["output_sha256"] for item in run_facts}
        scale_hashes = {item["scale_sha256"] for item in run_facts}
        correctness = {
            "maximum_output_absolute_error": maximum_output_error,
            "maximum_chroma_scale_absolute_error": maximum_scale_error,
            "in_gamut_pixels_bit_exact": in_gamut_exact,
            "finite": finite,
        }
        replay = {
            "formal_runs": len(run_facts),
            "output_exact": len(output_hashes) == 1,
            "scale_exact": len(scale_hashes) == 1,
            "output_sha256": next(iter(output_hashes))
            if len(output_hashes) == 1
            else None,
            "scale_sha256": next(iter(scale_hashes))
            if len(scale_hashes) == 1
            else None,
        }
        automatic_pass = bool(
            maximum_output_error <= gates["maximum_output_absolute_error"]
            and maximum_scale_error <= gates["maximum_chroma_scale_absolute_error"]
            and in_gamut_exact
            and finite
            and replay["output_exact"]
            and replay["scale_exact"]
            and max(item["wall_seconds"] for item in run_facts)
            <= gates["maximum_native_kernel_wall_seconds"]
        )
        result: dict[str, Any] = {
            "schema": "neuro-film.u1-4c20-native-rec2020-oklab-ingress-result.v1",
            "experiment_id": contract["experiment_id"],
            "contract_sha256": contract_sha256,
            "input": {
                "sha256": source["sha256"],
                "shape": list(shape),
                "pixel_count": shape[0] * shape[1],
                "embedded_icc_sha256": profile_sha256,
            },
            "build": {
                key: build[key]
                for key in ("toolchain", "source_sha256", "header_sha256", "dll_sha256")
            },
            "correctness": correctness,
            "replay": replay,
            "resources": {
                "native_kernel_wall_seconds": [
                    item["wall_seconds"] for item in run_facts
                ],
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
