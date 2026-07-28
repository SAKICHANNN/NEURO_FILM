"""U6.P7G resolution and partition invariance for the P7F challenger."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab
from skimage.transform import resize

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import sha256_file
from src.eval.physical_neutral_gauged_chain import (
    _render_arms,
    apply_gauge_to_intermediate,
    validate_contract as validate_p7f_contract,
)
from src.eval.physical_virtual_scan_sampling import (
    _render_physical,
    compile_virtual_scan_profile,
)
from src.film_physics import required_spatial_response_halo


SCHEMA = "neuro_film.u6_p7g_neutral_gauged_invariance_contract.v1"


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: dict[str, Any]) -> tuple[Any, Any]:
    if (
        config.get("schema") != SCHEMA
        or config["execution"].get("post_result_retuning_allowed")
        or config["selected_candidate_id"] != "gauged_spatial_4000"
    ):
        raise ValueError("unsupported U6.P7G contract")
    decision = _load_exact_json(
        root,
        config["parent_decision"],
        config["parent_decision_sha256"],
    )
    parent = _load_exact_json(
        root,
        config["parent_contract"],
        config["parent_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P7G")
        or decision["production_default_changed"]
        or decision["visual_result"]["decision"] != "complete_pass"
        or int(parent["sampling_dpi"])
        != int(config["reference_sampling_dpi"])
    ):
        raise ValueError("U6.P7G parent evidence drift")
    return validate_p7f_contract(root, parent)


def _validate_encoded(source: np.ndarray) -> np.ndarray:
    encoded = np.asarray(source, dtype=np.float64)
    if (
        encoded.ndim != 3
        or encoded.shape[-1] != 3
        or not np.all(np.isfinite(encoded))
        or np.any(encoded < 0.0)
        or np.any(encoded > 1.0)
    ):
        raise ValueError("invariance input must be finite encoded HxWx3")
    return encoded


def render_challenger(
    source: np.ndarray,
    runtime: Any,
    gauge: Any,
    *,
    sampling_dpi: int,
) -> np.ndarray:
    return _render_arms(
        _validate_encoded(source),
        runtime,
        gauge,
        sampling_dpi=sampling_dpi,
    )["gauged_spatial_4000"]


def render_challenger_row_tiled(
    source: np.ndarray,
    runtime: Any,
    gauge: Any,
    *,
    sampling_dpi: int,
    tile_rows: int,
    order: str = "forward",
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Render exact row cores with one frame-level AO6 source context."""

    encoded = _validate_encoded(source)
    if (
        isinstance(tile_rows, bool)
        or not isinstance(tile_rows, int)
        or tile_rows <= 0
        or order not in {"forward", "reverse"}
    ):
        raise ValueError("invalid row partition")
    compiled = replace(
        runtime,
        profile=compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=sampling_dpi
        ),
    )
    halo = required_spatial_response_halo(compiled.profile)
    linear = encoded_srgb_to_linear(encoded)
    apply_colour = runtime.build_source_context_colour(encoded)
    ranges = [
        (y0, min(encoded.shape[0], y0 + tile_rows))
        for y0 in range(0, encoded.shape[0], tile_rows)
    ]
    if order == "reverse":
        ranges.reverse()
    gauged_encoded = np.empty_like(encoded, dtype=np.float64)
    seams: list[int] = []
    for y0, y1 in ranges:
        source_y0 = max(0, y0 - halo)
        source_y1 = min(encoded.shape[0], y1 + halo)
        physical = _render_physical(
            linear[source_y0:source_y1],
            compiled,
        )
        core = physical[y0 - source_y0 : y1 - source_y0]
        gauged = apply_gauge_to_intermediate(core, gauge)
        gauged_encoded[y0:y1] = linear_srgb_to_encoded(gauged)
        if 0 < y0 < encoded.shape[0]:
            seams.append(y0)
    if (
        not np.all(np.isfinite(gauged_encoded))
        or np.any(gauged_encoded < 0.0)
        or np.any(gauged_encoded > 1.0)
    ):
        raise RuntimeError("tiled physical intermediate left encoded RGB")
    output = apply_colour(gauged_encoded)
    if (
        output.shape != encoded.shape
        or not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
    ):
        raise RuntimeError("tiled challenger left encoded RGB")
    return output, tuple(sorted(seams))


def _resize(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resize(
        np.asarray(values, dtype=np.float64),
        (*shape, 3),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float64, copy=False)


def _edge_energy(values: np.ndarray) -> float:
    luma = (
        np.asarray(values, dtype=np.float64)
        @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    )
    dy = np.diff(luma, axis=0)
    dx = np.diff(luma, axis=1)
    return float(
        0.5 * (np.mean(np.square(dy)) + np.mean(np.square(dx)))
    )


def _resolution_metrics(
    direct: np.ndarray, high_reference: np.ndarray
) -> dict[str, float]:
    first = rgb2lab(np.clip(direct, 0.0, 1.0))
    second = rgb2lab(np.clip(high_reference, 0.0, 1.0))
    delta = np.linalg.norm(first - second, axis=-1)
    denominator = max(_edge_energy(high_reference), 1e-15)
    return {
        "median_delta_e76": float(np.median(delta)),
        "p95_delta_e76": float(np.percentile(delta, 95.0)),
        "maximum_delta_e76": float(np.max(delta)),
        "edge_energy_ratio": _edge_energy(direct) / denominator,
    }


def _tile_metrics(
    reference: np.ndarray,
    tiled: np.ndarray,
    seams: Iterable[int],
) -> dict[str, float]:
    difference = np.abs(reference - tiled)
    seam_rows = sorted(
        {
            row
            for seam in seams
            for row in (seam - 1, seam)
            if 0 <= row < reference.shape[0]
        }
    )
    quantized_reference = np.rint(reference * 255.0).astype(np.uint8)
    quantized_tiled = np.rint(tiled * 255.0).astype(np.uint8)
    return {
        "maximum_float_absolute_error": float(np.max(difference)),
        "quantized_mismatch_fraction": float(
            np.mean(quantized_reference != quantized_tiled)
        ),
        "maximum_seam_absolute_error": (
            float(np.max(difference[seam_rows]))
            if seam_rows
            else 0.0
        ),
    }


def _analytic_chart(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.mgrid[
        0.0 : 1.0 : complex(height),
        0.0 : 1.0 : complex(width),
    ]
    checker = ((np.floor(x * 24) + np.floor(y * 16)) % 2) * 0.16
    spot = np.exp(-((x - 0.72) ** 2 + (y - 0.33) ** 2) / 0.0008)
    edge = (x >= 0.44).astype(np.float64)
    return np.clip(
        np.stack(
            (
                0.05 + 0.72 * x + checker + 0.18 * spot,
                0.04 + 0.68 * y + 0.11 * checker + 0.12 * edge,
                0.06 + 0.34 * x + 0.31 * y + 0.16 * spot,
            ),
            axis=-1,
        ),
        0.0,
        1.0,
    )


def _bounded_real_source(
    root: Path,
    runtime: Any,
    sample_id: str,
    *,
    max_long_edge: int,
) -> np.ndarray:
    row = runtime.source_rows[sample_id]
    path = root / row["decoded_path"]
    if sha256_file(path) != row["decoded_sha256"]:
        raise ValueError(f"source hash drift: {sample_id}")
    with Image.open(path) as image:
        source = (
            np.asarray(
                ImageOps.exif_transpose(image).convert("RGB"),
                dtype=np.float64,
            )
            / 255.0
        )
    scale = min(1.0, max_long_edge / max(source.shape[:2]))
    if scale < 1.0:
        shape = (
            max(1, int(round(source.shape[0] * scale))),
            max(1, int(round(source.shape[1] * scale))),
        )
        source = _resize(source, shape)
    return source


def evaluate_invariance(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    runtime, gauge = validate_contract(root, config)
    dpi = int(config["reference_sampling_dpi"])
    tile = config["tile_audit"]
    synthetic = np.random.default_rng(int(tile["synthetic_seed"])).random(
        tuple(int(value) for value in tile["synthetic_shape"]) + (3,)
    )
    tile_sources = {"synthetic": synthetic}
    for sample_id in tile["real_ids"]:
        tile_sources[sample_id] = _bounded_real_source(
            root,
            runtime,
            sample_id,
            max_long_edge=int(tile["real_max_long_edge"]),
        )
    tile_rows: list[dict[str, Any]] = []
    repeat_exact = True
    for sample_id, source in tile_sources.items():
        reference = render_challenger(
            source, runtime, gauge, sampling_dpi=dpi
        )
        sizes = (
            tile["tile_rows"]
            if sample_id == "synthetic"
            else tile["real_tile_rows"]
        )
        for size in sizes:
            for order in tile["orders"]:
                first, seams = render_challenger_row_tiled(
                    source,
                    runtime,
                    gauge,
                    sampling_dpi=dpi,
                    tile_rows=int(size),
                    order=order,
                )
                second, second_seams = render_challenger_row_tiled(
                    source,
                    runtime,
                    gauge,
                    sampling_dpi=dpi,
                    tile_rows=int(size),
                    order=order,
                )
                repeat_exact &= np.array_equal(first, second)
                repeat_exact &= seams == second_seams
                tile_rows.append(
                    {
                        "sample_id": sample_id,
                        "tile_rows": int(size),
                        "order": order,
                        **_tile_metrics(reference, first, seams),
                    }
                )

    resolution = config["resolution_audit"]
    high_shape = tuple(int(value) for value in resolution["high_shape"])
    resolution_sources = {"synthetic": _analytic_chart(high_shape)}
    for sample_id in resolution["real_ids"]:
        real = _bounded_real_source(
            root,
            runtime,
            sample_id,
            max_long_edge=max(high_shape),
        )
        resolution_sources[sample_id] = _resize(real, high_shape)
    resolution_rows: list[dict[str, Any]] = []
    for sample_id, high_source in resolution_sources.items():
        high = render_challenger(
            high_source, runtime, gauge, sampling_dpi=dpi
        )
        for scale in resolution["scale_factors"]:
            factor = float(scale)
            low_shape = (
                int(round(high_shape[0] * factor)),
                int(round(high_shape[1] * factor)),
            )
            low_source = _resize(high_source, low_shape)
            direct = render_challenger(
                low_source,
                runtime,
                gauge,
                sampling_dpi=int(round(dpi * factor)),
            )
            reference = _resize(high, low_shape)
            resolution_rows.append(
                {
                    "sample_id": sample_id,
                    "scale_factor": factor,
                    "sampling_dpi": int(round(dpi * factor)),
                    **_resolution_metrics(direct, reference),
                }
            )

    gates = config["gates"]
    tile_pass = all(
        row["maximum_float_absolute_error"]
        <= float(gates["maximum_tile_float_absolute_error"])
        and row["quantized_mismatch_fraction"]
        <= float(gates["maximum_tile_quantized_mismatch_fraction"])
        and row["maximum_seam_absolute_error"]
        <= float(gates["maximum_tile_seam_absolute_error"])
        for row in tile_rows
    )
    resolution_pass = all(
        row["median_delta_e76"]
        <= float(gates["maximum_resolution_median_delta_e76"])
        and row["p95_delta_e76"]
        <= float(gates["maximum_resolution_p95_delta_e76"])
        and row["maximum_delta_e76"]
        <= float(gates["maximum_resolution_max_delta_e76"])
        and float(gates["minimum_resolution_edge_energy_ratio"])
        <= row["edge_energy_ratio"]
        <= float(gates["maximum_resolution_edge_energy_ratio"])
        for row in resolution_rows
    )
    if not repeat_exact:
        branch = "tile_fail"
    elif not tile_pass:
        branch = "tile_fail"
    elif not resolution_pass:
        branch = "resolution_fail"
    else:
        branch = "pass"
    core = {
        "schema": "neuro_film.u6_p7g_neutral_gauged_invariance_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "repeat_exact": repeat_exact,
        "tile_rows": tile_rows,
        "resolution_rows": resolution_rows,
        "tile_pass": tile_pass,
        "resolution_pass": resolution_pass,
        "decision": branch,
        "branch": config["branch_rule"][branch],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_invariance",
    "render_challenger",
    "render_challenger_row_tiled",
    "validate_contract",
    "write_report",
]
