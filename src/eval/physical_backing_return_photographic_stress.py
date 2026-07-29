"""Held-out photographic severe-artifact stress for U6.P3E backing return."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import uniform_filter

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_compiled_backing_return,
    apply_compiled_backing_return_row_tiled,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
)


SCHEMA = "neuro_film.u6_p3f_backing_return_photographic_stress_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3F contract")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(row: dict[str, Any], root: Path) -> tuple[np.ndarray, str]:
    path = root / row["decoded_path"]
    digest = _sha256(path)
    if digest != row["decoded_sha256"]:
        raise ValueError(f"decoded input hash drift for {row['id']}")
    with Image.open(path) as image:
        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    linear = encoded_srgb_to_linear(encoded).astype(np.float32)
    return linear, digest


def _array(values: np.ndarray, pitch: float) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(pitch),
    )


def _isolated_excursions(
    difference: np.ndarray,
    *,
    threshold: float,
    radius: int,
    minimum_support: int,
) -> int:
    excursion = np.max(difference, axis=-1) > threshold
    width = 2 * radius + 1
    support = uniform_filter(
        excursion.astype(np.float32),
        size=width,
        mode="constant",
        cval=0.0,
    ) * float(width * width)
    return int(np.count_nonzero(excursion & (support < minimum_support)))


def _diagnostic_map(values: np.ndarray) -> np.ndarray:
    bounded = values / (1.0 + values)
    return linear_srgb_to_encoded(np.asarray(bounded, dtype=np.float64))


def _preview(encoded: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(
        np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _save_contact_sheet(
    visual_rows: list[dict[str, Any]],
    fixed_ids: list[str],
    gain: float,
    path: Path,
) -> str:
    by_id = {row["id"]: row for row in visual_rows}
    tile_width, tile_height, header = 300, 200, 24
    canvas = Image.new(
        "RGB",
        (3 * tile_width, len(fixed_ids) * (tile_height + header)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, sample_id in enumerate(fixed_ids):
        row = by_id[sample_id]
        y = row_index * (tile_height + header)
        draw.text((4, y + 4), sample_id, fill=(235, 235, 235))
        source_mapped = _diagnostic_map(row["source"])
        candidate_mapped = _diagnostic_map(row["candidate"])
        difference = np.clip(
            0.5 + gain * (candidate_mapped - source_mapped), 0.0, 1.0
        )
        for column, values in enumerate(
            (source_mapped, candidate_mapped, difference)
        ):
            image = _preview(values, (tile_width, tile_height))
            x = column * tile_width + (tile_width - image.width) // 2
            canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False, compress_level=9)
    return _sha256(path)


def evaluate_photographic_stress(
    contract: dict[str, Any],
    manifest: list[dict[str, Any]],
    parent_contract: dict[str, Any],
    *,
    root: Path,
    contact_sheet_path: Path,
) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("invalid U6.P3F contract")
    expected = contract["input"]
    manifest_path = root / expected["manifest"]
    preflight_path = root / expected["preflight_report"]
    if _sha256(manifest_path) != expected["manifest_sha256"]:
        raise ValueError("photographic manifest hash drift")
    if _sha256(preflight_path) != expected["preflight_report_sha256"]:
        raise ValueError("photographic preflight hash drift")
    if len(manifest) != int(expected["expected_rows"]):
        raise ValueError("photographic manifest row count drift")
    if len({row["make"] for row in manifest}) != int(
        expected["expected_camera_makes"]
    ):
        raise ValueError("photographic camera-make count drift")
    for row in manifest:
        if row["allowed_use"] != expected["required_allowed_use"]:
            raise ValueError("photographic allowed-use drift")
        if row["rights_scope"] != expected["required_rights_scope"]:
            raise ValueError("photographic rights-scope drift")
        if row["decoded_color_state"] != expected["decoded_color_state"]:
            raise ValueError("photographic color-state drift")

    reference = backing_return_profile_from_contract(parent_contract)
    compiled = compile_backing_return_profile(reference)
    partitions = [int(value) for value in contract["pipeline"]["row_partitions"]]
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    for source_row in manifest:
        source, input_sha = _load_source(source_row, root)
        source_array = _array(source, compiled.pixel_pitch_um)
        candidate = apply_compiled_backing_return(source_array, compiled).values
        difference = candidate - source
        partition_exact: dict[str, bool] = {}
        for tile_rows in partitions:
            tiled = apply_compiled_backing_return_row_tiled(
                source_array, compiled, tile_rows=tile_rows
            ).values
            partition_exact[str(tile_rows)] = np.array_equal(candidate, tiled)
        row = {
            "id": source_row["id"],
            "make": source_row["make"],
            "input_sha256": input_sha,
            "shape": list(source.shape),
            "finite_nonnegative": bool(
                np.all(np.isfinite(candidate)) and np.all(candidate >= 0.0)
            ),
            "minimum_direct_increment": float(np.min(difference)),
            "maximum_increment_rgb": np.max(
                difference, axis=(0, 1)
            ).astype(np.float64).tolist(),
            "p95_increment_rgb": np.quantile(
                difference, 0.95, axis=(0, 1)
            ).astype(np.float64).tolist(),
            "partition_exact": partition_exact,
            "isolated_excursion_count": _isolated_excursions(
                difference,
                threshold=float(gates["isolated_excursion_threshold"]),
                radius=int(gates["isolated_support_radius_pixels"]),
                minimum_support=int(gates["minimum_isolated_support_count"]),
            ),
        }
        rows.append(row)
        if source_row["id"] in contract["visual_protocol"]["fixed_ids"]:
            visual_rows.append(
                {"id": source_row["id"], "source": source, "candidate": candidate}
            )
    fixed_ids = contract["visual_protocol"]["fixed_ids"]
    if {row["id"] for row in visual_rows} != set(fixed_ids):
        raise ValueError("fixed visual IDs are incomplete")
    contact_sha = _save_contact_sheet(
        visual_rows,
        fixed_ids,
        float(contract["pipeline"]["difference_display_gain"]),
        contact_sheet_path,
    )
    increment_limit = np.asarray(gates["maximum_increment_rgb"], dtype=np.float64)
    decisions = {
        "input_hashes": all(
            row["input_sha256"] == source["decoded_sha256"]
            for row, source in zip(rows, manifest, strict=True)
        ),
        "finite_nonnegative": all(row["finite_nonnegative"] for row in rows),
        "direct_retention": min(
            row["minimum_direct_increment"] for row in rows
        )
        >= float(gates["minimum_direct_increment"]),
        "increment_bound": all(
            np.all(np.asarray(row["maximum_increment_rgb"]) <= increment_limit)
            for row in rows
        ),
        "partition_exact": all(
            all(row["partition_exact"].values()) for row in rows
        ),
        "isolated_excursions": sum(
            row["isolated_excursion_count"] for row in rows
        )
        <= int(gates["maximum_isolated_excursion_count"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p3f_backing_return_photographic_stress_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "rows": rows,
        "population_maximum_increment_rgb": np.max(
            np.asarray([row["maximum_increment_rgb"] for row in rows]), axis=0
        ).tolist(),
        "total_isolated_excursion_count": sum(
            row["isolated_excursion_count"] for row in rows
        ),
        "contact_sheet_sha256": contact_sha,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
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
