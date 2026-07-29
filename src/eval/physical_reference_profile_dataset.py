"""U6.P4M deterministic synthetic reference-profile dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.eval.physical_exact_area_streaming import _profiles
from src.film_physics.density_conditioned_structure import (
    iter_density_conditioned_structure_area_lod_rows,
    render_density_conditioned_structure_area_lod,
)


SCHEMA = "neuro_film.u6_p4m_reference_profile_dataset_contract.v1"
MANIFEST_SCHEMA = (
    "neuro_film.u6_p4m_reference_profile_dataset_manifest.v1"
)
REPORT_SCHEMA = "neuro_film.u6_p4m_reference_profile_dataset_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    absolute = root / path
    if _sha256(absolute) != expected:
        raise ValueError(f"U6.P4M parent drift: {path}")
    return json.loads(absolute.read_text(encoding="utf-8"))


def load_contract(
    root: Path, path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4M contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or contract["dataset"]["photograph_or_scan_inputs_allowed"]
        or contract["dataset"]["measured_film_statistics_allowed"]
        or contract["execution"]["network_allowed"]
        or contract["execution"]["gpu_allowed"]
        or contract["execution"]["per_image_fit_or_normalization_allowed"]
        or contract["execution"]["output_clipping_allowed"]
    ):
        raise ValueError("unsupported U6.P4M contract")
    parents = contract["parents"]
    parent = _load_exact(
        root,
        parents["p4d_contract"],
        parents["p4d_contract_sha256"],
    )
    decision = _load_exact(
        root,
        parents["p4l_decision"],
        parents["p4l_decision_sha256"],
    )
    if (
        decision["decision"]
        != "retain_local_exact_area_reference_performance_open_profile_dataset"
        or not decision["next_leaf"].startswith("U6.P4M")
    ):
        raise ValueError("U6.P4L did not open profile dataset generation")
    return contract, parent


def _field(
    family: str,
    *,
    seed: int,
    shape: tuple[int, int],
    scales: np.ndarray,
    density_domain: tuple[float, float],
) -> np.ndarray:
    height, width = shape
    ys, xs = np.indices(shape, dtype=np.float64)
    x = (xs + 0.5) / float(width)
    y = (ys + 0.5) / float(height)
    phase = (seed % 97) / 97.0
    low = 0.10 + 0.05 * (seed % 5)
    high = 1.45 + 0.08 * (seed % 6)
    if family == "flat":
        base = np.full(shape, 0.20 + 1.30 * phase)
    elif family == "gradient":
        base = low + (high - low) * np.clip(
            0.65 * x + 0.35 * y, 0.0, 1.0
        )
    elif family == "step":
        boundary = 0.30 + 0.05 * (seed % 7)
        base = np.where(x < boundary, low, high)
    elif family == "checker":
        cell = 3 + seed % 9
        base = np.where(
            ((ys.astype(np.int64) // cell) + (xs.astype(np.int64) // cell))
            % 2
            == 0,
            low,
            high,
        )
    elif family == "islands":
        base = np.full(shape, 0.55 + 0.04 * (seed % 5))
        radius = 0.07 + 0.01 * (seed % 4)
        first = (x - (0.25 + 0.12 * phase)) ** 2 + (
            y - (0.38 + 0.08 * phase)
        ) ** 2
        second = (x - (0.72 - 0.09 * phase)) ** 2 + (
            y - (0.65 - 0.11 * phase)
        ) ** 2
        base[first <= radius * radius] = high
        base[second <= (0.8 * radius) ** 2] = low
    elif family == "harmonic":
        wave = (
            0.52
            + 0.24 * np.sin(2.0 * np.pi * ((2 + seed % 4) * x + phase))
            + 0.18
            * np.cos(2.0 * np.pi * ((3 + seed % 5) * y - phase))
            + 0.06
            * np.sin(2.0 * np.pi * (2.0 * x + 3.0 * y + phase))
        )
        base = low + (high - low) * np.clip(wave, 0.0, 1.0)
    else:
        raise ValueError(f"unsupported U6.P4M field family: {family}")
    result = np.clip(
        base[..., None] * scales[None, None, :],
        density_domain[0],
        density_domain[1],
    )
    return np.asarray(result, dtype="<f4")


def _write_npy(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype="<f4"), allow_pickle=False)
        handle.flush()
    temporary.replace(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _render_streamed(
    target: np.ndarray,
    profiles: tuple,
    *,
    factor: int,
    row_tile_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    density = np.empty_like(target, dtype=np.float32)
    transmittance = np.empty_like(target, dtype=np.float32)
    covered = 0
    for y0, result in iter_density_conditioned_structure_area_lod_rows(
        target.astype(np.float64),
        profiles,
        pixel_size_factor=factor,
        row_tile_height=row_tile_height,
    ):
        y1 = y0 + result.density.shape[0]
        if y0 != covered:
            raise RuntimeError("U6.P4M row stream is not contiguous")
        density[y0:y1] = result.density
        transmittance[y0:y1] = result.transmittance
        covered = y1
    if covered != target.shape[0]:
        raise RuntimeError("U6.P4M row stream did not cover the target")
    return density, transmittance


def generate_reference_dataset(
    contract: dict[str, Any],
    parent: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"U6.P4M output already exists: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    dataset = contract["dataset"]
    shape = tuple(int(value) for value in dataset["coarse_shape"])
    scales = np.asarray(
        dataset["channel_density_scales"], dtype=np.float64
    )
    density_domain = tuple(
        float(value) for value in dataset["density_domain"]
    )
    factor = int(dataset["pixel_size_factor"])
    tile = int(contract["execution"]["row_tile_height"])
    records: list[dict[str, Any]] = []
    parity_exact = True
    for split, seeds in dataset["splits"].items():
        for seed in (int(value) for value in seeds):
            profiles = _profiles(parent, seed_offset=seed)
            for family in dataset["field_families"]:
                target = _field(
                    family,
                    seed=seed,
                    shape=shape,
                    scales=scales,
                    density_domain=density_domain,
                )
                density, transmittance = _render_streamed(
                    target,
                    profiles,
                    factor=factor,
                    row_tile_height=tile,
                )
                full = render_density_conditioned_structure_area_lod(
                    target.astype(np.float64),
                    profiles,
                    pixel_size_factor=factor,
                )
                parity_exact = bool(
                    parity_exact
                    and np.array_equal(density, full.density)
                    and np.array_equal(transmittance, full.transmittance)
                )
                prefix = Path(split) / f"group_{seed}" / family
                files = {
                    "input_density": _write_npy(
                        output_root / f"{prefix}_input_density.npy",
                        target,
                    ),
                    "target_density": _write_npy(
                        output_root / f"{prefix}_target_density.npy",
                        density,
                    ),
                    "target_transmittance": _write_npy(
                        output_root / f"{prefix}_target_transmittance.npy",
                        transmittance,
                    ),
                }
                for value in files.values():
                    value["path"] = str(
                        Path(value["path"]).relative_to(output_root)
                    ).replace("\\", "/")
                records.append(
                    {
                        "split": split,
                        "group_id": f"group_{seed}",
                        "seed_offset": seed,
                        "field_family": family,
                        "input_array_sha256": hashlib.sha256(
                            target.tobytes(order="C")
                        ).hexdigest(),
                        "target_density_array_sha256": hashlib.sha256(
                            density.tobytes(order="C")
                        ).hexdigest(),
                        "target_transmittance_array_sha256": hashlib.sha256(
                            transmittance.tobytes(order="C")
                        ).hexdigest(),
                        "minimum_density": float(np.min(density)),
                        "maximum_transmittance": float(
                            np.max(transmittance)
                        ),
                        "files": files,
                    }
                )
    records.sort(
        key=lambda row: (
            row["split"],
            row["group_id"],
            row["field_family"],
        )
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "node": contract["node"],
        "shape": list(shape),
        "pixel_size_factor": factor,
        "records": records,
        "row_stream_full_region_byte_exact": parity_exact,
    }
    manifest_path = output_root / "manifest.json"
    encoded = (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(encoded)
    inventory = {
        str(path.relative_to(output_root)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    return {
        "manifest": manifest,
        "inventory": inventory,
        "dataset_bytes": sum(
            row["bytes"] for row in inventory.values()
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_reference_datasets(
    contract: dict[str, Any],
    generations: list[dict[str, Any]],
    *,
    owned_temp_residue_count: int,
) -> dict[str, Any]:
    expected = int(contract["execution"]["independent_generations"])
    if len(generations) != expected:
        raise ValueError("U6.P4M generation count does not match contract")
    first_manifest = generations[0]["manifest"]
    records = first_manifest["records"]
    splits = contract["dataset"]["splits"]
    split_groups = {
        split: {f"group_{int(value)}" for value in seeds}
        for split, seeds in splits.items()
    }
    group_overlap = 0
    names = list(split_groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            group_overlap += len(split_groups[left] & split_groups[right])

    def hash_overlap(key: str) -> int:
        by_split = {
            split: {
                row[key] for row in records if row["split"] == split
            }
            for split in splits
        }
        overlap = 0
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap += len(by_split[left] & by_split[right])
        return overlap

    counts = {
        split: sum(row["split"] == split for row in records)
        for split in splits
    }
    gates = contract["automatic_gates"]
    checks = {
        "record_count": len(records) == int(gates["expected_record_count"]),
        "split_counts": (
            counts["development"]
            == int(gates["expected_development_records"])
            and counts["confirmation"]
            == int(gates["expected_confirmation_records"])
            and counts["stress"] == int(gates["expected_stress_records"])
        ),
        "group_split_disjoint": group_overlap
        == int(gates["group_split_overlap_count"]),
        "input_hash_split_disjoint": hash_overlap("input_array_sha256")
        == int(gates["cross_split_input_hash_overlap_count"]),
        "target_hash_split_disjoint": hash_overlap(
            "target_density_array_sha256"
        )
        == int(gates["cross_split_target_hash_overlap_count"]),
        "repeat_inventory_byte_exact": all(
            row["inventory"] == generations[0]["inventory"]
            for row in generations[1:]
        )
        == bool(gates["repeat_inventory_byte_exact"]),
        "row_stream_full_region_byte_exact": all(
            row["manifest"]["row_stream_full_region_byte_exact"]
            for row in generations
        )
        == bool(gates["row_stream_full_region_byte_exact"]),
        "physical_domain": (
            min(row["minimum_density"] for row in records) >= 0.0
            and max(row["maximum_transmittance"] for row in records) <= 1.0
        )
        == bool(gates["density_and_transmittance_domain_valid"]),
        "dataset_size": max(
            int(row["dataset_bytes"]) for row in generations
        )
        <= int(gates["maximum_dataset_bytes"]),
        "generation_wall": max(
            float(row["wall_seconds"]) for row in generations
        )
        <= float(gates["maximum_generation_wall_seconds"]),
        "owned_temp_cleanup": owned_temp_residue_count
        == int(gates["owned_temp_residue_count"]),
    }
    passed = all(checks.values())
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "record_counts": counts,
        "group_split_overlap_count": group_overlap,
        "cross_split_input_hash_overlap_count": hash_overlap(
            "input_array_sha256"
        ),
        "cross_split_target_hash_overlap_count": hash_overlap(
            "target_density_array_sha256"
        ),
        "dataset_bytes": [
            row["dataset_bytes"] for row in generations
        ],
        "generation_wall_seconds": [
            row["wall_seconds"] for row in generations
        ],
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                generations[0]["inventory"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "checks": checks,
        "automatic_pass": passed,
        "decision": (
            contract["branch_rule"]["pass"]
            if passed
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "MANIFEST_SCHEMA",
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_reference_datasets",
    "generate_reference_dataset",
    "load_contract",
]
