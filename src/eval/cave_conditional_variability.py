"""U5.R2H0C1 measured-reflectance conditional-variability evaluator.

This module is research-only and deliberately isolated from production
rendering. It verifies transferred CAVE bytes against the Columbia official
ZIP central directory, samples scene-grouped spectra, and measures ambiguity
under the frozen H0A datasheet witness without fitting a model.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from src.eval.velvia_datasheet_witness import (
    CurveBank,
    SpectralContext,
    VelviaWitnessError,
    build_curve_bank,
    build_spectral_context,
    delta_e76,
    load_json,
    render_witness,
    sha256_file,
    validate_curve_evidence,
    validate_sources,
    xyz_to_lab,
)


class CaveVariabilityError(ValueError):
    """Raised when frozen CAVE evidence or evaluator state is invalid."""


@dataclass(frozen=True)
class OfficialMember:
    path: str
    crc32: int
    compressed_size: int
    uncompressed_size: int


@dataclass(frozen=True)
class RepresentativePopulation:
    spectra: np.ndarray
    xyz: np.ndarray
    lab: np.ndarray
    scene: np.ndarray
    cell_row: np.ndarray
    cell_column: np.ndarray


@dataclass(frozen=True)
class PairPopulation:
    first: np.ndarray
    second: np.ndarray
    input_delta_e76: np.ndarray
    spectral_rms: np.ndarray


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _normalized_official_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) >= 3 and parts[0] == parts[1]:
        normalized = "/".join((parts[0], *parts[2:]))
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise CaveVariabilityError(f"unsafe archive path: {path}")
    return normalized


def parse_official_central_directory(
    tail_path: Path, config: Mapping[str, Any]
) -> dict[str, OfficialMember]:
    source = config["source"]
    tail = tail_path.read_bytes()
    expected_hash = str(source["central_directory_sha256"])
    archive_bytes = int(source["expected_bytes"])
    offset = int(source["central_directory_offset"])
    size = int(source["central_directory_bytes"])
    tail_base = archive_bytes - len(tail)
    start = offset - tail_base
    if start < 0 or start + size > len(tail):
        raise CaveVariabilityError("official central directory is outside tail evidence")
    central = tail[start : start + size]
    if hashlib.sha256(central).hexdigest() != expected_hash:
        raise CaveVariabilityError("official central-directory hash mismatch")
    eocd_index = tail.rfind(b"PK\x05\x06")
    if eocd_index < 0:
        raise CaveVariabilityError("official ZIP EOCD is absent")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_index)
    if int(eocd[4]) != int(source["official_entry_count"]):
        raise CaveVariabilityError("official ZIP entry count mismatch")
    if int(eocd[5]) != size or int(eocd[6]) != offset:
        raise CaveVariabilityError("official ZIP central-directory coordinates mismatch")

    members: dict[str, OfficialMember] = {}
    parsed_entries = 0
    position = 0
    while position < len(central):
        if central[position : position + 4] != b"PK\x01\x02":
            raise CaveVariabilityError("invalid central-directory member signature")
        fields = struct.unpack_from("<4s6H3L5H2L", central, position)
        filename_bytes, extra_bytes, comment_bytes = fields[10:13]
        raw_name = central[
            position + 46 : position + 46 + int(filename_bytes)
        ].decode("cp437")
        name = _normalized_official_path(raw_name)
        member = OfficialMember(
            path=name,
            crc32=int(fields[7]),
            compressed_size=int(fields[8]),
            uncompressed_size=int(fields[9]),
        )
        if name in members:
            prior = members[name]
            if not (
                name.endswith("/")
                and prior.crc32 == member.crc32 == 0
                and prior.compressed_size == member.compressed_size == 0
                and prior.uncompressed_size == member.uncompressed_size == 0
            ):
                raise CaveVariabilityError(f"duplicate normalized archive path: {name}")
        else:
            members[name] = member
        parsed_entries += 1
        position += 46 + int(filename_bytes) + int(extra_bytes) + int(comment_bytes)
    if position != len(central) or parsed_entries != int(source["official_entry_count"]):
        raise CaveVariabilityError("official central-directory parse did not close exactly")
    return members


def _is_spectral_png(path: str) -> bool:
    name = Path(path).name
    return path.lower().endswith(".png") and "_ms_" in name


def audit_cave_snapshot(
    snapshot_root: Path,
    tail_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    members = parse_official_central_directory(tail_path, config)
    official = {name: row for name, row in members.items() if _is_spectral_png(name)}
    local = {
        path.relative_to(snapshot_root).as_posix(): path
        for path in snapshot_root.rglob("*_ms_*.png")
    }
    if set(local) != set(official):
        missing = sorted(set(official) - set(local))
        extra = sorted(set(local) - set(official))
        raise CaveVariabilityError(
            f"snapshot member mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    by_scene: dict[str, list[Path]] = {}
    mismatches: list[str] = []
    for name, member in sorted(official.items()):
        path = local[name]
        payload = path.read_bytes()
        if len(payload) != member.uncompressed_size:
            mismatches.append(name)
        if (zlib.crc32(payload) & 0xFFFFFFFF) != member.crc32:
            mismatches.append(name)
        with Image.open(path) as image:
            image.load()
            modes[image.mode] += 1
            dimensions[f"{image.width}x{image.height}"] += 1
        scene = name.split("/", 1)[0]
        by_scene.setdefault(scene, []).append(path)
    if mismatches:
        raise CaveVariabilityError(f"official size/CRC mismatch: {mismatches[:3]}")

    population = config["spectral_population"]
    excluded = set(population.get("excluded_scenes", {}))
    expected_bands = int(population["bands"])
    expected_size = (int(population["image_width"]), int(population["image_height"]))
    retained: dict[str, list[Path]] = {}
    for scene, paths in sorted(by_scene.items()):
        if len(paths) != expected_bands:
            raise CaveVariabilityError(f"scene band count mismatch: {scene}")
        ordered = sorted(paths, key=lambda path: path.stem.rsplit("_", 1)[-1])
        if scene in excluded:
            continue
        for path in ordered:
            with Image.open(path) as image:
                if image.mode != "I;16" or image.size != expected_size:
                    raise CaveVariabilityError(f"nonconforming retained band: {path}")
        retained[scene] = ordered

    if excluded != (set(by_scene) - set(retained)):
        raise CaveVariabilityError("configured scene exclusions do not match audit")
    return (
        {
            "official_entry_count": int(config["source"]["official_entry_count"]),
            "official_normalized_member_count": len(members),
            "official_spectral_png_count": len(official),
            "local_spectral_png_count": len(local),
            "official_crc_mismatch_count": 0,
            "image_modes": dict(sorted(modes.items())),
            "image_dimensions": dict(sorted(dimensions.items())),
            "source_scene_count": len(by_scene),
            "retained_scene_count": len(retained),
            "excluded_scenes": sorted(excluded),
            "retained_band_count": sum(len(value) for value in retained.values()),
        },
        retained,
    )


def build_overlap_witness(
    root: Path,
    h0a_config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> tuple[CurveBank, SpectralContext]:
    validate_sources(root, h0a_config)
    validate_curve_evidence(root, h0a_config, curve_data)
    full_curves = build_curve_bank(h0a_config, curve_data)
    full_context = build_spectral_context(root, h0a_config, full_curves)
    population = experiment_config["spectral_population"]
    target = np.arange(
        int(population["wavelength_nm_start"]),
        int(population["wavelength_nm_end"]) + int(population["wavelength_nm_step"]),
        int(population["wavelength_nm_step"]),
        dtype=np.float64,
    )
    indices = np.array(
        [int(np.flatnonzero(np.isclose(full_curves.wavelength_nm, value))[0]) for value in target]
    )
    curves = CurveBank(
        wavelength_nm=target,
        sensitivity={name: value[indices] for name, value in full_curves.sensitivity.items()},
        dye_density={name: value[indices] for name, value in full_curves.dye_density.items()},
        characteristic_x=full_curves.characteristic_x,
        characteristic_y=full_curves.characteristic_y,
    )
    cmf = full_context.cmf[indices]
    d65 = full_context.d65[indices]
    d50 = full_context.d50[indices]
    step = float(population["wavelength_nm_step"])
    normalizer = 1.0 / float(np.sum(d65 * cmf[:, 1]) * step)
    xyz_matrix = normalizer * (cmf * d65[:, None] * step).T
    film = np.vstack(
        [d65 * curves.sensitivity[name] * step for name in ("blue", "green", "red")]
    )
    neutral = float(h0a_config["witness"]["neutral_reflectance"])
    target_h = 10.0 ** float(h0a_config["witness"]["neutral_log_h"])
    layer_gain = target_h / (film @ np.full(target.size, neutral))
    context = SpectralContext(
        wavelength_nm=target,
        cmf=cmf,
        d65=d65,
        d50=d50,
        xyz_from_reflectance_d65=xyz_matrix,
        film_response=film,
        layer_gain=layer_gain,
    )
    return curves, context


def load_representatives(
    scenes: Mapping[str, Sequence[Path]],
    context: SpectralContext,
    config: Mapping[str, Any],
) -> RepresentativePopulation:
    population = config["spectral_population"]
    rows = int(population["grid_rows"])
    columns = int(population["grid_columns"])
    height = int(population["image_height"])
    width = int(population["image_width"])
    cell_height = height // rows
    cell_width = width // columns
    if rows * cell_height != height or columns * cell_width != width:
        raise CaveVariabilityError("grid must exactly partition every scene")
    scale = float(population["uint16_scale"])

    spectra_rows: list[np.ndarray] = []
    scene_rows: list[str] = []
    cell_rows: list[int] = []
    cell_columns: list[int] = []
    for scene, paths in sorted(scenes.items()):
        bands = []
        for path in paths:
            with Image.open(path) as image:
                band = np.asarray(image, dtype=np.uint16)
            bands.append(band)
        cube = np.stack(bands, axis=-1).astype(np.float64) / scale
        tiled = cube.reshape(rows, cell_height, columns, cell_width, cube.shape[-1])
        representatives = np.median(tiled, axis=(1, 3))
        for row in range(rows):
            for column in range(columns):
                spectra_rows.append(representatives[row, column])
                scene_rows.append(scene)
                cell_rows.append(row)
                cell_columns.append(column)

    spectra = np.asarray(spectra_rows, dtype=np.float64)
    xyz = spectra @ context.xyz_from_reflectance_d65.T
    y = xyz[:, 1]
    keep = (
        np.all(np.isfinite(spectra), axis=1)
        & np.all((spectra >= 0.0) & (spectra <= 1.0), axis=1)
        & (y >= float(population["relative_y_min"]))
        & (y <= float(population["relative_y_max"]))
    )
    return RepresentativePopulation(
        spectra=spectra[keep],
        xyz=xyz[keep],
        lab=xyz_to_lab(xyz[keep]),
        scene=np.asarray(scene_rows, dtype="U64")[keep],
        cell_row=np.asarray(cell_rows, dtype=np.int16)[keep],
        cell_column=np.asarray(cell_columns, dtype=np.int16)[keep],
    )


def select_conditional_pairs(
    population: RepresentativePopulation,
    config: Mapping[str, Any],
    radius: float | None = None,
) -> PairPopulation:
    policy = config["pair_policy"]
    maximum_delta = float(policy["input_delta_e76_max"] if radius is None else radius)
    raw_pairs = cKDTree(population.lab).query_pairs(maximum_delta, output_type="ndarray")
    candidates: list[tuple[float, str, int, int, str, int, int, int, int, float]] = []
    spectral_min = float(policy["spectral_rms_min"])
    for first, second in raw_pairs:
        scene_a = str(population.scene[first])
        scene_b = str(population.scene[second])
        if scene_a == scene_b:
            continue
        rms = float(
            np.sqrt(np.mean((population.spectra[first] - population.spectra[second]) ** 2))
        )
        if rms < spectral_min:
            continue
        delta = float(np.linalg.norm(population.lab[first] - population.lab[second]))
        key_a = (scene_a, int(population.cell_row[first]), int(population.cell_column[first]))
        key_b = (scene_b, int(population.cell_row[second]), int(population.cell_column[second]))
        if key_b < key_a:
            first, second = second, first
            scene_a, scene_b = scene_b, scene_a
            key_a, key_b = key_b, key_a
        candidates.append(
            (delta, *key_a, *key_b, int(first), int(second), rms)
        )
    candidates.sort()

    used: set[int] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    scene_counts: Counter[str] = Counter()
    selected: list[tuple[int, int, float, float]] = []
    pair_cap = int(policy["pairs_per_scene_pair_max"])
    scene_cap = int(policy["pair_incidences_per_scene_max"])
    for candidate in candidates:
        delta, scene_a, _, _, scene_b, _, _, first, second, rms = candidate
        if first in used or second in used:
            continue
        scene_pair = tuple(sorted((str(scene_a), str(scene_b))))
        if pair_counts[scene_pair] >= pair_cap:
            continue
        if scene_counts[str(scene_a)] >= scene_cap or scene_counts[str(scene_b)] >= scene_cap:
            continue
        used.update((int(first), int(second)))
        pair_counts[scene_pair] += 1
        scene_counts[str(scene_a)] += 1
        scene_counts[str(scene_b)] += 1
        selected.append((int(first), int(second), float(delta), float(rms)))
    if not selected:
        return PairPopulation(
            first=np.empty(0, dtype=np.int64),
            second=np.empty(0, dtype=np.int64),
            input_delta_e76=np.empty(0, dtype=np.float64),
            spectral_rms=np.empty(0, dtype=np.float64),
        )
    values = np.asarray(selected, dtype=np.float64)
    return PairPopulation(
        first=values[:, 0].astype(np.int64),
        second=values[:, 1].astype(np.int64),
        input_delta_e76=values[:, 2],
        spectral_rms=values[:, 3],
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    mass = weights[order]
    cumulative = np.cumsum(mass)
    target = quantile * float(cumulative[-1])
    return float(ordered[min(int(np.searchsorted(cumulative, target, side="left")), len(ordered) - 1)])


def scene_bootstrap(
    population: RepresentativePopulation,
    pairs: PairPopulation,
    output_delta: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    scenes = np.unique(population.scene[np.concatenate((pairs.first, pairs.second))])
    pair_a = population.scene[pairs.first]
    pair_b = population.scene[pairs.second]
    rng = np.random.default_rng(seed)
    medians = []
    p95s = []
    attempts = 0
    while len(medians) < repeats and attempts < repeats * 20:
        attempts += 1
        draw = rng.choice(scenes, size=len(scenes), replace=True)
        counts = Counter(str(value) for value in draw)
        weights = np.asarray(
            [counts[str(first)] * counts[str(second)] for first, second in zip(pair_a, pair_b)],
            dtype=np.float64,
        )
        positive = weights > 0
        if not np.any(positive):
            continue
        medians.append(_weighted_quantile(output_delta[positive], weights[positive], 0.5))
        p95s.append(_weighted_quantile(output_delta[positive], weights[positive], 0.95))
    if len(medians) != repeats:
        raise CaveVariabilityError("scene bootstrap produced an empty induced sample")
    return {
        "median": {
            "lower": float(np.percentile(medians, 2.5)),
            "upper": float(np.percentile(medians, 97.5)),
        },
        "p95": {
            "lower": float(np.percentile(p95s, 2.5)),
            "upper": float(np.percentile(p95s, 97.5)),
        },
    }


def _support_report(
    population: RepresentativePopulation, pairs: PairPopulation, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    scene_a = population.scene[pairs.first]
    scene_b = population.scene[pairs.second]
    scene_incidence = Counter(str(value) for value in np.concatenate((scene_a, scene_b)))
    scene_pairs = Counter(tuple(sorted((str(a), str(b)))) for a, b in zip(scene_a, scene_b))
    pair_count = int(len(pairs.first))
    report = {
        "pairs": pair_count,
        "scenes": len(scene_incidence),
        "scene_pairs": len(scene_pairs),
        "max_scene_incidence_share": (
            float(max(scene_incidence.values()) / (2 * pair_count)) if pair_count else 1.0
        ),
        "max_scene_pair_share": (
            float(max(scene_pairs.values()) / pair_count) if pair_count else 1.0
        ),
        "scene_incidence": dict(sorted(scene_incidence.items())),
        "scene_pair_counts": {"|".join(key): value for key, value in sorted(scene_pairs.items())},
    }
    gates = config["support_gates"]
    checks = {
        "pairs": pair_count >= int(gates["pairs_min"]),
        "scenes": len(scene_incidence) >= int(gates["scenes_min"]),
        "scene_pairs": len(scene_pairs) >= int(gates["scene_pairs_min"]),
        "scene_share": report["max_scene_incidence_share"]
        <= float(gates["max_scene_incidence_share"]),
        "scene_pair_share": report["max_scene_pair_share"]
        <= float(gates["max_scene_pair_share"]),
    }
    return report, checks


def _unconditioned_pairs(
    population: RepresentativePopulation, count: int, seed: int
) -> PairPopulation:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(population.scene))
    used: set[int] = set()
    selected = []
    for first in order:
        if int(first) in used:
            continue
        candidates = order[
            (population.scene[order] != population.scene[first])
            & np.asarray([int(value) not in used for value in order])
        ]
        if not len(candidates):
            continue
        second = int(candidates[0])
        used.update((int(first), second))
        delta = float(np.linalg.norm(population.lab[first] - population.lab[second]))
        rms = float(np.sqrt(np.mean((population.spectra[first] - population.spectra[second]) ** 2)))
        selected.append((int(first), second, delta, rms))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise CaveVariabilityError("could not form unconditioned control pairs")
    values = np.asarray(selected, dtype=np.float64)
    return PairPopulation(
        first=values[:, 0].astype(np.int64),
        second=values[:, 1].astype(np.int64),
        input_delta_e76=values[:, 2],
        spectral_rms=values[:, 3],
    )


def evaluate_cave_variability(
    root: Path,
    config: Mapping[str, Any],
    h0a_config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
    snapshot_root: Path,
    tail_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    audit, scenes = audit_cave_snapshot(snapshot_root, tail_path, config)
    curves, context = build_overlap_witness(root, h0a_config, curve_data, config)
    population = load_representatives(scenes, context, config)
    pairs = select_conditional_pairs(population, config)
    support, support_checks = _support_report(population, pairs, config)
    diagnostics = {}
    for radius in config["pair_policy"]["diagnostic_radii_delta_e76"]:
        diagnostic = select_conditional_pairs(population, config, float(radius))
        diagnostic_support, _ = _support_report(population, diagnostic, config)
        diagnostics[str(radius)] = diagnostic_support

    arrays = {
        "representative_spectra": population.spectra,
        "representative_xyz": population.xyz,
        "representative_lab": population.lab,
        "representative_scene": population.scene,
        "representative_cell_row": population.cell_row,
        "representative_cell_column": population.cell_column,
        "pair_first": pairs.first,
        "pair_second": pairs.second,
        "pair_input_delta_e76": pairs.input_delta_e76,
        "pair_spectral_rms": pairs.spectral_rms,
    }
    if not all(support_checks.values()):
        report = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "decision": "insufficient_conditional_support",
            "source_audit": audit,
            "population": {
                "representatives": int(len(population.spectra)),
                "scenes": int(len(np.unique(population.scene))),
                "wavelengths": int(population.spectra.shape[1]),
            },
            "support": support,
            "support_checks": support_checks,
            "diagnostic_support": diagnostics,
            "claim_ceiling": config["claim_ceiling"],
        }
        return report, arrays

    rendered = render_witness(population.spectra, context, curves, "D65")
    output_delta = delta_e76(rendered["xyz"][pairs.first], rendered["xyz"][pairs.second])
    arrays["representative_output_xyz"] = rendered["xyz"]
    arrays["pair_output_delta_e76"] = output_delta
    summary = _summary(output_delta)
    bootstrap = scene_bootstrap(
        population,
        pairs,
        output_delta,
        int(config["gates"]["scene_bootstrap_repeats"]),
        int(config["seed"]),
    )
    control_pairs = _unconditioned_pairs(population, len(pairs.first), int(config["seed"]))
    control_delta = delta_e76(
        rendered["xyz"][control_pairs.first], rendered["xyz"][control_pairs.second]
    )
    arrays["control_pair_first"] = control_pairs.first
    arrays["control_pair_second"] = control_pairs.second
    arrays["control_output_delta_e76"] = control_delta
    comparators = config["comparators"]
    ratios = {
        "median": summary["median"]
        / float(comparators["h0a_adversarial_output_delta_e76_median"]),
        "p95": summary["p95"]
        / float(comparators["h0a_adversarial_output_delta_e76_p95"]),
    }
    gates = config["gates"]
    materially_narrower = (
        ratios["median"] <= float(gates["materially_narrower_ratio_max"])
        and ratios["p95"] <= float(gates["materially_narrower_ratio_max"])
    )
    bounded = (
        materially_narrower
        and summary["median"] <= float(gates["bounded_output_delta_e76_median_max"])
        and summary["p95"] <= float(gates["bounded_output_delta_e76_p95_max"])
        and bootstrap["median"]["upper"]
        <= float(gates["bounded_median_bootstrap_95_ucb_max"])
        and bootstrap["p95"]["upper"]
        <= float(gates["bounded_p95_bootstrap_95_ucb_max"])
    )
    if not materially_narrower:
        decision = "empirical_ambiguity_broad"
    elif not bounded:
        decision = "materially_narrower_but_not_bounded"
    else:
        decision = "bounded_empirical_prior_candidate"
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "source_audit": audit,
        "population": {
            "representatives": int(len(population.spectra)),
            "scenes": int(len(np.unique(population.scene))),
            "wavelengths": int(population.spectra.shape[1]),
        },
        "support": support,
        "support_checks": support_checks,
        "diagnostic_support": diagnostics,
        "metrics": {
            "input_delta_e76": _summary(pairs.input_delta_e76),
            "spectral_rms": _summary(pairs.spectral_rms),
            "output_delta_e76": summary,
            "h0a_adversarial_spread_ratio": ratios,
            "scene_bootstrap_95_interval": bootstrap,
            "unconditioned_input_delta_e76": _summary(control_pairs.input_delta_e76),
            "unconditioned_output_delta_e76": _summary(control_delta),
            "same_spectrum_replay_delta_e76_max": float(
                np.max(delta_e76(rendered["xyz"], rendered["xyz"]))
            ),
            "raw_linear_srgb_min": float(np.min(rendered["linear_srgb"])),
            "raw_linear_srgb_max": float(np.max(rendered["linear_srgb"])),
            "raw_linear_srgb_out_of_gamut_fraction": float(
                np.mean((rendered["linear_srgb"] < 0.0) | (rendered["linear_srgb"] > 1.0))
            ),
        },
        "checks": {
            "materially_narrower": materially_narrower,
            "bounded_empirical_prior": bounded,
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "report": canonical_sha256(report),
        "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())},
    }
