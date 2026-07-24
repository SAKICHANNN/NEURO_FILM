"""U5.R2H0C3 cross-source measured-spectrum replication evaluator."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from src.eval.cave_conditional_variability import (
    array_sha256,
    audit_cave_snapshot,
    build_overlap_witness,
    load_representatives,
)
from src.eval.hard_spectrum_canonicalizer import canonical_sha256
from src.eval.velvia_datasheet_witness import (
    delta_e76,
    reconstruct_reflectances,
    render_witness,
    sha256_file,
    xyz_to_lab,
)


class ExternalSpectrumError(ValueError):
    """Raised when the frozen H0C3 source or population is invalid."""


@dataclass(frozen=True)
class ExternalSpectrumPopulation:
    spectra: np.ndarray
    xyz: np.ndarray
    lab: np.ndarray
    member: np.ndarray
    header: np.ndarray
    chapter: np.ndarray
    instrument: np.ndarray
    sample_group: np.ndarray


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to match the official record
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_usgs_archive(
    archive_path: Path, source_decision: Mapping[str, Any]
) -> dict[str, Any]:
    source = source_decision["source"]
    if archive_path.stat().st_size != int(source["bytes"]):
        raise ExternalSpectrumError("USGS archive byte count mismatch")
    if _file_md5(archive_path) != str(source["md5"]):
        raise ExternalSpectrumError("USGS archive MD5 mismatch")
    if sha256_file(archive_path) != str(source["sha256"]):
        raise ExternalSpectrumError("USGS archive SHA-256 mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        archive.testzip()
    members = len(infos)
    uncompressed = sum(info.file_size for info in infos)
    if members != int(source["zip_members"]):
        raise ExternalSpectrumError("USGS ZIP member count mismatch")
    if uncompressed != int(source["uncompressed_bytes"]):
        raise ExternalSpectrumError("USGS uncompressed byte count mismatch")
    return {
        "bytes": archive_path.stat().st_size,
        "md5": str(source["md5"]),
        "sha256": str(source["sha256"]),
        "zip_members": members,
        "uncompressed_bytes": uncompressed,
    }


def _numeric_lines(payload: bytes) -> tuple[str, np.ndarray]:
    lines = payload.decode("utf-8", "replace").splitlines()
    if not lines:
        raise ExternalSpectrumError("empty USGS text member")
    values = []
    for line in lines[1:]:
        if line.strip():
            try:
                values.append(float(line))
            except ValueError as error:
                raise ExternalSpectrumError("non-numeric USGS sample row") from error
    return lines[0].strip(), np.asarray(values, dtype=np.float64)


def sample_group_from_member(member: str, pattern: str) -> str:
    stem = PurePosixPath(member).stem
    match = re.match(pattern, stem)
    if match is None or not match.group(1):
        raise ExternalSpectrumError(f"cannot derive sample group: {member}")
    return match.group(1)


def load_usgs_population(
    archive_path: Path,
    context: Any,
    source_decision: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[ExternalSpectrumPopulation, dict[str, int]]:
    eligibility = source_decision["eligibility"]
    rejection: Counter[str] = Counter()
    spectra_rows: list[np.ndarray] = []
    members: list[str] = []
    headers: list[str] = []
    chapters: list[str] = []
    instruments: list[str] = []
    groups: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        wavelengths: dict[int, np.ndarray] = {}
        for member in archive.namelist():
            if "Wavelengths_" not in member or not member.endswith(".txt"):
                continue
            _, values = _numeric_lines(archive.read(member))
            wavelengths[len(values)] = values * 1000.0
        for member in archive.namelist():
            if not member.endswith(".txt") or "Wavelengths_" in member:
                continue
            if bool(eligibility["exclude_errorbars"]) and "/errorbars/" in member:
                rejection["errorbars"] += 1
                continue
            header, values = _numeric_lines(archive.read(member))
            if not header.endswith(" " + str(eligibility["measurement_type"])):
                rejection["not_aref"] += 1
                continue
            wavelength = wavelengths.get(len(values))
            if wavelength is None:
                rejection["unknown_wavelength_record"] += 1
                continue
            valid = np.isfinite(values) & (
                values > float(eligibility["missing_value_below"])
            )
            if np.count_nonzero(valid) < 2:
                rejection["insufficient_valid_samples"] += 1
                continue
            measured_wavelength = wavelength[valid]
            measured_value = values[valid]
            order = np.argsort(measured_wavelength, kind="mergesort")
            measured_wavelength = measured_wavelength[order]
            measured_value = measured_value[order]
            measured_wavelength, unique_index = np.unique(
                measured_wavelength, return_index=True
            )
            measured_value = measured_value[unique_index]
            if (
                measured_wavelength[0] > context.wavelength_nm[0]
                or measured_wavelength[-1] < context.wavelength_nm[-1]
            ):
                rejection["insufficient_visible_overlap"] += 1
                continue
            spectrum = np.interp(
                context.wavelength_nm, measured_wavelength, measured_value
            )
            if not np.all(np.isfinite(spectrum)):
                rejection["nonfinite_interpolation"] += 1
                continue
            if np.any(spectrum < float(eligibility["reflectance_min"])) or np.any(
                spectrum > float(eligibility["reflectance_max"])
            ):
                rejection["reflectance_out_of_bounds"] += 1
                continue
            xyz = spectrum @ context.xyz_from_reflectance_d65.T
            if not (
                float(eligibility["relative_y_min"])
                <= xyz[1]
                <= float(eligibility["relative_y_max"])
            ):
                rejection["relative_y_out_of_bounds"] += 1
                continue
            parts = PurePosixPath(member).parts
            if len(parts) < 3:
                raise ExternalSpectrumError(f"missing USGS chapter: {member}")
            tokens = header.split()
            if len(tokens) < 2:
                raise ExternalSpectrumError(f"missing USGS instrument: {member}")
            spectra_rows.append(spectrum)
            members.append(member)
            headers.append(header)
            chapters.append(parts[1])
            instruments.append(tokens[-2])
            groups.append(sample_group_from_member(member, config["sample_group_regex"]))
    spectra = np.asarray(spectra_rows, dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape[1] != len(context.wavelength_nm):
        raise ExternalSpectrumError("invalid USGS population shape")
    xyz = spectra @ context.xyz_from_reflectance_d65.T
    return (
        ExternalSpectrumPopulation(
            spectra=spectra,
            xyz=xyz,
            lab=xyz_to_lab(xyz),
            member=np.asarray(members, dtype="U256"),
            header=np.asarray(headers, dtype="U256"),
            chapter=np.asarray(chapters, dtype="U64"),
            instrument=np.asarray(instruments, dtype="U32"),
            sample_group=np.asarray(groups, dtype="U192"),
        ),
        dict(sorted(rejection.items())),
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
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
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(ordered) - 1)
    return float(ordered[index])


def policy_group_bootstrap(
    groups: np.ndarray,
    smooth_error: np.ndarray,
    hard_error: np.ndarray,
    tie_tolerance: float,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    win_rates = []
    reductions = []
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        counts = Counter(str(value) for value in draw)
        weights = np.asarray([counts[str(group)] for group in groups], dtype=np.float64)
        selected = weights > 0
        mass = float(np.sum(weights[selected]))
        wins = hard_error[selected] < smooth_error[selected] - tie_tolerance
        win_rates.append(float(np.sum(weights[selected] * wins) / mass))
        smooth_median = _weighted_quantile(smooth_error[selected], weights[selected], 0.5)
        hard_median = _weighted_quantile(hard_error[selected], weights[selected], 0.5)
        reductions.append((smooth_median - hard_median) / max(smooth_median, 1e-12))
    return {
        "win_rate": {
            "lower": float(np.percentile(win_rates, 2.5)),
            "upper": float(np.percentile(win_rates, 97.5)),
        },
        "median_relative_error_reduction": {
            "lower": float(np.percentile(reductions, 2.5)),
            "upper": float(np.percentile(reductions, 97.5)),
        },
    }


def evaluate_external_replication(
    root: Path,
    config: Mapping[str, Any],
    source_decision: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    h0a_config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
    snapshot_root: Path,
    tail_path: Path,
    usgs_archive: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_audit = audit_usgs_archive(usgs_archive, source_decision)
    cave_audit, scenes = audit_cave_snapshot(snapshot_root, tail_path, parent_config)
    curves, context = build_overlap_witness(root, h0a_config, curve_data, parent_config)
    bank = load_representatives(scenes, context, parent_config)
    query, rejection = load_usgs_population(usgs_archive, context, source_decision, config)
    nearest_distance, nearest = cKDTree(bank.lab).query(query.lab, k=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float64)

    target = render_witness(query.spectra, context, curves, "D65")
    smooth_spectra = reconstruct_reflectances(query.xyz, context.xyz_from_reflectance_d65)
    smooth = render_witness(smooth_spectra, context, curves, "D65")
    hard = render_witness(bank.spectra[nearest], context, curves, "D65")
    identity_error = delta_e76(query.xyz, target["xyz"])
    smooth_error = delta_e76(smooth["xyz"], target["xyz"])
    hard_error = delta_e76(hard["xyz"], target["xyz"])
    threshold = float(config["threshold_delta_e76"])
    selected = nearest_distance <= threshold
    policy_error = np.where(selected, hard_error, smooth_error)
    if not np.any(selected):
        raise ExternalSpectrumError("frozen threshold selects no USGS query")

    groups = query.sample_group[selected]
    chapters = query.chapter[selected]
    group_counts = Counter(str(value) for value in groups)
    chapter_counts = Counter(str(value) for value in chapters)
    tolerance = float(config["tie_tolerance_delta_e76"])
    wins = hard_error[selected] < smooth_error[selected] - tolerance
    losses = hard_error[selected] > smooth_error[selected] + tolerance
    smooth_selected = _summary(smooth_error[selected])
    hard_selected = _summary(hard_error[selected])
    median_reduction = (smooth_selected["median"] - hard_selected["median"]) / max(
        smooth_selected["median"], 1e-12
    )
    bootstrap = policy_group_bootstrap(
        groups,
        smooth_error[selected],
        hard_error[selected],
        tolerance,
        int(config["gates"]["sample_group_bootstrap_repeats"]),
        int(config["seed"]),
    )
    chapter_metrics: dict[str, Any] = {}
    evaluable = []
    for chapter in sorted(chapter_counts):
        use = selected & (query.chapter == chapter)
        record = {
            "rows": int(np.sum(use)),
            "smooth_error_delta_e76": _summary(smooth_error[use]),
            "hard_error_delta_e76": _summary(hard_error[use]),
            "win_rate": float(np.mean(hard_error[use] < smooth_error[use] - tolerance)),
        }
        record["hard_median_lower"] = (
            record["hard_error_delta_e76"]["median"]
            < record["smooth_error_delta_e76"]["median"]
        )
        chapter_metrics[chapter] = record
        if record["rows"] >= int(config["gates"]["evaluable_chapter_rows_min"]):
            evaluable.append(record)
    lower_chapter_share = float(
        np.mean([record["hard_median_lower"] for record in evaluable])
    ) if evaluable else 0.0

    gates = config["gates"]
    smooth_all = _summary(smooth_error)
    policy_all = _summary(policy_error)
    checks = {
        "eligible_queries": len(query.spectra) >= int(gates["eligible_queries_min"]),
        "eligible_chapters": len(np.unique(query.chapter))
        >= int(gates["eligible_chapters_min"]),
        "selected_queries": int(np.sum(selected)) >= int(gates["selected_queries_min"]),
        "coverage": float(np.mean(selected)) >= float(gates["coverage_min"]),
        "selected_sample_groups": len(group_counts)
        >= int(gates["selected_sample_groups_min"]),
        "sample_group_share": max(group_counts.values()) / int(np.sum(selected))
        <= float(gates["max_selected_sample_group_share"]),
        "selected_chapters": len(chapter_counts) >= int(gates["selected_chapters_min"]),
        "chapter_share": max(chapter_counts.values()) / int(np.sum(selected))
        <= float(gates["max_selected_chapter_share"]),
        "evaluable_chapters": len(evaluable) >= int(gates["evaluable_chapters_min"]),
        "chapter_direction": lower_chapter_share
        >= float(gates["chapters_with_lower_hard_median_share_min"]),
        "win_rate": float(np.mean(wins)) >= float(gates["selected_win_rate_min"]),
        "win_rate_bootstrap": bootstrap["win_rate"]["lower"]
        >= float(gates["selected_win_rate_bootstrap_95_lcb_min"]),
        "median_reduction": median_reduction
        >= float(gates["selected_median_relative_error_reduction_min"]),
        "median_reduction_bootstrap": bootstrap["median_relative_error_reduction"][
            "lower"
        ]
        >= float(
            gates["selected_median_relative_error_reduction_bootstrap_95_lcb_min"]
        ),
        "hard_median": hard_selected["median"]
        <= float(gates["selected_hard_error_delta_e76_median_max"]),
        "hard_p95": hard_selected["p95"]
        <= float(gates["selected_hard_error_delta_e76_p95_max"]),
        "full_p95_noninferiority": policy_all["p95"]
        <= smooth_all["p95"]
        + float(gates["fallback_policy_p95_vs_smooth_max_increase"]),
    }
    passed = all(checks.values())
    all_rgb = np.concatenate(
        [target["linear_srgb"], smooth["linear_srgb"], hard["linear_srgb"]], axis=0
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": (
            "external_hard_canonicalizer_replication"
            if passed
            else "external_replication_failed"
        ),
        "source_audit": source_audit,
        "cave_source_audit": cave_audit,
        "rejection_counts": rejection,
        "population": {
            "queries": len(query.spectra),
            "chapters": len(np.unique(query.chapter)),
            "sample_groups": len(np.unique(query.sample_group)),
            "bank_rows": len(bank.spectra),
            "bank_scenes": len(np.unique(bank.scene)),
            "wavelengths": query.spectra.shape[1],
        },
        "policy": {
            "threshold_delta_e76": threshold,
            "selected_queries": int(np.sum(selected)),
            "coverage": float(np.mean(selected)),
            "selected_sample_groups": len(group_counts),
            "max_selected_sample_group_share": float(
                max(group_counts.values()) / int(np.sum(selected))
            ),
            "selected_chapters": len(chapter_counts),
            "max_selected_chapter_share": float(
                max(chapter_counts.values()) / int(np.sum(selected))
            ),
            "chapter_counts": dict(sorted(chapter_counts.items())),
            "evaluable_chapters": len(evaluable),
            "chapters_with_lower_hard_median_share": lower_chapter_share,
            "win_rate": float(np.mean(wins)),
            "tie_rate": float(np.mean(~(wins | losses))),
            "loss_rate": float(np.mean(losses)),
            "smooth_selected_error_delta_e76": smooth_selected,
            "hard_selected_error_delta_e76": hard_selected,
            "selected_median_relative_error_reduction": float(median_reduction),
            "sample_group_bootstrap_95_interval": bootstrap,
            "fallback_policy_error_delta_e76": policy_all,
            "fallback_policy_p95_change_vs_smooth": float(
                policy_all["p95"] - smooth_all["p95"]
            ),
            "chapter_metrics": chapter_metrics,
            "checks": checks,
            "eligible": passed,
        },
        "metrics": {
            "nearest_cave_delta_e76": _summary(nearest_distance),
            "known_target_effect_delta_e76": _summary(identity_error),
            "smooth_error_delta_e76": smooth_all,
            "raw_hard_error_delta_e76": _summary(hard_error),
            "raw_linear_srgb_min": float(np.min(all_rgb)),
            "raw_linear_srgb_max": float(np.max(all_rgb)),
            "raw_linear_srgb_out_of_gamut_fraction": float(
                np.mean((all_rgb < 0.0) | (all_rgb > 1.0))
            ),
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "member": query.member,
        "header": query.header,
        "chapter": query.chapter,
        "instrument": query.instrument,
        "sample_group": query.sample_group,
        "nearest_index": nearest,
        "nearest_scene": bank.scene[nearest],
        "nearest_cell_row": bank.cell_row[nearest],
        "nearest_cell_column": bank.cell_column[nearest],
        "nearest_distance_delta_e76": nearest_distance,
        "identity_error_delta_e76": identity_error,
        "smooth_error_delta_e76": smooth_error,
        "hard_error_delta_e76": hard_error,
        "policy_error_delta_e76": policy_error,
        "selected": selected,
        "target_output_xyz": target["xyz"],
        "smooth_output_xyz": smooth["xyz"],
        "hard_output_xyz": hard["xyz"],
    }
    return report, arrays


def result_hashes(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "report": canonical_sha256(report),
        "arrays": {name: array_sha256(value) for name, value in sorted(arrays.items())},
    }

