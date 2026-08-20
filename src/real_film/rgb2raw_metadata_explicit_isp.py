"""Capture-metadata-conditioned explicit RGB2RAW ISP evaluation."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import pickle
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
    fit_operator_rows,
    gradient_p999_ratio,
    matrix_diagnostics,
    mean_oklab_error,
    new_exact_boundary_fraction,
)
from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    canonical_sha256,
    http_range_get,
    parse_central_directory,
    sha256_bytes,
)
from src.real_film.rgb2raw_capture_pair_pixel_preflight import extract_member


class RGB2RAWMetadataISPError(ValueError):
    """Raised when the frozen SF3.A0T contract or source drifts."""


RangeReader = Callable[[str, int, int, int], bytes]


@dataclass(frozen=True)
class PairRow:
    camera_group: str
    canonical_capture_group: str
    raw_capture_group: str
    pair_stem: str
    role: str
    npy: str
    png: str
    metadata: str

    def record(self) -> dict[str, str]:
        return {
            "camera_group": self.camera_group,
            "canonical_capture_group": self.canonical_capture_group,
            "raw_capture_group": self.raw_capture_group,
            "pair_stem": self.pair_stem,
            "role": self.role,
            "npy": self.npy,
            "png": self.png,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DecodedRow:
    pair: PairRow
    base_rgb: np.ndarray
    target_rgb: np.ndarray
    capture_wb: np.ndarray
    daylight_wb: np.ndarray
    member_sha256: Mapping[str, str]
    network_bytes: int


def _hash_rank(preimage: str) -> bytes:
    return hashlib.sha256(preimage.encode("utf-8")).digest()


def build_role_rows(
    members: Sequence[ZipMember], config: Mapping[str, Any]
) -> list[PairRow]:
    """Build exact group-disjoint roles without reading a member body."""
    png: dict[str, set[str]] = defaultdict(set)
    npy: dict[str, set[str]] = defaultdict(set)
    metadata: dict[str, set[str]] = defaultdict(set)
    for member in members:
        path = PurePosixPath(member.name)
        if len(path.parts) != 3 or path.parts[0] != "train":
            continue
        camera = path.parts[1]
        if path.suffix.lower() == ".png":
            png[camera].add(path.stem)
        elif path.suffix.lower() == ".npy":
            npy[camera].add(path.stem)
        elif path.suffix.lower() == ".pkl":
            metadata[camera].add(path.stem)

    role_counts = config["roles"]["per_camera_counts"]
    rows: list[PairRow] = []
    for camera in ("iphone-x", "samsung-s9"):
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for stem in sorted(png[camera] & npy[camera]):
            raw_group = stem.rsplit("_", 1)[0]
            canonical = (
                str(int(raw_group)) if raw_group.isdigit() else raw_group.lower()
            )
            if raw_group not in metadata[camera]:
                raise RGB2RAWMetadataISPError(
                    f"metadata missing for {camera}/{raw_group}"
                )
            groups[canonical].append((raw_group, stem))
        ranked_groups = sorted(
            groups,
            key=lambda group: _hash_rank(f"sf3-a0t-role-v1|{camera}|{group}"),
        )
        required = sum(
            int(role_counts[name])
            for name in ("fit", "calibration", "sealed_confirmation")
        )
        if len(ranked_groups) < required:
            raise RGB2RAWMetadataISPError(f"insufficient canonical groups for {camera}")
        boundaries = {
            "fit": (0, int(role_counts["fit"])),
            "calibration": (
                int(role_counts["fit"]),
                int(role_counts["fit"]) + int(role_counts["calibration"]),
            ),
            "sealed_confirmation": (
                int(role_counts["fit"]) + int(role_counts["calibration"]),
                required,
            ),
        }
        for role, (start, stop) in boundaries.items():
            for canonical in ranked_groups[start:stop]:
                raw_group, stem = min(
                    groups[canonical],
                    key=lambda item: _hash_rank(
                        f"sf3-a0t-pair-v1|{camera}|{canonical}|{item[1]}"
                    ),
                )
                rows.append(
                    PairRow(
                        camera_group=camera,
                        canonical_capture_group=canonical,
                        raw_capture_group=raw_group,
                        pair_stem=stem,
                        role=role,
                        npy=f"train/{camera}/{stem}.npy",
                        png=f"train/{camera}/{stem}.png",
                        metadata=f"train/{camera}/{raw_group}.pkl",
                    )
                )
    return sorted(
        rows,
        key=lambda row: (
            row.role,
            row.camera_group,
            int(row.canonical_capture_group)
            if row.canonical_capture_group.isdigit()
            else row.canonical_capture_group,
        ),
    )


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            (
                "numpy._core.multiarray",
                "_reconstruct",
            ): np._core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        try:
            return allowed[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"forbidden pickle global: {module}.{name}"
            ) from error


def decode_metadata(payload: bytes, config: Mapping[str, Any]) -> dict[str, Any]:
    """Decode only the exact numpy/basic metadata schema through a whitelist."""
    try:
        value = _RestrictedNumpyUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError, ValueError, TypeError) as error:
        raise RGB2RAWMetadataISPError("metadata restricted decode failed") from error
    required = set(config["metadata_decoder"]["required_keys"])
    if not isinstance(value, dict) or set(value) != required:
        raise RGB2RAWMetadataISPError("metadata key schema differs")
    white_level = int(value["white_level"])
    black = np.asarray(value["black_levels"], dtype=np.float64)
    capture = np.asarray(value["camera_whitebalance"], dtype=np.float64)
    daylight = np.asarray(value["daylight_whitebalance"], dtype=np.float64)
    matrix = np.asarray(value["color_matrix"], dtype=np.float64)
    cfa = np.asarray(value["cfa_pattern"])
    tone = np.asarray(value["tone_curve"])
    if white_level <= 0 or black.shape != (4,):
        raise RGB2RAWMetadataISPError("white/black metadata shape differs")
    if capture.shape != (4,) or daylight.shape != (4,):
        raise RGB2RAWMetadataISPError("white-balance metadata shape differs")
    if matrix.shape != (3, 4) or cfa.shape != (2, 2):
        raise RGB2RAWMetadataISPError("matrix/CFA metadata shape differs")
    if tone.shape != (65536,) or tone.dtype != np.uint16:
        raise RGB2RAWMetadataISPError("tone curve metadata differs")
    if not np.array_equal(tone, np.arange(65536, dtype=np.uint16)):
        raise RGB2RAWMetadataISPError("tone curve is not the frozen identity table")
    numeric = np.concatenate((black, capture, daylight, matrix.reshape(-1)))
    if not np.all(np.isfinite(numeric)) or capture[1] <= 0.0 or daylight[1] <= 0.0:
        raise RGB2RAWMetadataISPError("metadata contains invalid numeric values")
    if not isinstance(value["sizes"], dict):
        raise RGB2RAWMetadataISPError("sizes metadata is not a mapping")
    return {
        "white_level": white_level,
        "black_levels": black,
        "camera_whitebalance": capture,
        "daylight_whitebalance": daylight,
        "color_matrix": matrix,
        "cfa_pattern": cfa,
        "sizes": dict(value["sizes"]),
        "tone_curve": tone,
    }


def _wb_rgb(values: np.ndarray) -> np.ndarray:
    wb = np.asarray(values, dtype=np.float64)
    return np.asarray((wb[0] / wb[1], 1.0, wb[2] / wb[1]), dtype=np.float64)


def _gray_world_wb(base: np.ndarray) -> np.ndarray:
    medians = np.median(np.asarray(base, dtype=np.float64), axis=(0, 1))
    gains = medians[1] / np.maximum(medians, 1.0e-8)
    gains[1] = 1.0
    return np.clip(gains, 0.25, 4.0)


def _precondition(base: np.ndarray, wb: np.ndarray, epsilon: float) -> np.ndarray:
    return np.clip(np.asarray(base, dtype=np.float64) * wb, epsilon, 1.0 - epsilon)


def _decode_row(
    row: PairRow,
    *,
    url: str,
    archive_size: int,
    members: Mapping[str, ZipMember],
    config: Mapping[str, Any],
    range_reader: RangeReader,
) -> DecodedRow:
    payloads: dict[str, bytes] = {}
    consumed = 0
    identities: dict[str, str] = {}
    for role, name in (("npy", row.npy), ("png", row.png), ("metadata", row.metadata)):
        try:
            member = members[name]
        except KeyError as error:
            raise RGB2RAWMetadataISPError(f"selected member missing: {name}") from error
        payload, read_bytes = extract_member(url, member, archive_size, range_reader)
        payloads[role] = payload
        consumed += read_bytes
        identities[role] = sha256_bytes(payload)
    metadata = decode_metadata(payloads["metadata"], config)
    raw = np.load(io.BytesIO(payloads["npy"]), allow_pickle=False)
    if raw.shape != (512, 512, 4) or raw.dtype != np.uint16:
        raise RGB2RAWMetadataISPError("RAW array shape or dtype differs")
    normalized = np.asarray(raw, dtype=np.float64) / float(metadata["white_level"])
    if (
        not np.all(np.isfinite(normalized))
        or normalized.min() < 0.0
        or normalized.max() > 4.01
    ):
        raise RGB2RAWMetadataISPError("RAW normalization exceeds frozen bounds")
    base = np.stack(
        (
            normalized[..., 0],
            0.5 * (normalized[..., 1] + normalized[..., 2]),
            normalized[..., 3],
        ),
        axis=-1,
    ).astype(np.float32)
    with Image.open(io.BytesIO(payloads["png"])) as image:
        image.load()
        if image.size != (1024, 1024):
            raise RGB2RAWMetadataISPError("phone-ISP RGB shape differs")
        target = np.asarray(
            image.convert("RGB").resize((512, 512), Image.Resampling.BOX),
            dtype=np.float32,
        ) / np.float32(255.0)
    return DecodedRow(
        pair=row,
        base_rgb=base,
        target_rgb=target,
        capture_wb=_wb_rgb(metadata["camera_whitebalance"]),
        daylight_wb=_wb_rgb(metadata["daylight_whitebalance"]),
        member_sha256=identities,
        network_bytes=consumed,
    )


def _read_rows(
    rows: Sequence[PairRow],
    *,
    reverse: bool,
    url: str,
    archive_size: int,
    members: Mapping[str, ZipMember],
    config: Mapping[str, Any],
    range_reader: RangeReader,
) -> tuple[list[DecodedRow], int]:
    ordered = list(reversed(rows)) if reverse else list(rows)
    with ThreadPoolExecutor(max_workers=8) as executor:
        decoded = list(
            executor.map(
                lambda row: _decode_row(
                    row,
                    url=url,
                    archive_size=archive_size,
                    members=members,
                    config=config,
                    range_reader=range_reader,
                ),
                ordered,
            )
        )
    decoded.sort(
        key=lambda item: (
            item.pair.camera_group,
            int(item.pair.canonical_capture_group),
        )
    )
    return decoded, sum(item.network_bytes for item in decoded)


def _sample_indices(row: DecodedRow, count: int) -> np.ndarray:
    total = row.base_rgb.shape[0] * row.base_rgb.shape[1]
    if count > total:
        raise RGB2RAWMetadataISPError("fit sample exceeds pixel count")
    prefix = (
        f"sf3-a0t-fit-pixel-v1|{row.pair.camera_group}|"
        f"{row.pair.canonical_capture_group}|"
    ).encode()
    ranked = sorted(
        range(total),
        key=lambda index: hashlib.sha256(prefix + str(index).encode()).digest(),
    )[:count]
    return np.asarray(ranked, dtype=np.int64)


def _role_wb_map(rows: Sequence[DecodedRow]) -> dict[tuple[str, str], np.ndarray]:
    mapping: dict[tuple[str, str], np.ndarray] = {}
    for camera in ("iphone-x", "samsung-s9"):
        camera_rows = sorted(
            (row for row in rows if row.pair.camera_group == camera),
            key=lambda row: int(row.pair.canonical_capture_group),
        )
        for index, row in enumerate(camera_rows):
            mapping[(camera, row.pair.canonical_capture_group)] = camera_rows[
                (index + 1) % len(camera_rows)
            ].capture_wb
    return mapping


def _mode_wb(
    row: DecodedRow,
    mode: str,
    permuted: Mapping[tuple[str, str], np.ndarray],
) -> np.ndarray:
    if mode == "capture_wb":
        return row.capture_wb
    if mode == "static":
        return np.ones(3, dtype=np.float64)
    if mode == "daylight":
        return row.daylight_wb
    if mode == "gray_world":
        return _gray_world_wb(row.base_rgb)
    if mode == "permuted_capture_wb":
        return permuted[(row.pair.camera_group, row.pair.canonical_capture_group)]
    raise RGB2RAWMetadataISPError(f"unknown preconditioner mode: {mode}")


def fit_operators(
    rows: Sequence[DecodedRow], config: Mapping[str, Any]
) -> dict[str, dict[str, LogitAffineOperator]]:
    modes = ("capture_wb", "static", "daylight", "gray_world", "permuted_capture_wb")
    epsilon = float(config["input_interpretation"]["epsilon"])
    count = int(config["operator"]["fit_pixels_per_group"])
    ridge = float(config["operator"]["ridge_alpha"])
    permuted = _role_wb_map(rows)
    sampled_indexes = {
        (row.pair.camera_group, row.pair.canonical_capture_group): _sample_indices(
            row, count
        )
        for row in rows
    }
    operators: dict[str, dict[str, LogitAffineOperator]] = {}
    for mode in modes:
        operators[mode] = {}
        for camera in ("iphone-x", "samsung-s9"):
            source_parts: list[np.ndarray] = []
            target_parts: list[np.ndarray] = []
            for row in rows:
                if row.pair.camera_group != camera:
                    continue
                indexes = sampled_indexes[
                    (row.pair.camera_group, row.pair.canonical_capture_group)
                ]
                source = _precondition(
                    row.base_rgb,
                    _mode_wb(row, mode, permuted),
                    epsilon,
                ).reshape(-1, 3)[indexes]
                target = row.target_rgb.reshape(-1, 3)[indexes]
                source_parts.append(source)
                target_parts.append(target)
            matrix, bias = fit_operator_rows(
                np.concatenate(source_parts),
                np.concatenate(target_parts),
                ridge_alpha=ridge,
            )
            operators[mode][camera] = LogitAffineOperator(matrix, bias, 1.0)
    return operators


def _operator_record(operator: LogitAffineOperator) -> dict[str, Any]:
    return {
        "matrix": np.asarray(operator.matrix, dtype=np.float64).tolist(),
        "bias": np.asarray(operator.bias, dtype=np.float64).tolist(),
        "dose": float(operator.dose),
        **matrix_diagnostics(operator),
    }


def _operator_from_record(value: Mapping[str, Any]) -> LogitAffineOperator:
    return LogitAffineOperator(
        np.asarray(value["matrix"], dtype=np.float64),
        np.asarray(value["bias"], dtype=np.float64),
        float(value["dose"]),
    )


def score_role(
    rows: Sequence[DecodedRow],
    operators: Mapping[str, Mapping[str, LogitAffineOperator]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    epsilon = float(config["input_interpretation"]["epsilon"])
    modes = ("capture_wb", "static", "daylight", "gray_world", "permuted_capture_wb")
    legitimate = ("static", "daylight", "gray_world")
    permuted = _role_wb_map(rows)
    serialized = {
        mode: {
            camera: _operator_record(operator) for camera, operator in by_camera.items()
        }
        for mode, by_camera in operators.items()
    }
    per_row: list[dict[str, Any]] = []
    maximum_reload_error = 0.0
    for row in rows:
        errors: dict[str, float] = {}
        outputs: dict[str, np.ndarray] = {}
        camera = row.pair.camera_group
        for mode in modes:
            source = _precondition(row.base_rgb, _mode_wb(row, mode, permuted), epsilon)
            output = apply_operator(source, operators[mode][camera])
            reload_output = apply_operator(
                source, _operator_from_record(serialized[mode][camera])
            )
            maximum_reload_error = max(
                maximum_reload_error,
                float(
                    np.max(
                        np.abs(
                            output.astype(np.float64) - reload_output.astype(np.float64)
                        )
                    )
                ),
            )
            outputs[mode] = output
            errors[mode] = mean_oklab_error(output, row.target_rgb)
        errors["identity_raw_proxy"] = mean_oklab_error(
            np.clip(row.base_rgb, 0.0, 1.0), row.target_rgb
        )
        per_row.append(
            {
                "camera_group": camera,
                "canonical_capture_group": row.pair.canonical_capture_group,
                "pair_stem": row.pair.pair_stem,
                "errors": errors,
                "candidate_gradient_p999_ratio": gradient_p999_ratio(
                    np.clip(row.base_rgb, 0.0, 1.0), outputs["capture_wb"]
                ),
                "candidate_new_exact_boundary_fraction": new_exact_boundary_fraction(
                    np.clip(row.base_rgb, 0.0, 1.0), outputs["capture_wb"]
                ),
                "member_sha256": dict(row.member_sha256),
            }
        )
    aggregate_errors = {
        mode: float(np.mean([item["errors"][mode] for item in per_row]))
        for mode in (*modes, "identity_raw_proxy")
    }
    strongest = min(
        legitimate, key=lambda mode: (aggregate_errors[mode], legitimate.index(mode))
    )

    def reductions(control: str) -> np.ndarray:
        candidate = np.asarray([item["errors"]["capture_wb"] for item in per_row])
        baseline = np.asarray([item["errors"][control] for item in per_row])
        return (baseline - candidate) / np.maximum(baseline, 1.0e-12)

    strongest_reduction = reductions(strongest)
    permuted_reduction = reductions("permuted_capture_wb")
    candidate_errors = np.asarray([item["errors"]["capture_wb"] for item in per_row])
    camera_rates = {}
    for camera in ("iphone-x", "samsung-s9"):
        indexes = [
            index
            for index, item in enumerate(per_row)
            if item["camera_group"] == camera
        ]
        camera_rates[camera] = float(np.mean(strongest_reduction[indexes] > 0.0))
    summary = {
        "row_count": len(per_row),
        "aggregate_mean_oklab_error": aggregate_errors,
        "strongest_legitimate_control": strongest,
        "improvement_rate_vs_strongest_legitimate": float(
            np.mean(strongest_reduction > 0.0)
        ),
        "median_relative_error_reduction_vs_strongest_legitimate": float(
            np.median(strongest_reduction)
        ),
        "worst_relative_error_reduction_vs_strongest_legitimate": float(
            np.min(strongest_reduction)
        ),
        "each_camera_improvement_rate_vs_strongest_legitimate": camera_rates,
        "improvement_rate_vs_permuted_metadata": float(
            np.mean(permuted_reduction > 0.0)
        ),
        "median_relative_error_reduction_vs_permuted_metadata": float(
            np.median(permuted_reduction)
        ),
        "median_mean_oklab_error": float(np.median(candidate_errors)),
        "p95_mean_oklab_error": float(np.quantile(candidate_errors, 0.95)),
        "maximum_gradient_p999_ratio": float(
            max(item["candidate_gradient_p999_ratio"] for item in per_row)
        ),
        "maximum_new_exact_boundary_fraction": float(
            max(item["candidate_new_exact_boundary_fraction"] for item in per_row)
        ),
        "maximum_fresh_reload_error": maximum_reload_error,
    }
    return {"summary": summary, "rows": per_row}


def evaluate_gates(
    score: Mapping[str, Any],
    operators: Mapping[str, Mapping[str, LogitAffineOperator]],
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    summary = score["summary"]
    diagnostics = [
        matrix_diagnostics(operator) for operator in operators["capture_wb"].values()
    ]
    return {
        "row_count": summary["row_count"] == int(gates["required_rows"]),
        "rows_per_camera": all(
            sum(row["camera_group"] == camera for row in score["rows"])
            == int(gates["required_rows_per_camera"])
            for camera in ("iphone-x", "samsung-s9")
        ),
        "improvement_rate_vs_strongest_legitimate": summary[
            "improvement_rate_vs_strongest_legitimate"
        ]
        >= float(gates["minimum_improvement_rate_vs_strongest_legitimate"]),
        "median_gain_vs_strongest_legitimate": summary[
            "median_relative_error_reduction_vs_strongest_legitimate"
        ]
        >= float(
            gates["minimum_median_relative_error_reduction_vs_strongest_legitimate"]
        ),
        "worst_tail_vs_strongest_legitimate": summary[
            "worst_relative_error_reduction_vs_strongest_legitimate"
        ]
        >= float(
            gates["minimum_worst_relative_error_reduction_vs_strongest_legitimate"]
        ),
        "each_camera_improvement_rate": min(
            summary["each_camera_improvement_rate_vs_strongest_legitimate"].values()
        )
        >= float(gates["minimum_each_camera_improvement_rate_vs_strongest_legitimate"]),
        "improvement_rate_vs_permuted_metadata": summary[
            "improvement_rate_vs_permuted_metadata"
        ]
        >= float(gates["minimum_improvement_rate_vs_permuted_metadata"]),
        "median_gain_vs_permuted_metadata": summary[
            "median_relative_error_reduction_vs_permuted_metadata"
        ]
        >= float(gates["minimum_median_relative_error_reduction_vs_permuted_metadata"]),
        "median_absolute_error": summary["median_mean_oklab_error"]
        <= float(gates["maximum_median_mean_oklab_error"]),
        "p95_absolute_error": summary["p95_mean_oklab_error"]
        <= float(gates["maximum_p95_mean_oklab_error"]),
        "gradient_tail": summary["maximum_gradient_p999_ratio"]
        <= float(gates["maximum_gradient_p999_ratio"]),
        "new_exact_boundary": summary["maximum_new_exact_boundary_fraction"]
        <= float(gates["maximum_new_exact_boundary_fraction"]),
        "operator_determinant": min(item["determinant"] for item in diagnostics)
        >= float(gates["minimum_operator_determinant"]),
        "operator_condition_number": max(
            item["condition_number"] for item in diagnostics
        )
        <= float(gates["maximum_operator_condition_number"]),
        "operator_minimum_singular_value": min(
            item["minimum_singular_value"] for item in diagnostics
        )
        >= float(gates["minimum_operator_singular_value"]),
        "all_finite": all(
            np.isfinite(value)
            for value in (
                summary["median_mean_oklab_error"],
                summary["p95_mean_oklab_error"],
                summary["maximum_gradient_p999_ratio"],
            )
        ),
        "fresh_reload_exact": summary["maximum_fresh_reload_error"] == 0.0,
        "reverse_row_order_exact_by_canonicalized_execution": True,
    }


def run_metadata_isp_d0(
    config_path: Path,
    *,
    pre_calibration_lock_writer: Callable[[dict[str, Any]], None],
    reverse: bool = False,
    range_reader: RangeReader = http_range_get,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if (
        config.get("schema")
        != "neuro-film.sf3-a0t-rgb2raw-metadata-explicit-isp-d0-contract.v1"
    ):
        raise RGB2RAWMetadataISPError("contract schema differs")
    source = config["source"]
    url = (
        f"https://huggingface.co/datasets/{source['dataset_id']}/resolve/"
        f"{source['revision']}/{source['archive_path']}?download=true"
    )
    central = range_reader(
        url,
        int(source["central_offset"]),
        int(source["central_offset"]) + int(source["central_size"]) - 1,
        int(source["archive_size"]),
    )
    if sha256_bytes(central) != source["central_sha256"]:
        raise RGB2RAWMetadataISPError("central-directory identity differs")
    member_list = parse_central_directory(central)
    members = {member.name: member for member in member_list}
    role_rows = build_role_rows(member_list, config)
    role_lock = canonical_sha256([row.record() for row in role_rows])
    if role_lock != config["roles"]["expected_role_lock_sha256"]:
        raise RGB2RAWMetadataISPError("role-lock identity differs")
    fit_rows = [row for row in role_rows if row.role == "fit"]
    calibration_rows = [row for row in role_rows if row.role == "calibration"]
    sealed_rows = [row for row in role_rows if row.role == "sealed_confirmation"]
    fit_data, fit_bytes = _read_rows(
        fit_rows,
        reverse=reverse,
        url=url,
        archive_size=int(source["archive_size"]),
        members=members,
        config=config,
        range_reader=range_reader,
    )
    operators = fit_operators(fit_data, config)
    operator_records = {
        mode: {
            camera: _operator_record(operator) for camera, operator in values.items()
        }
        for mode, values in operators.items()
    }
    fit_member_identities = canonical_sha256(
        [
            {
                "camera_group": row.pair.camera_group,
                "canonical_capture_group": row.pair.canonical_capture_group,
                "pair_stem": row.pair.pair_stem,
                "member_sha256": dict(row.member_sha256),
            }
            for row in fit_data
        ]
    )
    pre_calibration_lock = {
        "schema": "neuro-film.sf3-a0t-rgb2raw-metadata-explicit-isp-prescore-lock.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "source_revision": source["revision"],
        "role_lock_sha256": role_lock,
        "operators": operator_records,
        "fit_member_identities_sha256": fit_member_identities,
        "fit_target_reads": len(fit_rows),
        "calibration_target_reads": 0,
        "sealed_target_reads": 0,
    }
    pre_calibration_lock["stable_identity"] = canonical_sha256(pre_calibration_lock)
    pre_calibration_lock_writer(pre_calibration_lock)
    del fit_data
    gc.collect()
    calibration_data, calibration_bytes = _read_rows(
        calibration_rows,
        reverse=reverse,
        url=url,
        archive_size=int(source["archive_size"]),
        members=members,
        config=config,
        range_reader=range_reader,
    )
    calibration = score_role(calibration_data, operators, config)
    calibration_gates = evaluate_gates(
        calibration, operators, config["calibration_gates"]
    )
    calibration_pass = all(calibration_gates.values())
    del calibration_data
    gc.collect()
    sealed: dict[str, Any] | None = None
    sealed_gates: dict[str, bool] | None = None
    sealed_bytes = 0
    if calibration_pass:
        sealed_data, sealed_bytes = _read_rows(
            sealed_rows,
            reverse=reverse,
            url=url,
            archive_size=int(source["archive_size"]),
            members=members,
            config=config,
            range_reader=range_reader,
        )
        sealed = score_role(sealed_data, operators, config)
        sealed_gates = evaluate_gates(sealed, operators, config["calibration_gates"])
        del sealed_data
        gc.collect()
    passed = (
        calibration_pass and sealed_gates is not None and all(sealed_gates.values())
    )
    stop = config["stop_rules"]
    decision = (
        stop["pass"]
        if passed
        else stop["confirmation_failure"]
        if calibration_pass
        else stop["calibration_failure"]
    )
    report = {
        "schema": "neuro-film.sf3-a0t-rgb2raw-metadata-explicit-isp-d0-report.v1",
        "experiment_id": config["experiment_id"],
        "contract_sha256": sha256_bytes(config_bytes),
        "source_revision": source["revision"],
        "role_lock_sha256": role_lock,
        "role_counts": {
            role: sum(row.role == role for row in role_rows)
            for role in ("fit", "calibration", "sealed_confirmation")
        },
        "cross_role_group_overlap": 0,
        "operators": operator_records,
        "fit_member_identities_sha256": fit_member_identities,
        "pre_calibration_lock_stable_identity": pre_calibration_lock["stable_identity"],
        "pre_calibration_lock_persisted_before_calibration_target_reads": True,
        "calibration": calibration,
        "calibration_gates": calibration_gates,
        "sealed_confirmation": sealed,
        "sealed_confirmation_gates": sealed_gates,
        "calibration_pass": calibration_pass,
        "automatic_pass": passed,
        "decision": decision,
        "network_bytes_read": len(central)
        + fit_bytes
        + calibration_bytes
        + sealed_bytes,
        "fit_target_reads": len(fit_rows),
        "calibration_target_reads": len(calibration_rows),
        "sealed_target_reads": len(sealed_rows) if calibration_pass else 0,
        "member_payloads_persisted": 0,
        "bounded_final_candidate_counter_before": 0,
        "bounded_final_candidate_counter_after": 1,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_identity"] = canonical_sha256(report)
    return report


__all__ = [
    "PairRow",
    "RGB2RAWMetadataISPError",
    "build_role_rows",
    "decode_metadata",
    "evaluate_gates",
    "run_metadata_isp_d0",
]
