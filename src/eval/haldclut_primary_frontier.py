"""Frozen U5.R2AJ0C1 primary-bank Hald structural frontier.

The module evaluates exact external Hald members only after the C1 contract
commit.  It keeps one native uint8 table resident, streams native-grid
differences, and never renders a photograph.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import math
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.stats import qmc

from src.eval.haldclut_structural import (
    HaldCube,
    _fit_basic_once,
    _lab,
    _strength_metrics,
    array_sha256,
    canonical_json_bytes,
    canonical_sha256,
    gradient_population,
    strict_json_loads,
    synthetic_cube,
    write_canonical_json,
)
from src.eval.spectral_film_lut_bank import median_delta_e76
from src.eval.velvia_datasheet_witness import encoded_srgb_to_linear


CONFIG_SCHEMA = "u5-r2aj0c1-hald-primary-structural-frontier-v1"
MANIFEST_SCHEMA = "u5-r2aj0c1-hald-primary-manifest-v1"
REPORT_SCHEMA = "u5-r2aj0c1-hald-primary-report-v1"
REPEAT_SCHEMA = "u5-r2aj0c1-hald-primary-repeat-decision-v1"
EXPECTED_CONFIG_RAW_SHA256 = (
    "7a6d492476ae97458d1a8948d6fe37a9c309c79efb6aa84813fc8f65a6044ef6"
)
SINGLE_STATE = "primary-structural-run-non-photographic"
HEX = frozenset("0123456789abcdef")
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


class HaldPrimaryError(RuntimeError):
    """Raised when a frozen C1 invariant is violated."""


def _expect_keys(
    value: Mapping[str, Any], expected: Iterable[str], *, label: str
) -> None:
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        raise HaldPrimaryError(
            f"{label} keys differ; missing={sorted(wanted-observed)}, "
            f"extra={sorted(observed-wanted)}"
        )


def _hex_digest(value: Any, *, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in HEX for character in value)
    ):
        raise HaldPrimaryError(f"invalid {label}")
    return value


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), count


def load_config_snapshot(
    path: Path, *, expected_raw_sha256: str | None = None
) -> tuple[dict[str, Any], str, str]:
    raw = path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    expected = expected_raw_sha256 or EXPECTED_CONFIG_RAW_SHA256
    if raw_sha != expected or raw_sha != EXPECTED_CONFIG_RAW_SHA256:
        raise HaldPrimaryError("C1 config raw SHA-256 differs")
    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise HaldPrimaryError("C1 config must be an object")
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise HaldPrimaryError("C1 config schema differs")
    validate_config(value)
    return value, raw_sha, canonical_sha256(value)


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "node",
        "parents",
        "archive",
        "primary_universe",
        "runtime",
        "source_contracts",
        "colour_math",
        "memory_contract",
        "automatic_gates",
        "eligibility",
        "safety_margin",
        "equivalence",
        "component_representative",
        "diversity_selection",
        "frontier_gate",
        "evidence",
        "branches",
        "forbidden",
        "claim_ceiling",
    }
    _expect_keys(config, required, label="C1 config")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise HaldPrimaryError("C1 config schema differs")
    if config["node"] != "U5.R2AJ0C1":
        raise HaldPrimaryError("C1 node differs")
    if int(config["primary_universe"]["count"]) != 194:
        raise HaldPrimaryError("C1 primary count differs")
    if config["primary_universe"]["filename_semantics_allowed"] is not False:
        raise HaldPrimaryError("filename semantics must remain forbidden")
    if config["memory_contract"]["complete_float_copy_allowed"] is not False:
        raise HaldPrimaryError("complete float copy must remain forbidden")
    if int(config["diversity_selection"]["maximum_selected"]) != 12:
        raise HaldPrimaryError("C1 selection cap differs")
    if int(config["frontier_gate"]["minimum_selected_representatives"]) != 1:
        raise HaldPrimaryError("C1 singleton gate differs")
    if config["frontier_gate"]["photograph_render_allowed_by_c1"] is not False:
        raise HaldPrimaryError("C1 may not render photographs")
    canonical_json_bytes(config)


def _runtime_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": importlib.metadata.version("numpy"),
        "Pillow": importlib.metadata.version("Pillow"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit-image": importlib.metadata.version("scikit-image"),
        "tifffile": importlib.metadata.version("tifffile"),
    }


def _tracked_clean(root: Path) -> bool:
    return (
        subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=root
        ).returncode
        == 0
    )


def _strict_object_file(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise HaldPrimaryError(f"parent evidence hash differs: {path}")
    value = strict_json_loads(raw)
    if not isinstance(value, dict):
        raise HaldPrimaryError("parent evidence must be an object")
    return value


def bind_parents(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    parents = config["parents"]
    aj0b2_decision = _strict_object_file(
        root / parents["aj0b2_decision_path"],
        expected_sha256=parents["aj0b2_decision_sha256"],
    )
    aj0b2_manifest = _strict_object_file(
        root / parents["aj0b2_manifest_path"],
        expected_sha256=parents["aj0b2_manifest_sha256"],
    )
    _strict_object_file(
        root / parents["aj0b2_report_path"],
        expected_sha256=parents["aj0b2_report_sha256"],
    )
    c0b = _strict_object_file(
        root / parents["c0b_decision_path"],
        expected_sha256=parents["c0b_decision_sha256"],
    )
    if aj0b2_decision.get("decision") != parents["aj0b2_required_decision"]:
        raise HaldPrimaryError("AJ0B2 decision state differs")
    if c0b.get("decision") != parents["c0b_required_decision"]:
        raise HaldPrimaryError("C0B decision state differs")
    if c0b.get("software_commit") != parents["c0b_implementation_commit"]:
        raise HaldPrimaryError("C0B implementation identity differs")
    formal = c0b.get("formal_evidence", {})
    for key in (
        "manifest_sha256",
        "report_sha256",
        "repeat_decision_sha256",
    ):
        if formal.get(key) != parents[f"c0b_{key}"]:
            raise HaldPrimaryError(f"C0B formal identity differs: {key}")
    universe = config["primary_universe"]
    if aj0b2_manifest.get("primary_paths_sha256") != universe[
        "primary_paths_sha256"
    ]:
        raise HaldPrimaryError("primary path identity differs")
    if aj0b2_manifest.get("image_records_sha256") != universe[
        "image_records_sha256"
    ]:
        raise HaldPrimaryError("image record identity differs")
    primary_paths = aj0b2_manifest.get("primary_paths")
    image_records = aj0b2_manifest.get("image_records")
    if not isinstance(primary_paths, list) or not isinstance(image_records, list):
        raise HaldPrimaryError("AJ0B2 manifest arrays missing")
    if len(primary_paths) != int(universe["count"]):
        raise HaldPrimaryError("AJ0B2 primary count differs")
    records_by_path = {
        str(record["path"]): record for record in image_records
    }
    if len(records_by_path) != len(image_records):
        raise HaldPrimaryError("AJ0B2 image record paths are not unique")
    try:
        primary_records = [records_by_path[str(path)] for path in primary_paths]
    except KeyError as exc:
        raise HaldPrimaryError("primary path lacks image record") from exc
    _validate_primary_metadata(primary_paths, primary_records, config)
    return {
        "aj0b2_decision_sha256": parents["aj0b2_decision_sha256"],
        "aj0b2_manifest_sha256": parents["aj0b2_manifest_sha256"],
        "aj0b2_report_sha256": parents["aj0b2_report_sha256"],
        "c0b_decision_sha256": parents["c0b_decision_sha256"],
        "c0b_repeat_decision_sha256": parents[
            "c0b_repeat_decision_sha256"
        ],
        "primary_paths": [str(path) for path in primary_paths],
        "primary_records": primary_records,
    }


def _validate_primary_metadata(
    primary_paths: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    universe = config["primary_universe"]
    if len(records) != int(universe["count"]):
        raise HaldPrimaryError("primary metadata count differs")
    total_bytes = 0
    total_compressed = 0
    sides: dict[int, int] = {}
    for path, record in zip(primary_paths, records, strict=True):
        if record.get("path") != path:
            raise HaldPrimaryError("primary metadata order differs")
        if record.get("format") != universe["required_format"]:
            raise HaldPrimaryError("primary format differs")
        if record.get("mode") != universe["required_mode"]:
            raise HaldPrimaryError("primary mode differs")
        if record.get("bits_per_sample") != universe[
            "required_bits_per_sample"
        ]:
            raise HaldPrimaryError("primary bit depth differs")
        if record.get("icc_profile_present") is not False:
            raise HaldPrimaryError("primary ICC state differs")
        side = int(record["cube_side"])
        sides[side] = sides.get(side, 0) + 1
        total_bytes += int(record["bytes"])
        total_compressed += int(record["compressed_bytes"])
        _hex_digest(record["sha256"], length=64, label="member SHA-256")
        _hex_digest(record["crc32"], length=8, label="member CRC32")
    if total_bytes != int(universe["uncompressed_bytes"]):
        raise HaldPrimaryError("primary uncompressed bytes differ")
    if total_compressed != int(universe["compressed_bytes"]):
        raise HaldPrimaryError("primary compressed bytes differ")
    expected_sides = {
        144: int(universe["level12_cube_side144_count"]),
        256: int(universe["level16_cube_side256_count"]),
    }
    if sides != expected_sides:
        raise HaldPrimaryError("primary native-side counts differ")


def source_contract_evidence(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, contract in config["source_contracts"].items():
        path = root / str(contract["path"])
        raw_sha, _count = sha256_file(path)
        blob = subprocess.check_output(
            ["git", "hash-object", str(path)], cwd=root, text=True
        ).strip()
        records[name] = {
            "path": str(contract["path"]),
            "sha256": raw_sha,
            "git_blob": blob,
            "checks": {
                "sha256": raw_sha == contract["sha256"],
                "git_blob": blob == contract["git_blob"],
            },
        }
    if not all(
        all(record["checks"].values()) for record in records.values()
    ):
        raise HaldPrimaryError("C1 source contract differs")
    return records


def population_bundle(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    populations = {
        "style": synthetic_cube(33),
        "residual": synthetic_cube(17),
        "sobol": qmc.Sobol(d=3, scramble=False).random_base2(m=16),
        "gradients": gradient_population(4097),
    }
    expected = config["colour_math"]["probe_populations_and_hashes"]
    mapping = {
        "style": "style_basic_probe_sha256",
        "residual": "residual_novelty_probe_sha256",
        "sobol": "sobol_probe_sha256",
        "gradients": "gradient_population_sha256",
    }
    for name, key in mapping.items():
        if array_sha256(populations[name]) != expected[key]:
            raise HaldPrimaryError(f"C1 population hash differs: {name}")
    return populations


@dataclass(frozen=True)
class NativeDifferenceEvidence:
    signature_sha256: str
    signature_length: int
    maximum_adjacent_code_step: int
    maximum_second_code_difference: int
    endpoint_count: int
    endpoint_total: int

    @property
    def endpoint_fraction(self) -> float:
        return float(self.endpoint_count / self.endpoint_total)


def _difference_shape(
    side: int, *, axis: int, order: int
) -> tuple[int, int, int, int]:
    shape = [side, side, side, 3]
    shape[axis] -= order
    return tuple(shape)


def _difference_chunks(
    table: np.ndarray,
    *,
    axis: int,
    order: int,
    slab_rows: int,
) -> Iterable[np.ndarray]:
    side = int(table.shape[0])
    if axis == 0:
        output_rows = side - order
        for start in range(0, output_rows, slab_rows):
            stop = min(output_rows, start + slab_rows)
            source = table[start : stop + order].astype(
                np.int16, copy=False
            )
            yield np.ascontiguousarray(
                np.diff(source, n=order, axis=axis).reshape(-1)
            )
    else:
        for start in range(0, side, slab_rows):
            stop = min(side, start + slab_rows)
            source = table[start:stop].astype(np.int16, copy=False)
            yield np.ascontiguousarray(
                np.diff(source, n=order, axis=axis).reshape(-1)
            )


def stream_native_evidence(
    table: np.ndarray,
    *,
    interior_margin: float,
    endpoint_epsilon: float,
    slab_rows: int = 8,
) -> NativeDifferenceEvidence:
    source = np.asarray(table)
    if (
        source.dtype != np.uint8
        or source.ndim != 4
        or source.shape[-1] != 3
        or len(set(source.shape[:3])) != 1
    ):
        raise HaldPrimaryError("native table must be cubic RGB uint8")
    side = int(source.shape[0])
    signature_length = 0
    for order in (1, 2):
        for axis in range(3):
            signature_length += math.prod(
                _difference_shape(side, axis=axis, order=order)
            )
    header = canonical_json_bytes(
        {"dtype": np.dtype(np.int16).str, "shape": [signature_length]}
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    maximum_first = 0
    maximum_second = 0
    for order in (1, 2):
        for axis in range(3):
            for chunk in _difference_chunks(
                source,
                axis=axis,
                order=order,
                slab_rows=slab_rows,
            ):
                digest.update(chunk.tobytes(order="C"))
                maximum = int(np.max(np.abs(chunk))) if chunk.size else 0
                if order == 1:
                    maximum_first = max(maximum_first, maximum)
                else:
                    maximum_second = max(maximum_second, maximum)
    axis_values = np.linspace(0.0, 1.0, side, dtype=np.float64)
    indices = np.flatnonzero(
        (axis_values > interior_margin)
        & (axis_values < 1.0 - interior_margin)
    )
    endpoint_count = 0
    endpoint_total = int(len(indices) ** 3 * 3)
    for start in range(0, len(indices), slab_rows):
        red = indices[start : start + slab_rows]
        slab = source[np.ix_(red, indices, indices, np.arange(3))]
        normalized = slab.astype(np.float64) / 255.0
        endpoint_count += int(
            np.count_nonzero(
                (normalized <= endpoint_epsilon)
                | (normalized >= 1.0 - endpoint_epsilon)
            )
        )
    return NativeDifferenceEvidence(
        signature_sha256=digest.hexdigest(),
        signature_length=signature_length,
        maximum_adjacent_code_step=maximum_first,
        maximum_second_code_difference=maximum_second,
        endpoint_count=endpoint_count,
        endpoint_total=endpoint_total,
    )


def evaluate_cube(
    cube: HaldCube,
    *,
    config: Mapping[str, Any],
    populations: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, np.ndarray], float]:
    if cube.table.dtype != np.uint8:
        raise HaldPrimaryError("C1 primary table must remain uint8")
    gates = config["automatic_gates"]
    style = populations["style"]
    residual = populations["residual"]
    style_output = cube.apply(style)
    residual_output = cube.apply(residual)
    jacobian = cube.jacobian(populations["sobol"])
    determinants = np.linalg.det(jacobian)
    spectral_norms = np.linalg.svd(jacobian, compute_uv=False)[..., 0]
    gradients = populations["gradients"]
    gradient_output = cube.apply(gradients.reshape(-1, 3)).reshape(
        gradients.shape
    )
    input_steps = np.linalg.norm(np.diff(gradients, axis=1), axis=-1)
    output_steps = np.linalg.norm(
        np.diff(gradient_output, axis=1), axis=-1
    )
    gains = output_steps / input_steps
    second = np.diff(gradient_output, n=2, axis=1)
    neutral_luma = encoded_srgb_to_linear(gradient_output[0]) @ LUMA
    neutral_reversal = float(
        max(0.0, -float(np.min(np.diff(neutral_luma))))
    )
    margin = float(gates["interior_input_margin"])
    epsilon = float(gates["endpoint_epsilon"])
    interior = np.all((style > margin) & (style < 1.0 - margin), axis=-1)
    interior_output = style_output[interior]
    probe_endpoint_fraction = float(
        np.mean(
            (interior_output <= epsilon)
            | (interior_output >= 1.0 - epsilon)
        )
    )
    native = stream_native_evidence(
        cube.table,
        interior_margin=margin,
        endpoint_epsilon=epsilon,
    )
    endpoint_fraction = max(
        probe_endpoint_fraction, native.endpoint_fraction
    )
    style_basic, residual_basic = _fit_basic_once(
        style, style_output, residual
    )
    residual_signature = _lab(residual_output) - _lab(residual_basic)
    metrics: dict[str, Any] = {
        "finite": bool(
            np.all(np.isfinite(style_output))
            and np.all(np.isfinite(residual_output))
            and np.all(np.isfinite(jacobian))
            and np.all(np.isfinite(gradient_output))
        ),
        "output_minimum": float(np.min(style_output)),
        "output_maximum": float(np.max(style_output)),
        "probe_new_interior_endpoint_fraction": probe_endpoint_fraction,
        "native_new_interior_endpoint_fraction": native.endpoint_fraction,
        "new_interior_endpoint_fraction": endpoint_fraction,
        "negative_jacobian_fraction": float(
            np.mean(
                determinants < float(gates["negative_jacobian_threshold"])
            )
        ),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "maximum_jacobian_spectral_norm": float(np.max(spectral_norms)),
        "neutral_luma_reversal": neutral_reversal,
        "gradient_rgb_gain_p99": float(np.quantile(gains, 0.99)),
        "maximum_gradient_rgb_gain": float(np.max(gains)),
        "maximum_gradient_channel_second_difference": float(
            np.max(np.abs(second))
        ),
        "native_maximum_adjacent_code_step": (
            native.maximum_adjacent_code_step
        ),
        "native_maximum_second_code_difference": (
            native.maximum_second_code_difference
        ),
        "style_delta_e76_median": median_delta_e76(style, style_output),
        "joint_basic_residual_delta_e76_median": median_delta_e76(
            style_basic, style_output
        ),
    }
    checks = safety_checks(metrics, gates)
    safety_margin = normalized_safety_margin(metrics, gates)
    signatures = {
        "style_output": style_output,
        "residual_signature": residual_signature,
        "effect": residual_output - residual,
        "jacobian": jacobian,
        "gradient": gradient_output,
    }
    hashes = {
        "style_output_array_sha256": array_sha256(style_output),
        "residual_signature_array_sha256": array_sha256(
            residual_signature
        ),
        "jacobian_signature_array_sha256": array_sha256(jacobian),
        "gradient_signature_array_sha256": array_sha256(gradient_output),
        "native_difference_signature_array_sha256": (
            native.signature_sha256
        ),
    }
    return metrics | {"hashes": hashes}, checks, signatures, safety_margin


def safety_checks(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "finite": metrics["finite"] is bool(gates["all_outputs_finite"]),
        "range": metrics["output_minimum"] >= float(gates["output_minimum"])
        and metrics["output_maximum"] <= float(gates["output_maximum"]),
        "new_interior_endpoint_fraction": metrics[
            "new_interior_endpoint_fraction"
        ]
        <= float(gates["maximum_new_interior_endpoint_fraction"]),
        "negative_jacobian_fraction": metrics[
            "negative_jacobian_fraction"
        ]
        <= float(gates["maximum_negative_jacobian_fraction"]),
        "minimum_jacobian_determinant": metrics[
            "minimum_jacobian_determinant"
        ]
        >= float(gates["minimum_jacobian_determinant"]),
        "maximum_jacobian_spectral_norm": metrics[
            "maximum_jacobian_spectral_norm"
        ]
        <= float(gates["maximum_jacobian_spectral_norm"]),
        "neutral_luma_reversal": metrics["neutral_luma_reversal"]
        <= float(gates["maximum_neutral_luma_reversal"]),
        "gradient_rgb_gain_p99": metrics["gradient_rgb_gain_p99"]
        <= float(gates["maximum_gradient_rgb_gain_p99"]),
        "maximum_gradient_rgb_gain": metrics["maximum_gradient_rgb_gain"]
        <= float(gates["maximum_gradient_rgb_gain"]),
        "maximum_gradient_channel_second_difference": metrics[
            "maximum_gradient_channel_second_difference"
        ]
        <= float(gates["maximum_gradient_channel_second_difference"]),
        "native_maximum_adjacent_code_step": metrics[
            "native_maximum_adjacent_code_step"
        ]
        <= int(gates["native_maximum_adjacent_code_step"]),
        "native_maximum_second_code_difference": metrics[
            "native_maximum_second_code_difference"
        ]
        <= int(gates["native_maximum_second_code_difference"]),
    }


def normalized_safety_margin(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> float:
    pairs = (
        (
            "new_interior_endpoint_fraction",
            "maximum_new_interior_endpoint_fraction",
        ),
        (
            "negative_jacobian_fraction",
            "maximum_negative_jacobian_fraction",
        ),
        (
            "maximum_jacobian_spectral_norm",
            "maximum_jacobian_spectral_norm",
        ),
        ("neutral_luma_reversal", "maximum_neutral_luma_reversal"),
        ("gradient_rgb_gain_p99", "maximum_gradient_rgb_gain_p99"),
        ("maximum_gradient_rgb_gain", "maximum_gradient_rgb_gain"),
        (
            "maximum_gradient_channel_second_difference",
            "maximum_gradient_channel_second_difference",
        ),
        (
            "native_maximum_adjacent_code_step",
            "native_maximum_adjacent_code_step",
        ),
        (
            "native_maximum_second_code_difference",
            "native_maximum_second_code_difference",
        ),
    )
    margins = []
    for metric_name, gate_name in pairs:
        limit = float(gates[gate_name])
        margins.append(
            float(np.clip((limit - float(metrics[metric_name])) / limit, 0, 1))
        )
    determinant = float(metrics["minimum_jacobian_determinant"])
    margins.append(float(np.clip((determinant + 0.05) / 0.05, 0, 1)))
    return min(margins)


def candidate_identity(path: str, member_sha256: str) -> dict[str, str]:
    path_hash = hashlib.sha256(path.encode("utf-8")).hexdigest()
    identity = canonical_sha256(
        {
            "member_sha256": member_sha256,
            "path_identity_sha256": path_hash,
        }
    )
    return {
        "candidate_id": identity,
        "path_identity_sha256": path_hash,
    }


def _decode_member(
    archive: zipfile.ZipFile,
    *,
    path: str,
    record: Mapping[str, Any],
) -> tuple[HaldCube, dict[str, Any], int]:
    try:
        info = archive.getinfo(path)
    except KeyError as exc:
        raise HaldPrimaryError(f"primary member missing: {path}") from exc
    expected_crc = int(str(record["crc32"]), 16)
    if (
        int(info.CRC) != expected_crc
        or int(info.file_size) != int(record["bytes"])
        or int(info.compress_size) != int(record["compressed_bytes"])
    ):
        raise HaldPrimaryError(f"primary ZIP identity differs: {path}")
    raw = archive.read(info)
    if len(raw) != int(record["bytes"]):
        raise HaldPrimaryError(f"primary member length differs: {path}")
    member_sha = hashlib.sha256(raw).hexdigest()
    if member_sha != record["sha256"]:
        raise HaldPrimaryError(f"primary member SHA-256 differs: {path}")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size
                != (int(record["width"]), int(record["height"]))
                or image.info.get("icc_profile") is not None
            ):
                raise HaldPrimaryError(
                    f"primary decoded profile differs: {path}"
                )
            raster = np.asarray(image)
    except HaldPrimaryError:
        raise
    except Exception as exc:
        raise HaldPrimaryError(f"primary decode failed: {path}") from exc
    if raster.dtype != np.uint8 or not raster.flags.c_contiguous:
        raster = np.ascontiguousarray(raster, dtype=np.uint8)
    cube = HaldCube.from_raster(raster)
    if cube.side != int(record["cube_side"]):
        raise HaldPrimaryError(f"primary cube side differs: {path}")
    identity = candidate_identity(path, member_sha)
    source = {
        **identity,
        "archive_member_path": path,
        "member_bytes": len(raw),
        "member_bytes_sha256": member_sha,
        "crc32": str(record["crc32"]),
        "hald_level": int(record["hald_level"]),
        "cube_side": int(record["cube_side"]),
        "native_table_array_sha256": array_sha256(cube.table),
    }
    return cube, source, len(raw)


def _candidate_state(
    *,
    structurally_safe: bool,
    style_check: bool,
    non_basic_check: bool,
) -> str:
    if not structurally_safe:
        return "structural_veto"
    if not style_check:
        return "style_ineligible"
    if not non_basic_check:
        return "basic_explainable"
    return "eligible"


def evaluate_archive(
    *,
    root: Path,
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    populations = population_bundle(config)
    gates = config["automatic_gates"]
    archive_path = root / str(config["archive"]["path"])
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or any(parent.is_symlink() for parent in archive_path.parents)
    ):
        raise HaldPrimaryError("archive path must be a regular non-symlink file")
    records: list[dict[str, Any]] = []
    effects: dict[str, np.ndarray] = {}
    residual_signatures: dict[str, np.ndarray] = {}
    archive_digest = hashlib.sha256()
    archive_raw_bytes = 0
    primary_bytes = 0
    with archive_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            archive_digest.update(chunk)
            archive_raw_bytes += len(chunk)
        if archive_raw_bytes != int(config["archive"]["bytes"]):
            raise HaldPrimaryError("archive byte count differs")
        if archive_digest.hexdigest() != config["archive"]["sha256"]:
            raise HaldPrimaryError("archive SHA-256 differs")
        handle.seek(0)
        with zipfile.ZipFile(handle, "r") as archive:
            paths = parent["primary_paths"]
            metadata = parent["primary_records"]
            for path, source_record in zip(paths, metadata, strict=True):
                cube, source, member_bytes = _decode_member(
                    archive, path=path, record=source_record
                )
                primary_bytes += member_bytes
                metrics_with_hashes, checks, signatures, margin = (
                    evaluate_cube(
                        cube,
                        config=config,
                        populations=populations,
                    )
                )
                metric_hashes = metrics_with_hashes.pop("hashes")
                metrics = metrics_with_hashes
                style_check = metrics[
                    "style_delta_e76_median"
                ] >= float(gates["minimum_style_delta_e76_median"])
                non_basic_check = metrics[
                    "joint_basic_residual_delta_e76_median"
                ] >= float(
                    gates["minimum_joint_basic_residual_delta_e76_median"]
                )
                structurally_safe = all(checks.values())
                state = _candidate_state(
                    structurally_safe=structurally_safe,
                    style_check=style_check,
                    non_basic_check=non_basic_check,
                )
                candidate_id = source["candidate_id"]
                effects[candidate_id] = signatures["effect"]
                residual_signatures[candidate_id] = signatures[
                    "residual_signature"
                ]
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "source": source,
                        "hashes": {
                            "member_bytes_sha256": source[
                                "member_bytes_sha256"
                            ],
                            "native_table_array_sha256": source[
                                "native_table_array_sha256"
                            ],
                            **metric_hashes,
                        },
                        "metrics": metrics,
                        "safety_checks": checks,
                        "normalized_safety_margin": margin,
                        "structurally_safe": structurally_safe,
                        "style_check": bool(style_check),
                        "non_basic_check": bool(non_basic_check),
                        "eligible": state == "eligible",
                        "state": state,
                    }
                )
                del cube, signatures
    access = {
        "archive_open_calls": 1,
        "archive_raw_bytes_hashed": archive_raw_bytes,
        "primary_members_read": len(records),
        "primary_uncompressed_bytes": primary_bytes,
        "non_primary_member_bodies_read": 0,
        "photograph_renders": 0,
    }
    if access != config["evidence"]["required_access_ledger"]:
        raise HaldPrimaryError("C1 access ledger differs")
    frontier = build_frontier(
        records=records,
        effects=effects,
        residual_signatures=residual_signatures,
        config=config,
    )
    source_records = [
        {
            "candidate_id": row["candidate_id"],
            "path_identity_sha256": row["source"][
                "path_identity_sha256"
            ],
            "member_bytes_sha256": row["source"]["member_bytes_sha256"],
            "native_table_array_sha256": row["source"][
                "native_table_array_sha256"
            ],
            "cube_side": row["source"]["cube_side"],
        }
        for row in records
    ]
    manifest = {
        "population_hashes": {
            "style_basic_probe_sha256": array_sha256(populations["style"]),
            "residual_novelty_probe_sha256": array_sha256(
                populations["residual"]
            ),
            "sobol_probe_sha256": array_sha256(populations["sobol"]),
            "gradient_population_sha256": array_sha256(
                populations["gradients"]
            ),
        },
        "candidate_sources": source_records,
    }
    report = {
        "records": records,
        **frontier,
    }
    return (
        {"manifest": manifest, "report": report, "access_ledger": access},
        {
            "effects": effects,
            "residual_signatures": residual_signatures,
        },
    )


def _record_map(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result = {str(row["candidate_id"]): row for row in records}
    if len(result) != len(records):
        raise HaldPrimaryError("candidate identities are not unique")
    return result


def _exact_groups(
    ids: Sequence[str], records: Mapping[str, Mapping[str, Any]]
) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for candidate_id in ids:
        native_hash = str(
            records[candidate_id]["hashes"]["native_table_array_sha256"]
        )
        groups.setdefault(native_hash, []).append(candidate_id)
    return sorted(
        (sorted(group) for group in groups.values()),
        key=lambda group: group[0],
    )


def _strength_pass(
    metrics: Mapping[str, float], gates: Mapping[str, Any]
) -> bool:
    return bool(
        metrics["effect_cosine"]
        >= float(gates["minimum_positive_effect_cosine"])
        and metrics["minimum_explained_energy_fraction"]
        >= float(gates["minimum_explained_energy_fraction"])
        and metrics["bilateral_normalized_residual"]
        <= float(gates["maximum_bilateral_normalized_residual"])
    )


def strength_components(
    *,
    ordered_ids: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    effects: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[list[list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    exact_groups = _exact_groups(ordered_ids, records)
    group_ids = [canonical_sha256(group) for group in exact_groups]
    members = dict(zip(group_ids, exact_groups, strict=True))
    representative = {key: values[0] for key, values in members.items()}
    adjacency = {key: set() for key in group_ids}
    pairwise = []
    gates = config["equivalence"]["strength"]
    for left_index, left_group in enumerate(group_ids):
        for right_group in group_ids[left_index + 1 :]:
            left = representative[left_group]
            right = representative[right_group]
            metrics = _strength_metrics(effects[left], effects[right])
            passed = _strength_pass(metrics, gates)
            if passed:
                adjacency[left_group].add(right_group)
                adjacency[right_group].add(left_group)
            pairwise.append(
                {
                    "left_exact_group_id": left_group,
                    "right_exact_group_id": right_group,
                    "metrics": metrics,
                    "strength_equivalent": passed,
                }
            )
    unseen = set(group_ids)
    components: list[list[str]] = []
    ambiguities = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        graph_component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in graph_component:
                continue
            graph_component.add(current)
            stack.extend(sorted(adjacency[current] - graph_component))
        unseen -= graph_component
        ordered_groups = sorted(graph_component)
        clique = all(
            right in adjacency[left]
            for left_index, left in enumerate(ordered_groups)
            for right in ordered_groups[left_index + 1 :]
        )
        if clique:
            components.append(
                sorted(
                    candidate
                    for group_id in ordered_groups
                    for candidate in members[group_id]
                )
            )
        else:
            missing = [
                [left, right]
                for left_index, left in enumerate(ordered_groups)
                for right in ordered_groups[left_index + 1 :]
                if right not in adjacency[left]
            ]
            ambiguities.append(
                {
                    "exact_group_ids": ordered_groups,
                    "missing_edges": missing,
                    "decision": "retain_exact_groups",
                }
            )
            components.extend(
                sorted(members[group_id]) for group_id in ordered_groups
            )
    components = sorted(
        (sorted(component) for component in components),
        key=lambda component: component[0],
    )
    return components, pairwise, ambiguities


def _representative_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = record["metrics"]
    return (
        -float(metrics["style_delta_e76_median"]),
        -float(metrics["joint_basic_residual_delta_e76_median"]),
        -float(record["normalized_safety_margin"]),
        str(record["hashes"]["native_table_array_sha256"]),
        str(record["source"]["path_identity_sha256"]),
    )


def choose_representatives(
    components: Sequence[Sequence[str]],
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for component in components:
        representative = min(component, key=lambda value: _representative_key(
            records[value]
        ))
        result.append(
            {
                "component_id": canonical_sha256(sorted(component)),
                "members": sorted(component),
                "representative": representative,
                "ranking": {
                    "style_delta_e76_median": records[representative][
                        "metrics"
                    ]["style_delta_e76_median"],
                    "joint_basic_residual_delta_e76_median": records[
                        representative
                    ]["metrics"][
                        "joint_basic_residual_delta_e76_median"
                    ],
                    "normalized_safety_margin": records[representative][
                        "normalized_safety_margin"
                    ],
                    "native_table_array_sha256": records[representative][
                        "hashes"
                    ]["native_table_array_sha256"],
                    "path_identity_sha256": records[representative][
                        "source"
                    ]["path_identity_sha256"],
                },
            }
        )
    return sorted(result, key=lambda row: row["component_id"])


def residual_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.median(
            np.linalg.norm(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64),
                axis=-1,
            )
        )
    )


def select_diverse(
    *,
    representatives: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    residual_signatures: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    ids = [str(row["representative"]) for row in representatives]
    distances: dict[tuple[str, str], float] = {}
    distance_rows = []
    for left_index, left in enumerate(sorted(ids)):
        for right in sorted(ids)[left_index + 1 :]:
            distance = residual_distance(
                residual_signatures[left], residual_signatures[right]
            )
            distances[(left, right)] = distance
            distance_rows.append(
                {
                    "left": left,
                    "right": right,
                    "residual_delta_e76_median": distance,
                }
            )
    if not ids:
        return [], distance_rows, []
    selected = [min(ids, key=lambda value: _representative_key(
        records[value]
    ))]
    trace = [
        {
            "step": 0,
            "candidate_id": selected[0],
            "minimum_distance_to_selected": None,
        }
    ]
    remaining = set(ids) - set(selected)
    minimum_distance = float(
        config["diversity_selection"]["minimum_distinct_distance"]
    )
    maximum = int(config["diversity_selection"]["maximum_selected"])

    def distance(left: str, right: str) -> float:
        key = tuple(sorted((left, right)))
        return distances[key]

    while remaining and len(selected) < maximum:
        scored = {
            candidate: min(
                distance(candidate, chosen) for chosen in selected
            )
            for candidate in remaining
        }
        candidate = min(
            remaining,
            key=lambda value: (
                -scored[value],
                *_representative_key(records[value]),
            ),
        )
        if scored[candidate] < minimum_distance:
            break
        selected.append(candidate)
        remaining.remove(candidate)
        trace.append(
            {
                "step": len(selected) - 1,
                "candidate_id": candidate,
                "minimum_distance_to_selected": scored[candidate],
            }
        )
    return selected, distance_rows, trace


def build_frontier(
    *,
    records: Sequence[Mapping[str, Any]],
    effects: Mapping[str, np.ndarray],
    residual_signatures: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    record_map = _record_map(records)
    eligible_ids = [
        str(row["candidate_id"]) for row in records if row["eligible"]
    ]
    orders = {
        "parent_primary_order": eligible_ids,
        "reverse_parent_primary_order": list(reversed(eligible_ids)),
        "sha256_candidate_identity_order": sorted(
            eligible_ids,
            key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
        ),
    }
    results = {}
    pairwise = {}
    ambiguities = {}
    for name, order in orders.items():
        components, pairs, ambiguous = strength_components(
            ordered_ids=order,
            records=record_map,
            effects=effects,
            config=config,
        )
        results[name] = components
        pairwise[name] = pairs
        ambiguities[name] = ambiguous
    invariant = all(
        components == results["parent_primary_order"]
        for components in results.values()
    ) and all(
        value == ambiguities["parent_primary_order"]
        for value in ambiguities.values()
    )
    if not invariant:
        raise HaldPrimaryError("C1 component result is order dependent")
    components = results["parent_primary_order"]
    representatives = choose_representatives(components, record_map)
    selected, distance_rows, trace = select_diverse(
        representatives=representatives,
        records=record_map,
        residual_signatures=residual_signatures,
        config=config,
    )
    passed = len(selected) >= int(
        config["frontier_gate"]["minimum_selected_representatives"]
    )
    claim = (
        "structurally_distinct_frontier"
        if len(selected) >= 2
        else (
            "singleton_structural_challenger_not_bank_diversity"
            if len(selected) == 1
            else "no_structural_challenger"
        )
    )
    counts: dict[str, int] = {}
    for row in records:
        counts[str(row["state"])] = counts.get(str(row["state"]), 0) + 1
    return {
        "candidate_state_counts": counts,
        "eligible_candidate_ids": eligible_ids,
        "components_by_order": results,
        "strength_pairwise_parent_order": pairwise[
            "parent_primary_order"
        ],
        "nontransitive_strength_ambiguities": ambiguities[
            "parent_primary_order"
        ],
        "components_permutation_invariant": invariant,
        "component_representatives": representatives,
        "residual_distance_rows": distance_rows,
        "selection_trace": trace,
        "selected_candidate_ids": selected,
        "selected_count": len(selected),
        "frontier_claim": claim,
        "frontier_pass": passed,
        "photograph_rendered": False,
    }


COMMON_KEYS = {
    "experiment_id",
    "node",
    "state",
    "software_commit",
    "tracked_worktree_clean",
    "config",
    "runtime",
    "parent_evidence",
    "source_contracts",
    "access_ledger",
    "filename_semantics_used",
    "photograph_rendered",
    "visual_review_allowed",
    "evidence_promoted",
    "photographic_contract_design_allowed",
    "claim_ceiling",
}


def _common_evidence(
    *,
    config: Mapping[str, Any],
    config_raw_sha256: str,
    config_canonical_sha256: str,
    software_commit: str,
    parent: Mapping[str, Any],
    sources: Mapping[str, Any],
    access: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "state": SINGLE_STATE,
        "software_commit": software_commit,
        "tracked_worktree_clean": True,
        "config": {
            "raw_sha256_before": config_raw_sha256,
            "raw_sha256_after": config_raw_sha256,
            "canonical_sha256": config_canonical_sha256,
        },
        "runtime": _runtime_versions(),
        "parent_evidence": {
            key: value
            for key, value in parent.items()
            if key not in {"primary_paths", "primary_records"}
        },
        "source_contracts": sources,
        "access_ledger": dict(access),
        "filename_semantics_used": False,
        "photograph_rendered": False,
        "visual_review_allowed": False,
        "evidence_promoted": False,
        "photographic_contract_design_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def run_single_process(
    *,
    root: Path,
    config_path: Path,
    output_dir: Path,
    expected_config_raw_sha256: str,
    software_commit: str,
) -> dict[str, str]:
    config, raw_sha, canonical_sha = load_config_snapshot(
        config_path, expected_raw_sha256=expected_config_raw_sha256
    )
    if not _tracked_clean(root):
        raise HaldPrimaryError("tracked worktree must be clean for C1")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if software_commit != head:
        raise HaldPrimaryError("C1 software commit must equal HEAD")
    if _runtime_versions() != (
        config["runtime"]["packages"]
        | {"python": config["runtime"]["python"]}
    ):
        raise HaldPrimaryError("C1 runtime differs")
    parent = bind_parents(root, config)
    sources = source_contract_evidence(root, config)
    evaluated, _private = evaluate_archive(
        root=root, config=config, parent=parent
    )
    _config_after, after_sha, after_canonical = load_config_snapshot(
        config_path, expected_raw_sha256=raw_sha
    )
    if after_sha != raw_sha or after_canonical != canonical_sha:
        raise HaldPrimaryError("C1 config changed during execution")
    common = _common_evidence(
        config=config,
        config_raw_sha256=raw_sha,
        config_canonical_sha256=canonical_sha,
        software_commit=software_commit,
        parent=parent,
        sources=sources,
        access=evaluated["access_ledger"],
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        **common,
        **evaluated["manifest"],
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        **common,
        **evaluated["report"],
    }
    validate_manifest(manifest, config)
    validate_report(report, config)
    manifest_sha = write_canonical_json(
        output_dir / "manifest.json", manifest
    )
    report_sha = write_canonical_json(output_dir / "report.json", report)
    return {
        "manifest_sha256": manifest_sha,
        "report_sha256": report_sha,
    }


def _validate_common(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    if value["experiment_id"] != config["experiment_id"]:
        raise HaldPrimaryError("C1 experiment identity differs")
    if value["node"] != config["node"] or value["state"] != SINGLE_STATE:
        raise HaldPrimaryError("C1 node/state differs")
    _hex_digest(value["software_commit"], length=40, label="software commit")
    for key in (
        "tracked_worktree_clean",
        "filename_semantics_used",
        "photograph_rendered",
        "visual_review_allowed",
        "evidence_promoted",
        "photographic_contract_design_allowed",
    ):
        if not isinstance(value[key], bool):
            raise HaldPrimaryError(f"C1 Boolean type differs: {key}")
    if value["tracked_worktree_clean"] is not True:
        raise HaldPrimaryError("C1 tracked worktree was not clean")
    if any(
        value[key] is not False
        for key in (
            "filename_semantics_used",
            "photograph_rendered",
            "visual_review_allowed",
            "evidence_promoted",
            "photographic_contract_design_allowed",
        )
    ):
        raise HaldPrimaryError("single C1 child exceeds claim boundary")
    _expect_keys(
        value["config"],
        {"raw_sha256_before", "raw_sha256_after", "canonical_sha256"},
        label="C1 config evidence",
    )
    if (
        value["config"]["raw_sha256_before"] != EXPECTED_CONFIG_RAW_SHA256
        or value["config"]["raw_sha256_after"] != EXPECTED_CONFIG_RAW_SHA256
        or value["config"]["canonical_sha256"] != canonical_sha256(config)
    ):
        raise HaldPrimaryError("C1 config evidence differs")
    expected_runtime = config["runtime"]["packages"] | {
        "python": config["runtime"]["python"]
    }
    if value["runtime"] != expected_runtime:
        raise HaldPrimaryError("C1 runtime evidence differs")
    expected_parent = {
        "aj0b2_decision_sha256": config["parents"][
            "aj0b2_decision_sha256"
        ],
        "aj0b2_manifest_sha256": config["parents"][
            "aj0b2_manifest_sha256"
        ],
        "aj0b2_report_sha256": config["parents"]["aj0b2_report_sha256"],
        "c0b_decision_sha256": config["parents"]["c0b_decision_sha256"],
        "c0b_repeat_decision_sha256": config["parents"][
            "c0b_repeat_decision_sha256"
        ],
    }
    if value["parent_evidence"] != expected_parent:
        raise HaldPrimaryError("C1 parent evidence differs")
    _expect_keys(
        value["source_contracts"],
        set(config["source_contracts"]),
        label="C1 source contracts",
    )
    for name, contract in config["source_contracts"].items():
        observed = value["source_contracts"][name]
        _expect_keys(
            observed,
            {"path", "sha256", "git_blob", "checks"},
            label=f"C1 source contract {name}",
        )
        _expect_keys(
            observed["checks"],
            {"sha256", "git_blob"},
            label=f"C1 source checks {name}",
        )
        if (
            observed["path"] != contract["path"]
            or observed["sha256"] != contract["sha256"]
            or observed["git_blob"] != contract["git_blob"]
            or observed["checks"] != {"sha256": True, "git_blob": True}
        ):
            raise HaldPrimaryError(f"C1 source evidence differs: {name}")
    if value["access_ledger"] != config["evidence"][
        "required_access_ledger"
    ]:
        raise HaldPrimaryError("C1 access evidence differs")
    if value["claim_ceiling"] != config["claim_ceiling"]:
        raise HaldPrimaryError("C1 claim ceiling differs")


def validate_manifest(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _expect_keys(
        value,
        COMMON_KEYS
        | {
            "schema_version",
            "population_hashes",
            "candidate_sources",
        },
        label="C1 manifest",
    )
    _validate_common(value, config)
    if value["schema_version"] != MANIFEST_SCHEMA:
        raise HaldPrimaryError("C1 manifest schema differs")
    expected_population = config["colour_math"][
        "probe_populations_and_hashes"
    ]
    if value["population_hashes"] != expected_population:
        raise HaldPrimaryError("C1 population evidence differs")
    sources = value["candidate_sources"]
    if not isinstance(sources, list) or len(sources) != 194:
        raise HaldPrimaryError("C1 manifest candidate count differs")
    expected_source_keys = {
        "candidate_id",
        "path_identity_sha256",
        "member_bytes_sha256",
        "native_table_array_sha256",
        "cube_side",
    }
    for row in sources:
        _expect_keys(row, expected_source_keys, label="C1 source row")
        for key in (
            "candidate_id",
            "path_identity_sha256",
            "member_bytes_sha256",
        ):
            _hex_digest(row[key], length=64, label=f"C1 source {key}")
        _hex_digest(
            row["native_table_array_sha256"],
            length=64,
            label="C1 native table hash",
        )
        if isinstance(row["cube_side"], bool) or row["cube_side"] not in (
            144,
            256,
        ):
            raise HaldPrimaryError("C1 source cube side differs")
    if len({row["candidate_id"] for row in sources}) != 194:
        raise HaldPrimaryError("C1 candidate IDs are not unique")
    canonical_json_bytes(value)


METRIC_KEYS = {
    "finite",
    "output_minimum",
    "output_maximum",
    "probe_new_interior_endpoint_fraction",
    "native_new_interior_endpoint_fraction",
    "new_interior_endpoint_fraction",
    "negative_jacobian_fraction",
    "minimum_jacobian_determinant",
    "maximum_jacobian_spectral_norm",
    "neutral_luma_reversal",
    "gradient_rgb_gain_p99",
    "maximum_gradient_rgb_gain",
    "maximum_gradient_channel_second_difference",
    "native_maximum_adjacent_code_step",
    "native_maximum_second_code_difference",
    "style_delta_e76_median",
    "joint_basic_residual_delta_e76_median",
}
SAFETY_KEYS = {
    "finite",
    "range",
    "new_interior_endpoint_fraction",
    "negative_jacobian_fraction",
    "minimum_jacobian_determinant",
    "maximum_jacobian_spectral_norm",
    "neutral_luma_reversal",
    "gradient_rgb_gain_p99",
    "maximum_gradient_rgb_gain",
    "maximum_gradient_channel_second_difference",
    "native_maximum_adjacent_code_step",
    "native_maximum_second_code_difference",
}
HASH_KEYS = {
    "member_bytes_sha256",
    "native_table_array_sha256",
    "style_output_array_sha256",
    "residual_signature_array_sha256",
    "jacobian_signature_array_sha256",
    "gradient_signature_array_sha256",
    "native_difference_signature_array_sha256",
}


def _validate_candidate_record(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _expect_keys(
        row,
        {
            "candidate_id",
            "source",
            "hashes",
            "metrics",
            "safety_checks",
            "normalized_safety_margin",
            "structurally_safe",
            "style_check",
            "non_basic_check",
            "eligible",
            "state",
        },
        label="C1 candidate record",
    )
    _hex_digest(row["candidate_id"], length=64, label="candidate ID")
    _expect_keys(
        row["source"],
        {
            "candidate_id",
            "path_identity_sha256",
            "archive_member_path",
            "member_bytes",
            "member_bytes_sha256",
            "crc32",
            "hald_level",
            "cube_side",
            "native_table_array_sha256",
        },
        label="C1 candidate source",
    )
    if row["source"]["candidate_id"] != row["candidate_id"]:
        raise HaldPrimaryError("C1 source candidate identity differs")
    for key in (
        "path_identity_sha256",
        "member_bytes_sha256",
        "native_table_array_sha256",
    ):
        _hex_digest(
            row["source"][key], length=64, label=f"C1 source {key}"
        )
    _hex_digest(row["source"]["crc32"], length=8, label="C1 source CRC")
    for key in ("member_bytes", "hald_level", "cube_side"):
        if (
            isinstance(row["source"][key], bool)
            or not isinstance(row["source"][key], int)
            or row["source"][key] <= 0
        ):
            raise HaldPrimaryError(f"C1 source integer differs: {key}")
    if not isinstance(row["source"]["archive_member_path"], str):
        raise HaldPrimaryError("C1 source path type differs")
    expected_identity = candidate_identity(
        row["source"]["archive_member_path"],
        row["source"]["member_bytes_sha256"],
    )
    if any(
        row["source"][key] != expected_identity[key]
        for key in ("candidate_id", "path_identity_sha256")
    ):
        raise HaldPrimaryError("C1 candidate identity preimage differs")
    _expect_keys(row["hashes"], HASH_KEYS, label="C1 candidate hashes")
    for name, digest in row["hashes"].items():
        _hex_digest(digest, length=64, label=f"C1 hash {name}")
    if (
        row["hashes"]["member_bytes_sha256"]
        != row["source"]["member_bytes_sha256"]
        or row["hashes"]["native_table_array_sha256"]
        != row["source"]["native_table_array_sha256"]
    ):
        raise HaldPrimaryError("C1 source and metric hash bindings differ")
    _expect_keys(row["metrics"], METRIC_KEYS, label="C1 metrics")
    _expect_keys(row["safety_checks"], SAFETY_KEYS, label="C1 safety")
    for key in (
        "structurally_safe",
        "style_check",
        "non_basic_check",
        "eligible",
    ):
        if not isinstance(row[key], bool):
            raise HaldPrimaryError(f"C1 candidate Boolean differs: {key}")
    for key, value in row["safety_checks"].items():
        if not isinstance(value, bool):
            raise HaldPrimaryError(f"C1 safety Boolean differs: {key}")
    for key, value in row["metrics"].items():
        if key == "finite":
            if not isinstance(value, bool):
                raise HaldPrimaryError("C1 finite metric is not Boolean")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise HaldPrimaryError(f"C1 metric differs: {key}")
    margin = row["normalized_safety_margin"]
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not 0.0 <= float(margin) <= 1.0
    ):
        raise HaldPrimaryError("C1 safety margin differs")
    state = _candidate_state(
        structurally_safe=row["structurally_safe"],
        style_check=row["style_check"],
        non_basic_check=row["non_basic_check"],
    )
    if row["state"] != state or row["eligible"] != (state == "eligible"):
        raise HaldPrimaryError("C1 candidate state contradicts checks")
    if row["structurally_safe"] != all(row["safety_checks"].values()):
        raise HaldPrimaryError("C1 structural state contradicts checks")
    expected_style = row["metrics"]["style_delta_e76_median"] >= float(
        config["automatic_gates"]["minimum_style_delta_e76_median"]
    )
    expected_non_basic = row["metrics"][
        "joint_basic_residual_delta_e76_median"
    ] >= float(
        config["automatic_gates"][
            "minimum_joint_basic_residual_delta_e76_median"
        ]
    )
    if row["style_check"] != expected_style:
        raise HaldPrimaryError("C1 style state contradicts metric")
    if row["non_basic_check"] != expected_non_basic:
        raise HaldPrimaryError("C1 non-basic state contradicts metric")


def validate_report(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    report_keys = {
        "schema_version",
        "records",
        "candidate_state_counts",
        "eligible_candidate_ids",
        "components_by_order",
        "strength_pairwise_parent_order",
        "nontransitive_strength_ambiguities",
        "components_permutation_invariant",
        "component_representatives",
        "residual_distance_rows",
        "selection_trace",
        "selected_candidate_ids",
        "selected_count",
        "frontier_claim",
        "frontier_pass",
        "photographic_contract_design_allowed",
        "photograph_rendered",
    }
    _expect_keys(value, COMMON_KEYS | report_keys, label="C1 report")
    _validate_common(value, config)
    if value["schema_version"] != REPORT_SCHEMA:
        raise HaldPrimaryError("C1 report schema differs")
    records = value["records"]
    if not isinstance(records, list) or len(records) != 194:
        raise HaldPrimaryError("C1 report candidate count differs")
    for row in records:
        _validate_candidate_record(row, config)
    candidate_ids = [row["candidate_id"] for row in records]
    if len(set(candidate_ids)) != 194:
        raise HaldPrimaryError("C1 report candidate IDs are not unique")
    eligible = [row["candidate_id"] for row in records if row["eligible"]]
    if value["eligible_candidate_ids"] != eligible:
        raise HaldPrimaryError("C1 eligible order differs")
    state_counts: dict[str, int] = {}
    for row in records:
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
    if value["candidate_state_counts"] != state_counts:
        raise HaldPrimaryError("C1 candidate counts differ")
    for key, count in value["candidate_state_counts"].items():
        if (
            key
            not in {
                "eligible",
                "structural_veto",
                "style_ineligible",
                "basic_explainable",
            }
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise HaldPrimaryError("C1 candidate state count type differs")
    expected_orders = {
        "parent_primary_order",
        "reverse_parent_primary_order",
        "sha256_candidate_identity_order",
    }
    _expect_keys(
        value["components_by_order"],
        expected_orders,
        label="C1 component orders",
    )
    parent_components = value["components_by_order"][
        "parent_primary_order"
    ]
    if any(
        components != parent_components
        for components in value["components_by_order"].values()
    ):
        raise HaldPrimaryError("C1 components differ by order")
    flattened = [
        candidate
        for component in parent_components
        for candidate in component
    ]
    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != set(eligible)
    ):
        raise HaldPrimaryError("C1 components do not partition eligibility")
    if value["components_permutation_invariant"] is not True:
        raise HaldPrimaryError("C1 component invariance is false")
    representatives = value["component_representatives"]
    if not isinstance(representatives, list) or len(representatives) != len(
        parent_components
    ):
        raise HaldPrimaryError("C1 representative count differs")
    representative_ids = []
    component_ids = set()
    for row in representatives:
        _expect_keys(
            row,
            {"component_id", "members", "representative", "ranking"},
            label="C1 representative",
        )
        _hex_digest(
            row["component_id"], length=64, label="C1 component ID"
        )
        if (
            row["component_id"] != canonical_sha256(sorted(row["members"]))
            or row["members"] not in parent_components
            or row["representative"] not in row["members"]
            or row["component_id"] in component_ids
        ):
            raise HaldPrimaryError("C1 representative binding differs")
        component_ids.add(row["component_id"])
        representative_ids.append(row["representative"])
        _expect_keys(
            row["ranking"],
            {
                "style_delta_e76_median",
                "joint_basic_residual_delta_e76_median",
                "normalized_safety_margin",
                "native_table_array_sha256",
                "path_identity_sha256",
            },
            label="C1 representative ranking",
        )
        record = next(
            item
            for item in records
            if item["candidate_id"] == row["representative"]
        )
        members_by_id = {
            item["candidate_id"]: item
            for item in records
            if item["candidate_id"] in row["members"]
        }
        expected_representative = min(
            row["members"],
            key=lambda candidate: _representative_key(
                members_by_id[candidate]
            ),
        )
        if row["representative"] != expected_representative:
            raise HaldPrimaryError("C1 representative is not rank-optimal")
        expected_ranking = {
            "style_delta_e76_median": record["metrics"][
                "style_delta_e76_median"
            ],
            "joint_basic_residual_delta_e76_median": record["metrics"][
                "joint_basic_residual_delta_e76_median"
            ],
            "normalized_safety_margin": record[
                "normalized_safety_margin"
            ],
            "native_table_array_sha256": record["hashes"][
                "native_table_array_sha256"
            ],
            "path_identity_sha256": record["source"][
                "path_identity_sha256"
            ],
        }
        if row["ranking"] != expected_ranking:
            raise HaldPrimaryError("C1 representative ranking differs")
    selected = value["selected_candidate_ids"]
    if (
        not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or not set(selected).issubset(set(representative_ids))
        or len(selected)
        > int(config["diversity_selection"]["maximum_selected"])
    ):
        raise HaldPrimaryError("C1 selected identities differ")
    trace = value["selection_trace"]
    if (
        not isinstance(trace, list)
        or [row["candidate_id"] for row in trace] != selected
    ):
        raise HaldPrimaryError("C1 selection trace differs")
    for index, row in enumerate(trace):
        _expect_keys(
            row,
            {"step", "candidate_id", "minimum_distance_to_selected"},
            label="C1 selection trace row",
        )
        if row["step"] != index:
            raise HaldPrimaryError("C1 selection trace step differs")
        distance = row["minimum_distance_to_selected"]
        if index == 0:
            if distance is not None:
                raise HaldPrimaryError("C1 seed distance must be null")
        elif (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(float(distance))
            or float(distance)
            < float(
                config["diversity_selection"][
                    "minimum_distinct_distance"
                ]
            )
        ):
            raise HaldPrimaryError("C1 selected distance differs")
    expected_claim = (
        "structurally_distinct_frontier"
        if len(selected) >= 2
        else (
            "singleton_structural_challenger_not_bank_diversity"
            if len(selected) == 1
            else "no_structural_challenger"
        )
    )
    if value["frontier_claim"] != expected_claim:
        raise HaldPrimaryError("C1 frontier claim differs")
    for row in value["strength_pairwise_parent_order"]:
        _expect_keys(
            row,
            {
                "left_exact_group_id",
                "right_exact_group_id",
                "metrics",
                "strength_equivalent",
            },
            label="C1 strength pair",
        )
        _hex_digest(
            row["left_exact_group_id"],
            length=64,
            label="C1 left exact group",
        )
        _hex_digest(
            row["right_exact_group_id"],
            length=64,
            label="C1 right exact group",
        )
        _expect_keys(
            row["metrics"],
            {
                "effect_cosine",
                "minimum_explained_energy_fraction",
                "bilateral_normalized_residual",
            },
            label="C1 strength metrics",
        )
        if not isinstance(row["strength_equivalent"], bool):
            raise HaldPrimaryError("C1 strength Boolean differs")
        if any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            for metric in row["metrics"].values()
        ):
            raise HaldPrimaryError("C1 strength metric differs")
    for row in value["nontransitive_strength_ambiguities"]:
        _expect_keys(
            row,
            {"exact_group_ids", "missing_edges", "decision"},
            label="C1 strength ambiguity",
        )
        if row["decision"] != "retain_exact_groups":
            raise HaldPrimaryError("C1 ambiguity decision differs")
    for row in value["residual_distance_rows"]:
        _expect_keys(
            row,
            {"left", "right", "residual_delta_e76_median"},
            label="C1 residual distance",
        )
        if (
            row["left"] not in representative_ids
            or row["right"] not in representative_ids
            or row["left"] >= row["right"]
            or isinstance(row["residual_delta_e76_median"], bool)
            or not isinstance(
                row["residual_delta_e76_median"], (int, float)
            )
            or not math.isfinite(
                float(row["residual_delta_e76_median"])
            )
        ):
            raise HaldPrimaryError("C1 residual distance differs")
    for key in (
        "components_permutation_invariant",
        "frontier_pass",
        "photographic_contract_design_allowed",
        "photograph_rendered",
    ):
        if not isinstance(value[key], bool):
            raise HaldPrimaryError(f"C1 report Boolean differs: {key}")
    if value["photograph_rendered"] is not False:
        raise HaldPrimaryError("C1 report claims photograph render")
    if (
        isinstance(value["selected_count"], bool)
        or not isinstance(value["selected_count"], int)
        or value["selected_count"] != len(value["selected_candidate_ids"])
    ):
        raise HaldPrimaryError("C1 selected count differs")
    passed = value["selected_count"] >= int(
        config["frontier_gate"]["minimum_selected_representatives"]
    )
    if value["frontier_pass"] != passed:
        raise HaldPrimaryError("C1 frontier gate contradicts selection")
    canonical_json_bytes(value)


def _load_child(
    directory: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    manifest_bytes = (directory / "manifest.json").read_bytes()
    report_bytes = (directory / "report.json").read_bytes()
    manifest = strict_json_loads(manifest_bytes)
    report = strict_json_loads(report_bytes)
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise HaldPrimaryError("C1 child evidence roots must be objects")
    validate_manifest(manifest, config)
    validate_report(report, config)
    projected_sources = [
        {
            "candidate_id": row["candidate_id"],
            "path_identity_sha256": row["source"][
                "path_identity_sha256"
            ],
            "member_bytes_sha256": row["source"]["member_bytes_sha256"],
            "native_table_array_sha256": row["source"][
                "native_table_array_sha256"
            ],
            "cube_side": row["source"]["cube_side"],
        }
        for row in report["records"]
    ]
    if manifest["candidate_sources"] != projected_sources:
        raise HaldPrimaryError("C1 manifest/report source binding differs")
    return manifest, report, manifest_bytes, report_bytes


def reconstruct_child_evidence(
    *,
    root: Path,
    config: Mapping[str, Any],
    common: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    parent = bind_parents(root, config)
    sources = source_contract_evidence(root, config)
    evaluated, _private = evaluate_archive(
        root=root, config=config, parent=parent
    )
    expected_common = dict(common)
    if expected_common["source_contracts"] != sources:
        raise HaldPrimaryError("C1 reconstructed sources differ")
    expected_parent = {
        key: value
        for key, value in parent.items()
        if key not in {"primary_paths", "primary_records"}
    }
    if expected_common["parent_evidence"] != expected_parent:
        raise HaldPrimaryError("C1 reconstructed parent differs")
    if expected_common["access_ledger"] != evaluated["access_ledger"]:
        raise HaldPrimaryError("C1 reconstructed access differs")
    manifest = canonical_json_bytes(
        {
            "schema_version": MANIFEST_SCHEMA,
            **expected_common,
            **evaluated["manifest"],
        }
    )
    report = canonical_json_bytes(
        {
            "schema_version": REPORT_SCHEMA,
            **expected_common,
            **evaluated["report"],
        }
    )
    return manifest, report


def finalize_repeat(
    *,
    root: Path,
    first_dir: Path,
    second_dir: Path,
    config: Mapping[str, Any],
    config_raw_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    first_manifest, first_report, first_manifest_bytes, first_report_bytes = (
        _load_child(first_dir, config)
    )
    (
        second_manifest,
        second_report,
        second_manifest_bytes,
        second_report_bytes,
    ) = _load_child(second_dir, config)
    manifest_bytes_exact = first_manifest_bytes == second_manifest_bytes
    report_bytes_exact = first_report_bytes == second_report_bytes
    common = {key: first_manifest[key] for key in COMMON_KEYS}
    reconstructed_manifest, reconstructed_report = (
        reconstruct_child_evidence(root=root, config=config, common=common)
    )
    strict_reconstruction = (
        first_manifest_bytes == reconstructed_manifest
        and second_manifest_bytes == reconstructed_manifest
        and first_report_bytes == reconstructed_report
        and second_report_bytes == reconstructed_report
    )
    identities_exact = all(
        first_manifest[key] == second_manifest[key]
        and first_report[key] == second_report[key]
        for key in COMMON_KEYS
    )
    selected_exact = (
        first_report["selected_candidate_ids"]
        == second_report["selected_candidate_ids"]
    )
    child_frontier_pass = bool(
        first_report["frontier_pass"] and second_report["frontier_pass"]
    )
    evidence_conformance = bool(
        manifest_bytes_exact
        and report_bytes_exact
        and strict_reconstruction
        and identities_exact
        and selected_exact
        and first_manifest["software_commit"] == software_commit
        and first_manifest["config"]["raw_sha256_before"]
        == config_raw_sha256
    )
    automatic_pass = evidence_conformance and child_frontier_pass
    selected = first_report["selected_candidate_ids"]
    frontier_claim = first_report["frontier_claim"]
    decision = {
        "schema_version": REPEAT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "node": config["node"],
        "decision": (
            "primary_structural_frontier_pass"
            if automatic_pass
            else (
                "close_no_structural_survivor"
                if evidence_conformance and not child_frontier_pass
                else "close_repeat_or_evidence_failure"
            )
        ),
        "software_commit": software_commit,
        "config": {
            "raw_sha256": config_raw_sha256,
            "canonical_sha256": canonical_sha256(config),
        },
        "child_evidence": {
            "manifest_sha256": hashlib.sha256(
                first_manifest_bytes
            ).hexdigest(),
            "report_sha256": hashlib.sha256(
                first_report_bytes
            ).hexdigest(),
            "manifest_bytes_exact": manifest_bytes_exact,
            "report_bytes_exact": report_bytes_exact,
            "identities_exact": identities_exact,
            "selected_ids_exact": selected_exact,
            "strict_reconstruction_exact": strict_reconstruction,
            "evidence_conformance_pass": evidence_conformance,
            "child_frontier_pass": child_frontier_pass,
        },
        "candidate_state_counts": first_report["candidate_state_counts"],
        "eligible_count": len(first_report["eligible_candidate_ids"]),
        "component_count": len(
            first_report["components_by_order"]["parent_primary_order"]
        ),
        "nontransitive_strength_ambiguity_count": len(
            first_report["nontransitive_strength_ambiguities"]
        ),
        "selected_candidate_ids": selected,
        "selected_count": len(selected),
        "frontier_claim": frontier_claim,
        "access_ledger": first_manifest["access_ledger"],
        "automatic_pass": automatic_pass,
        "photographic_contract_design_allowed": automatic_pass,
        "photograph_rendered": False,
        "visual_review_allowed": False,
        "evidence_promoted": automatic_pass,
        "claim_ceiling": config["claim_ceiling"],
    }
    validate_repeat(decision, config)
    return decision


def validate_repeat(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _expect_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "node",
            "decision",
            "software_commit",
            "config",
            "child_evidence",
            "candidate_state_counts",
            "eligible_count",
            "component_count",
            "nontransitive_strength_ambiguity_count",
            "selected_candidate_ids",
            "selected_count",
            "frontier_claim",
            "access_ledger",
            "automatic_pass",
            "photographic_contract_design_allowed",
            "photograph_rendered",
            "visual_review_allowed",
            "evidence_promoted",
            "claim_ceiling",
        },
        label="C1 repeat decision",
    )
    if value["schema_version"] != REPEAT_SCHEMA:
        raise HaldPrimaryError("C1 repeat schema differs")
    if (
        value["experiment_id"] != config["experiment_id"]
        or value["node"] != config["node"]
    ):
        raise HaldPrimaryError("C1 repeat identity differs")
    _hex_digest(value["software_commit"], length=40, label="repeat commit")
    _expect_keys(
        value["config"],
        {"raw_sha256", "canonical_sha256"},
        label="C1 repeat config",
    )
    if (
        value["config"]["raw_sha256"] != EXPECTED_CONFIG_RAW_SHA256
        or value["config"]["canonical_sha256"] != canonical_sha256(config)
    ):
        raise HaldPrimaryError("C1 repeat config differs")
    child_keys = {
        "manifest_sha256",
        "report_sha256",
        "manifest_bytes_exact",
        "report_bytes_exact",
        "identities_exact",
        "selected_ids_exact",
        "strict_reconstruction_exact",
        "evidence_conformance_pass",
        "child_frontier_pass",
    }
    _expect_keys(value["child_evidence"], child_keys, label="C1 child")
    for key in ("manifest_sha256", "report_sha256"):
        _hex_digest(
            value["child_evidence"][key],
            length=64,
            label=f"C1 child {key}",
        )
    for key in child_keys - {"manifest_sha256", "report_sha256"}:
        if not isinstance(value["child_evidence"][key], bool):
            raise HaldPrimaryError(f"C1 child Boolean differs: {key}")
    for key in (
        "automatic_pass",
        "photographic_contract_design_allowed",
        "photograph_rendered",
        "visual_review_allowed",
        "evidence_promoted",
    ):
        if not isinstance(value[key], bool):
            raise HaldPrimaryError(f"C1 repeat Boolean differs: {key}")
    if value["photograph_rendered"] or value["visual_review_allowed"]:
        raise HaldPrimaryError("C1 repeat exceeds photographic boundary")
    expected_pass = bool(
        value["child_evidence"]["evidence_conformance_pass"]
        and value["child_evidence"]["child_frontier_pass"]
    )
    if any(
        value[key] != expected_pass
        for key in (
            "automatic_pass",
            "photographic_contract_design_allowed",
            "evidence_promoted",
        )
    ):
        raise HaldPrimaryError("C1 repeat promotion contradicts evidence")
    expected_decision = (
        "primary_structural_frontier_pass"
        if expected_pass
        else (
            "close_no_structural_survivor"
            if value["child_evidence"]["evidence_conformance_pass"]
            else "close_repeat_or_evidence_failure"
        )
    )
    if value["decision"] != expected_decision:
        raise HaldPrimaryError("C1 repeat decision label differs")
    if value["selected_count"] != len(value["selected_candidate_ids"]):
        raise HaldPrimaryError("C1 repeat selected count differs")
    if value["access_ledger"] != config["evidence"][
        "required_access_ledger"
    ]:
        raise HaldPrimaryError("C1 repeat access differs")
    if value["claim_ceiling"] != config["claim_ceiling"]:
        raise HaldPrimaryError("C1 repeat claim ceiling differs")
    canonical_json_bytes(value)
