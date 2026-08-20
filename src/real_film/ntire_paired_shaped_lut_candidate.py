"""SF3.A0Y fixed paired-capture shaped-LUT candidate evaluation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
    fit_operator_rows,
    gradient_p999_ratio,
    matrix_diagnostics,
    mean_oklab_error,
    new_exact_boundary_fraction,
)
from src.real_film.ntire_night_geometry_preflight import (
    _members,
    structural_view,
    validate_metadata,
)
from src.real_film.ntire_night_metadata_preflight import extract_member
from src.real_film.ppisp_capture_pair_source_lock import (
    ZipMember,
    canonical_sha256,
    http_range_get,
    sha256_bytes,
)
from src.roll2film.global_lut_distillation import (
    MonotoneRGBShaper,
    ShapedGlobalLUT,
    fit_monotone_shaper,
    fit_shaped_global_lut,
)

SCHEMA = "neuro-film.sf3-a0y-ntire-paired-shaped-lut-candidate2-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0y-ntire-paired-shaped-lut-candidate2-report.v1"
RangeReader = Callable[[str, int, int, int], bytes]


class NTIREPairedCandidateError(RuntimeError):
    """Raised when the frozen paired-capture experiment drifts."""


def curl_range_get(url: str, start: int, end: int, archive_size: int) -> bytes:
    """Read one exact HTTP range through curl without persisting member bytes."""
    if start < 0 or end < start or end >= archive_size:
        raise NTIREPairedCandidateError("invalid bounded archive range")
    expected = end - start + 1
    failures = []
    for attempt in range(4):
        try:
            result = subprocess.run(
                [
                    "curl.exe",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--range",
                    f"{start}-{end}",
                    "--write-out",
                    "%{stderr}%{http_code}",
                    url,
                ],
                capture_output=True,
                check=False,
                timeout=300,
            )
            status = result.stderr.decode("utf-8", errors="replace").strip()
            if (
                result.returncode == 0
                and status == "206"
                and len(result.stdout) == expected
            ):
                return result.stdout
            failures.append(
                f"attempt={attempt + 1}:returncode={result.returncode}:"
                f"status={status!r}:bytes={len(result.stdout)}"
            )
        except subprocess.TimeoutExpired:
            failures.append(f"attempt={attempt + 1}:timeout")
        if attempt < 3:
            time.sleep(1.0)
    raise NTIREPairedCandidateError("curl exact range failed; " + "; ".join(failures))


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise NTIREPairedCandidateError("paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    roles = payload.get("roles", {})
    candidate = payload.get("candidate", {})
    control = payload.get("controls", {}).get("bounded_logit_affine", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id")
        != "SF3.A0Y_NTIRE_PAIRED_SHAPED_LUT_CANDIDATE2_V1"
        or [
            len(roles.get(key, []))
            for key in ("fit", "calibration", "sealed_confirmation")
        ]
        != [32, 12, 12]
        or len(
            set(
                roles.get("fit", [])
                + roles.get("calibration", [])
                + roles.get("sealed_confirmation", [])
            )
        )
        != 56
        or roles.get("replacement_allowed") is not False
        or candidate.get("id") != "zhuise.shaped-global-lut.v1"
        or candidate.get("interpolation") != "tetrahedral"
        or control.get("ridge_alpha") != 0.01
        or control.get("identity_dose_binary_search_iterations") != 30
        or payload.get("bounded_final_candidate_counter_before") != 1
        or payload.get("bounded_final_candidate_counter_after_calibration_score") != 2
    ):
        raise NTIREPairedCandidateError("frozen candidate contract drift")
    for key in ("source_lock", "geometry_base", "orientation_overlay"):
        _relative(payload["source_contracts"][key])
    _relative(payload["parent_geometry_evidence"]["path"])
    _relative(payload["cache"]["root"])
    return payload


def _load_geometry(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent = root / _relative(contract["parent_geometry_evidence"]["path"])
    if _hash_file(parent) != contract["parent_geometry_evidence"]["sha256"]:
        raise NTIREPairedCandidateError("parent geometry evidence drift")
    parent_payload = json.loads(parent.read_text(encoding="utf-8"))
    if (
        parent_payload.get("formal_replay", {}).get("stable_identity")
        != contract["parent_geometry_evidence"]["stable_identity"]
    ):
        raise NTIREPairedCandidateError("parent geometry identity drift")
    base = root / _relative(contract["source_contracts"]["geometry_base"])
    overlay = root / _relative(contract["source_contracts"]["orientation_overlay"])
    base_payload = json.loads(base.read_text(encoding="utf-8"))
    overlay_payload = json.loads(overlay.read_text(encoding="utf-8"))
    if (
        hashlib.sha256(base.read_bytes()).hexdigest()
        != overlay_payload["base_contract"]["sha256"]
    ):
        raise NTIREPairedCandidateError("geometry contract identity drift")
    base_payload["metadata_contract"]["orientation_allowed"] = [
        *overlay_payload["allowed_numeric_orientations"],
        *overlay_payload["official_orientation_mapping"].keys(),
    ]
    return base_payload


def _member_map(geometry: dict[str, Any], reader: RangeReader) -> dict[str, ZipMember]:
    members, _, _ = _members(geometry, reader)
    return members


def _cache_paths(
    cache_root: Path, role: str, numeric_id: int
) -> tuple[Path, Path, Path]:
    folder = cache_root / role
    return (
        folder / f"{numeric_id}_source.npy",
        folder / f"{numeric_id}_target.npy",
        folder / f"{numeric_id}_facts.json",
    )


def _load_or_fetch(
    *,
    numeric_id: int,
    role: str,
    need_target: bool,
    geometry: dict[str, Any],
    members: Mapping[str, ZipMember],
    cache_root: Path,
    reader: RangeReader,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    source_path, target_path, facts_path = _cache_paths(cache_root, role, numeric_id)
    if (
        source_path.is_file()
        and facts_path.is_file()
        and (not need_target or target_path.is_file())
    ):
        source = np.load(source_path, allow_pickle=False)
        target = np.load(target_path, allow_pickle=False) if need_target else None
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        if _hash_file(source_path) != facts["source_npy_sha256"]:
            raise NTIREPairedCandidateError("cached source drift")
        if need_target and _hash_file(target_path) != facts["target_npy_sha256"]:
            raise NTIREPairedCandidateError("cached target drift")
        return (
            source.astype(np.float64) / 65535.0,
            None if target is None else target.astype(np.float64) / 255.0,
            facts,
        )

    raw_archive = geometry["archives"]["raw"]
    target_archive = geometry["archives"]["target"]
    names = {
        "metadata": f"raw:raw/{numeric_id}.json",
        "raw": f"raw:raw/{numeric_id}.png",
        "target": f"target:sony/{numeric_id}.JPG",
    }
    if any(name not in members for name in names.values()):
        raise NTIREPairedCandidateError("candidate member missing")
    metadata_bytes, _ = extract_member(
        raw_archive["url"], members[names["metadata"]], raw_archive["bytes"], reader
    )
    raw_bytes, _ = extract_member(
        raw_archive["url"], members[names["raw"]], raw_archive["bytes"], reader
    )
    metadata = json.loads(metadata_bytes)
    validate_metadata(metadata, geometry)
    raw = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise NTIREPairedCandidateError("candidate RAW decode failed")
    source = structural_view(raw, metadata, geometry["official_baseline"]["geometry"])
    size = (
        int(geometry.get("candidate_cache_width", 256)),
        int(geometry.get("candidate_cache_height", 256)),
    )
    source = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    source_u16 = np.rint(np.clip(source, 0.0, 1.0) * 65535.0).astype("<u2")
    target_u8: np.ndarray | None = None
    target_member_sha = None
    if need_target:
        target_bytes, _ = extract_member(
            target_archive["url"],
            members[names["target"]],
            target_archive["bytes"],
            reader,
        )
        target_bgr = cv2.imdecode(
            np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if target_bgr is None or target_bgr.shape != (2000, 2000, 3):
            raise NTIREPairedCandidateError("candidate target decode/shape failed")
        target_u8 = cv2.resize(
            cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB),
            size,
            interpolation=cv2.INTER_AREA,
        )
        target_member_sha = sha256_bytes(target_bytes)

    source_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(source_path, source_u16, allow_pickle=False)
    if target_u8 is not None:
        np.save(target_path, target_u8, allow_pickle=False)
    facts = {
        "numeric_id": numeric_id,
        "role": role,
        "metadata_sha256": sha256_bytes(metadata_bytes),
        "raw_member_sha256": sha256_bytes(raw_bytes),
        "target_member_sha256": target_member_sha,
        "source_npy_sha256": _hash_file(source_path),
        "target_npy_sha256": _hash_file(target_path) if target_u8 is not None else None,
        "source_shape": list(source_u16.shape),
        "target_shape": list(target_u8.shape) if target_u8 is not None else None,
    }
    facts_path.write_bytes(_canonical_bytes(facts))
    return (
        source_u16.astype(np.float64) / 65535.0,
        None if target_u8 is None else target_u8.astype(np.float64) / 255.0,
        facts,
    )


def fit_pixel_indexes(scene_id: int, pixel_count: int, count: int) -> np.ndarray:
    """Return the frozen scene-specific deterministic fit sample."""
    if count > pixel_count:
        raise NTIREPairedCandidateError("fit sample count exceeds pixels")
    prefix = f"sf3-a0y-fit-pixel-v1:{scene_id}:".encode()
    return np.asarray(
        heapq.nsmallest(
            count,
            range(pixel_count),
            key=lambda index: hashlib.sha256(prefix + str(index).encode()).digest(),
        ),
        dtype=np.int64,
    )


def _sample_indices(scene_id: int, count: int, total: int) -> np.ndarray:
    return fit_pixel_indexes(scene_id, total, count)


def bounded_affine(
    matrix: np.ndarray, bias: np.ndarray, spec: Mapping[str, Any]
) -> LogitAffineOperator:
    """Dose a fitted affine toward identity until all frozen matrix gates hold."""
    matrix = np.asarray(matrix, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or bias.shape != (3,)
        or not (np.isfinite(matrix).all() and np.isfinite(bias).all())
    ):
        raise NTIREPairedCandidateError("affine parameters are invalid")
    low, high = 0.0, 1.0
    for _ in range(int(spec["identity_dose_binary_search_iterations"])):
        dose = (low + high) * 0.5
        operator = LogitAffineOperator(
            np.eye(3) + dose * (matrix - np.eye(3)), dose * bias, dose
        )
        facts = matrix_diagnostics(operator)
        valid = (
            facts["determinant"] >= float(spec["minimum_matrix_determinant"])
            and facts["minimum_singular_value"]
            >= float(spec["minimum_matrix_singular_value"])
            and facts["condition_number"]
            <= float(spec["maximum_matrix_condition_number"])
        )
        if valid:
            low = dose
        else:
            high = dose
    return LogitAffineOperator(np.eye(3) + low * (matrix - np.eye(3)), low * bias, low)


def _fit_affine(
    source: np.ndarray, target: np.ndarray, spec: Mapping[str, Any]
) -> LogitAffineOperator:
    matrix, bias = fit_operator_rows(
        source, target, ridge_alpha=float(spec["ridge_alpha"]), diagonal=False
    )
    return bounded_affine(matrix, bias, spec)


def _model_payload(
    model: ShapedGlobalLUT | MonotoneRGBShaper | LogitAffineOperator,
) -> dict[str, Any]:
    if isinstance(model, LogitAffineOperator):
        return {
            "schema": "neuro-film.logit-affine-control.v1",
            "matrix": model.matrix.tolist(),
            "bias": model.bias.tolist(),
            "dose": model.dose,
        }
    return model.to_dict()


def _score_role(
    *,
    role: str,
    ids: list[int],
    models: Mapping[str, Any],
    geometry: dict[str, Any],
    members: Mapping[str, ZipMember],
    cache_root: Path,
    reader: RangeReader,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for numeric_id in ids:
        source, target, facts = _load_or_fetch(
            numeric_id=numeric_id,
            role=role,
            need_target=True,
            geometry=geometry,
            members=members,
            cache_root=cache_root,
            reader=reader,
        )
        assert target is not None
        outputs = {
            "candidate": models["candidate"].apply(source),
            "monotone": models["monotone"].apply(source),
            "affine": apply_operator(source, models["affine"]),
            "identity": source.astype(np.float32),
            "cyclic_wrong": models["cyclic_wrong"].apply(source),
        }
        errors = {
            key: mean_oklab_error(value, target) for key, value in outputs.items()
        }
        strongest_name = min(
            ("monotone", "affine", "identity"), key=lambda key: errors[key]
        )
        strongest = errors[strongest_name]
        candidate = errors["candidate"]
        wrong = errors["cyclic_wrong"]
        rows.append(
            {
                "numeric_id": numeric_id,
                "cache": facts,
                "mean_oklab_error": errors,
                "strongest_legitimate_control": strongest_name,
                "relative_reduction_vs_strongest_legitimate": (strongest - candidate)
                / max(strongest, 1e-12),
                "relative_reduction_vs_cyclic_wrong_target": (wrong - candidate)
                / max(wrong, 1e-12),
                "gradient_p999_ratio": gradient_p999_ratio(
                    source, outputs["candidate"]
                ),
                "new_exact_boundary_fraction": new_exact_boundary_fraction(
                    source, outputs["candidate"]
                ),
                "candidate_output_sha256": sha256_bytes(
                    np.asarray(outputs["candidate"], dtype="<f4").tobytes()
                ),
            }
        )
    return rows


def _aggregate(
    rows: list[dict[str, Any]], gates: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    reductions = np.asarray(
        [row["relative_reduction_vs_strongest_legitimate"] for row in rows]
    )
    wrong = np.asarray(
        [row["relative_reduction_vs_cyclic_wrong_target"] for row in rows]
    )
    errors = np.asarray([row["mean_oklab_error"]["candidate"] for row in rows])
    aggregate = {
        "row_count": len(rows),
        "improvement_rate_vs_strongest_legitimate": float(np.mean(reductions > 0.0)),
        "median_relative_reduction_vs_strongest_legitimate": float(
            np.median(reductions)
        ),
        "worst_relative_reduction_vs_strongest_legitimate": float(np.min(reductions)),
        "median_mean_oklab_error": float(np.median(errors)),
        "p95_mean_oklab_error": float(np.quantile(errors, 0.95)),
        "improvement_rate_vs_cyclic_wrong_target": float(np.mean(wrong > 0.0)),
        "median_relative_reduction_vs_cyclic_wrong_target": float(np.median(wrong)),
        "maximum_gradient_p999_ratio": max(row["gradient_p999_ratio"] for row in rows),
        "maximum_new_exact_boundary_fraction": max(
            row["new_exact_boundary_fraction"] for row in rows
        ),
    }
    checks = {
        "improvement_rate_vs_strongest_legitimate": aggregate[
            "improvement_rate_vs_strongest_legitimate"
        ]
        >= gates["minimum_improvement_rate_vs_strongest_legitimate"],
        "median_reduction_vs_strongest_legitimate": aggregate[
            "median_relative_reduction_vs_strongest_legitimate"
        ]
        >= gates["minimum_median_relative_reduction_vs_strongest_legitimate"],
        "worst_tail_vs_strongest_legitimate": aggregate[
            "worst_relative_reduction_vs_strongest_legitimate"
        ]
        >= gates["minimum_worst_relative_reduction_vs_strongest_legitimate"],
        "median_error": aggregate["median_mean_oklab_error"]
        <= gates["maximum_median_mean_oklab_error"],
        "p95_error": aggregate["p95_mean_oklab_error"]
        <= gates["maximum_p95_mean_oklab_error"],
        "improvement_rate_vs_cyclic_wrong_target": aggregate[
            "improvement_rate_vs_cyclic_wrong_target"
        ]
        >= gates.get("minimum_improvement_rate_vs_cyclic_wrong_target", 0.0),
        "median_reduction_vs_cyclic_wrong_target": aggregate[
            "median_relative_reduction_vs_cyclic_wrong_target"
        ]
        >= gates.get(
            "minimum_median_relative_reduction_vs_cyclic_wrong_target", -math.inf
        ),
        "gradient": aggregate["maximum_gradient_p999_ratio"]
        <= gates["maximum_gradient_p999_ratio"],
        "boundary": aggregate["maximum_new_exact_boundary_fraction"]
        <= gates["maximum_new_exact_boundary_fraction"],
    }
    return aggregate, checks


def evaluate(
    contract: Mapping[str, Any],
    root: Path,
    *,
    model_lock_path: Path,
    reverse_row_order: bool = False,
    cache_root_override: Path | None = None,
    range_reader: RangeReader = http_range_get,
) -> dict[str, Any]:
    geometry = _load_geometry(contract, root)
    geometry["candidate_cache_width"] = int(contract["cache"]["width"])
    geometry["candidate_cache_height"] = int(contract["cache"]["height"])
    logical_cache_root = root / _relative(contract["cache"]["root"])
    if cache_root_override is None:
        cache_root = logical_cache_root
        cache_storage_mode = "canonical_repo_relative"
    else:
        cache_root = cache_root_override.resolve()
        allowed = logical_cache_root.resolve()
        if cache_root != allowed and allowed not in cache_root.parents:
            raise NTIREPairedCandidateError(
                "cache override is outside the configured repo-relative fallback"
            )
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_storage_mode = "project_owned_d_fallback"
    members = _member_map(geometry, range_reader)
    fit_by_id: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    fit_read_order = list(contract["roles"]["fit"])
    if reverse_row_order:
        fit_read_order.reverse()
    for numeric_id in fit_read_order:
        source, target, facts = _load_or_fetch(
            numeric_id=numeric_id,
            role="fit",
            need_target=True,
            geometry=geometry,
            members=members,
            cache_root=cache_root,
            reader=range_reader,
        )
        assert target is not None
        indexes = _sample_indices(
            numeric_id,
            int(contract["fit_sampling"]["pixels_per_fit_scene"]),
            source.shape[0] * source.shape[1],
        )
        fit_by_id[numeric_id] = (
            source.reshape(-1, 3)[indexes],
            target.reshape(-1, 3)[indexes],
            facts,
        )
    fit_sources = [fit_by_id[item][0] for item in contract["roles"]["fit"]]
    fit_targets = [fit_by_id[item][1] for item in contract["roles"]["fit"]]
    fit_facts = sorted(
        (fit_by_id[item][2] for item in contract["roles"]["fit"]),
        key=lambda value: value["numeric_id"],
    )
    source_rows = np.concatenate(fit_sources)
    target_rows = np.concatenate(fit_targets)
    candidate_spec = contract["candidate"]
    candidate = fit_shaped_global_lut(
        source_rows,
        target_rows,
        shaper_knots=int(candidate_spec["shaper_knots"]),
        minimum_shaper_increment=float(candidate_spec["minimum_shaper_increment"]),
        lut_size=int(candidate_spec["lut_size"]),
        identity_regularization=float(candidate_spec["identity_regularization"]),
        difference_regularization=float(candidate_spec["difference_regularization"]),
        lsqr_iteration_limit=int(candidate_spec["lsqr_iteration_limit"]),
        output_margin=float(candidate_spec["output_margin"]),
        minimum_determinant=float(candidate_spec["minimum_determinant"]),
        strength_iterations=int(candidate_spec["strength_iterations"]),
    )
    monotone = fit_monotone_shaper(
        source_rows,
        target_rows,
        knots=int(candidate_spec["shaper_knots"]),
        minimum_increment=float(candidate_spec["minimum_shaper_increment"]),
    )
    affine = _fit_affine(
        source_rows, target_rows, contract["controls"]["bounded_logit_affine"]
    )
    cyclic_target_rows = np.concatenate(fit_targets[1:] + fit_targets[:1])
    cyclic_wrong = fit_shaped_global_lut(
        source_rows,
        cyclic_target_rows,
        shaper_knots=int(candidate_spec["shaper_knots"]),
        minimum_shaper_increment=float(candidate_spec["minimum_shaper_increment"]),
        lut_size=int(candidate_spec["lut_size"]),
        identity_regularization=float(candidate_spec["identity_regularization"]),
        difference_regularization=float(candidate_spec["difference_regularization"]),
        lsqr_iteration_limit=int(candidate_spec["lsqr_iteration_limit"]),
        output_margin=float(candidate_spec["output_margin"]),
        minimum_determinant=float(candidate_spec["minimum_determinant"]),
        strength_iterations=int(candidate_spec["strength_iterations"]),
    )
    models = {
        "candidate": candidate,
        "monotone": monotone,
        "affine": affine,
        "cyclic_wrong": cyclic_wrong,
    }
    model_payloads = {key: _model_payload(value) for key, value in models.items()}
    model_hashes = {
        key: hashlib.sha256(_canonical_bytes(value)).hexdigest()
        for key, value in model_payloads.items()
    }

    model_lock = {
        "schema": "neuro-film.sf3-a0y-paired-shaped-lut-model-lock.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hashlib.sha256(_canonical_bytes(contract)).hexdigest(),
        "models": model_payloads,
        "model_sha256": model_hashes,
    }
    model_lock_bytes = _canonical_bytes(model_lock)
    model_lock_path.parent.mkdir(parents=True, exist_ok=True)
    model_lock_path.write_bytes(model_lock_bytes)
    model_lock_sha256 = hashlib.sha256(model_lock_bytes).hexdigest()

    calibration_ids = list(contract["roles"]["calibration"])
    if reverse_row_order:
        calibration_ids.reverse()
    calibration_rows = _score_role(
        role="calibration",
        ids=calibration_ids,
        models=models,
        geometry=geometry,
        members=members,
        cache_root=cache_root,
        reader=range_reader,
    )
    calibration_order = {
        item: index for index, item in enumerate(contract["roles"]["calibration"])
    }
    calibration_rows.sort(key=lambda row: calibration_order[row["numeric_id"]])
    calibration_aggregate, calibration_gates = _aggregate(
        calibration_rows, contract["gates"]["calibration"]
    )
    calibration_gates["candidate_jacobian"] = float(
        np.min(candidate.lut.tetrahedron_jacobian_determinants())
    ) >= float(contract["gates"]["calibration"]["minimum_lut_jacobian_determinant"])
    calibration_gates["model_replay"] = all(
        np.array_equal(
            models[key].apply(source_rows[:1024]),
            ShapedGlobalLUT.from_dict(model_payloads[key]).apply(source_rows[:1024]),
        )
        for key in ("candidate", "cyclic_wrong")
    )
    minimum_lut_jacobian_determinant = float(
        np.min(candidate.lut.tetrahedron_jacobian_determinants())
    )
    calibration_pass = all(calibration_gates.values())
    sealed_rows: list[dict[str, Any]] = []
    sealed_aggregate = None
    sealed_gates = None
    if calibration_pass:
        sealed_ids = list(contract["roles"]["sealed_confirmation"])
        if reverse_row_order:
            sealed_ids.reverse()
        sealed_rows = _score_role(
            role="sealed_confirmation",
            ids=sealed_ids,
            models=models,
            geometry=geometry,
            members=members,
            cache_root=cache_root,
            reader=range_reader,
        )
        sealed_order = {
            item: index
            for index, item in enumerate(contract["roles"]["sealed_confirmation"])
        }
        sealed_rows.sort(key=lambda row: sealed_order[row["numeric_id"]])
        sealed_aggregate, sealed_gates = _aggregate(
            sealed_rows, contract["gates"]["sealed_confirmation"]
        )
    automatic_pass = bool(
        calibration_pass and sealed_gates is not None and all(sealed_gates.values())
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hashlib.sha256(_canonical_bytes(contract)).hexdigest(),
        "model_sha256": model_hashes,
        "model_lock_sha256": model_lock_sha256,
        "fit_cache_facts_sha256": canonical_sha256(fit_facts),
        "mechanics": {
            "model_lock_persisted_before_calibration_target_read": True,
            "minimum_lut_jacobian_determinant": minimum_lut_jacobian_determinant,
            "model_replay_exact": calibration_gates["model_replay"],
            "full_member_payload_persisted": False,
            "cache_storage_mode": cache_storage_mode,
            "logical_cache_root": contract["cache"]["root"],
            "range_reader": getattr(range_reader, "__name__", type(range_reader).__name__),
        },
        "calibration": {
            "rows": calibration_rows,
            "aggregate": calibration_aggregate,
            "gates": calibration_gates,
            "automatic_pass": calibration_pass,
        },
        "sealed_confirmation": {
            "target_read": calibration_pass,
            "rows": sealed_rows,
            "aggregate": sealed_aggregate,
            "gates": sealed_gates,
        },
        "bounded_final_candidate_counter_before": 1,
        "bounded_final_candidate_counter_after": 2,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable = dict(report)
    report["stable_identity"] = hashlib.sha256(_canonical_bytes(stable)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    data = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
