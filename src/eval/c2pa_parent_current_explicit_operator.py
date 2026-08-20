"""Evaluate a bounded colour operator from embedded C2PA parent/current pixels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps
from scipy.optimize import least_squares

from src.preprocess.raster_decode import (
    load_raster_working_image,
    working_image_to_srgb_float,
)
from src.roll2film.triangular_logit_transport import (
    TriangularLogitTransportError,
    fit_triangular_logit_transport,
    select_safe_transport,
)


class C2PAPairedOperatorError(ValueError):
    """Raised when the frozen paired-observation contract cannot be evaluated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise C2PAPairedOperatorError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolve(root: Path, relative: str) -> Path:
    return (root / relative).resolve(strict=True)


def _load_p210_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("p210_c2pa_source_audit", path)
    if spec is None or spec.loader is None:
        raise C2PAPairedOperatorError("cannot load P210 source audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_contract(contract_path: Path, root: Path) -> tuple[dict[str, Any], Any]:
    contract = _read_json(contract_path)
    if contract.get("schema") != (
        "neuro-film.u5-r2c2pa0-parent-current-explicit-operator-contract.v1"
    ):
        raise C2PAPairedOperatorError("unexpected contract schema")
    if contract.get("status") != (
        "FROZEN_BEFORE_PARENT_OR_CURRENT_PIXEL_DECODE_FIT_OR_SCORE"
    ):
        raise C2PAPairedOperatorError("contract is not frozen")
    parent = contract["parent_source_audit"]
    runtime = contract["runtime_source"]
    evidence_path = _resolve(root, parent["evidence"])
    script_path = _resolve(root, parent["audit_script"])
    wheel_path = _resolve(root, runtime["c2pa_wheel"])
    if sha256_file(evidence_path) != parent["evidence_sha256"]:
        raise C2PAPairedOperatorError("P210 evidence hash mismatch")
    if sha256_file(script_path) != parent["audit_script_sha256"]:
        raise C2PAPairedOperatorError("P210 audit-script hash mismatch")
    if wheel_path.stat().st_size != int(runtime["wheel_bytes"]):
        raise C2PAPairedOperatorError("C2PA wheel size mismatch")
    if sha256_file(wheel_path) != runtime["wheel_sha256"]:
        raise C2PAPairedOperatorError("C2PA wheel hash mismatch")
    evidence = _read_json(evidence_path)
    if evidence.get("decision") != parent["required_decision"]:
        raise C2PAPairedOperatorError("P210 decision mismatch")
    if evidence.get("allowed_use") != parent["required_allowed_use"]:
        raise C2PAPairedOperatorError("P210 rights scope mismatch")
    if evidence.get("report_id") != parent["report_id"]:
        raise C2PAPairedOperatorError("P210 report identity mismatch")
    if evidence.get("source_fact_id") != parent["source_fact_id"]:
        raise C2PAPairedOperatorError("P210 source-fact identity mismatch")
    strict = set(parent["strict_classifications"])
    rows = [row for row in evidence["rows"] if row["classification"] in strict]
    if len(rows) != int(parent["expected_strict_rows"]):
        raise C2PAPairedOperatorError("strict P210 row count mismatch")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise C2PAPairedOperatorError("duplicate P210 row identity")
    module = _load_p210_module(script_path)
    package_root = _resolve(root, runtime["c2pa_python_root"])
    c2pa, package_tree_id = module._load_c2pa(package_root)
    if package_tree_id != runtime["package_tree_id"]:
        raise C2PAPairedOperatorError("C2PA package tree mismatch")
    return contract, {"module": module, "c2pa": c2pa, "rows": rows}


def _locate_current_files(
    module: Any,
    input_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    expected = {str(row["current_file_sha256"]) for row in rows}
    found: dict[str, Path] = {}
    for path in module._iter_jpegs(input_root):
        try:
            header = module._inspect_jpeg_path(path)
        except (OSError, ValueError):
            continue
        if not header.embedded_c2pa_signal:
            continue
        digest = sha256_file(path)
        if digest in expected:
            if digest in found:
                raise C2PAPairedOperatorError("duplicate current-file identity")
            found[digest] = path
    if set(found) != expected:
        missing = sorted(expected - set(found))
        raise C2PAPairedOperatorError(
            f"missing hash-bound current files: {len(missing)}"
        )
    return found


def _extract_parent_thumbnail(
    c2pa: Any, module: Any, path: Path, maximum_bytes: int
) -> bytes:
    context = c2pa.Context.from_dict({"verify": {"remote_manifest_fetch": False}})
    if not context.is_valid:
        raise C2PAPairedOperatorError("invalid no-remote C2PA context")
    try:
        with c2pa.Reader(path, context=context) as reader:
            if reader.get_remote_url() is not None:
                raise C2PAPairedOperatorError("remote C2PA URL is forbidden")
            manifest = reader.get_active_manifest()
            ingredients = manifest.get("ingredients") if isinstance(manifest, dict) else None
            if not isinstance(ingredients, list) or len(ingredients) != 1:
                raise C2PAPairedOperatorError("parent ingredient count mismatch")
            ingredient = ingredients[0]
            if not isinstance(ingredient, dict) or ingredient.get("relationship") != "parentOf":
                raise C2PAPairedOperatorError("parentOf relationship mismatch")
            thumbnail = ingredient.get("thumbnail")
            identifier = thumbnail.get("identifier") if isinstance(thumbnail, dict) else None
            if not isinstance(identifier, str) or not identifier.startswith("self#jumbf="):
                raise C2PAPairedOperatorError("embedded parent thumbnail is missing")
            sink = module._BoundedBytesIO(maximum_bytes)
            count = reader.resource_to_stream(identifier, sink)
            payload = sink.getvalue()
            if count != len(payload) or not payload:
                raise C2PAPairedOperatorError("parent thumbnail length mismatch")
            return payload
    finally:
        context.close()


def _pil_to_srgb(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    profile = image.info.get("icc_profile")
    if profile:
        try:
            image = ImageCms.profileToProfile(
                image.convert("RGB"),
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except Exception as exc:
            raise C2PAPairedOperatorError("parent ICC conversion failed") from exc
    else:
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def _decode_parent(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return _pil_to_srgb(image)


def _decode_current(path: Path) -> np.ndarray:
    working = load_raster_working_image(path)
    if working.working_space != "linear_srgb" or working.transfer_state != "display_linear":
        raise C2PAPairedOperatorError("current image is not display-linear sRGB")
    return np.asarray(working_image_to_srgb_float(working), dtype=np.float64)


def _resize_maximum_side(rgb: np.ndarray, maximum_side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = float(maximum_side) / max(height, width)
    if abs(scale - 1.0) < 1e-12:
        return rgb.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, size, interpolation=interpolation)
    if not np.isfinite(resized).all():
        raise C2PAPairedOperatorError("registration resize produced non-finite RGB")
    return np.clip(resized, 0.0, 1.0)


def _registration_signal(rgb: np.ndarray) -> np.ndarray:
    encoded = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(encoded, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def register_parent_current(
    parent_rgb: np.ndarray, current_rgb: np.ndarray, settings: Mapping[str, Any]
) -> dict[str, Any]:
    maximum_side = int(settings["registration_image_maximum_side"])
    parent = _resize_maximum_side(parent_rgb, maximum_side)
    current = _resize_maximum_side(current_rgb, maximum_side)
    parent_signal = _registration_signal(parent)
    current_signal = _registration_signal(current)
    sift = cv2.SIFT_create()
    parent_keypoints, parent_descriptors = sift.detectAndCompute(parent_signal, None)
    current_keypoints, current_descriptors = sift.detectAndCompute(current_signal, None)
    if parent_descriptors is None or current_descriptors is None:
        return {"eligible": False, "failure": "missing-sift-descriptors"}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        parent_descriptors, current_descriptors, k=2
    )
    good = [
        first
        for pair in matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < float(settings["lowe_ratio"]) * second.distance
    ]
    if len(good) < int(settings["minimum_good_matches"]):
        return {
            "eligible": False,
            "failure": "insufficient-good-matches",
            "good_matches": len(good),
        }
    parent_points = np.float32(
        [parent_keypoints[item.queryIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    current_points = np.float32(
        [current_keypoints[item.trainIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        parent_points,
        current_points,
        cv2.RANSAC,
        float(settings["ransac_reprojection_threshold_pixels"]),
    )
    if homography is None or inlier_mask is None or not np.isfinite(homography).all():
        return {"eligible": False, "failure": "homography-failed"}
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_fraction = inlier_count / len(good)
    projected = cv2.perspectiveTransform(parent_points, homography)
    reprojection = np.linalg.norm(projected[:, 0] - current_points[:, 0], axis=1)
    median_reprojection = float(np.median(reprojection[inliers]))
    current_to_parent = np.linalg.inv(homography)
    height, width = parent.shape[:2]
    aligned = cv2.warpPerspective(
        current,
        current_to_parent,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpPerspective(
        np.ones(current.shape[:2], dtype=np.uint8),
        current_to_parent,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    overlap = float(np.mean(valid))
    parent_gray = cv2.cvtColor(parent.astype(np.float32), cv2.COLOR_RGB2GRAY)
    aligned_gray = cv2.cvtColor(aligned.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gx_parent = cv2.Sobel(parent_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy_parent = cv2.Sobel(parent_gray, cv2.CV_64F, 0, 1, ksize=3)
    gx_current = cv2.Sobel(aligned_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy_current = cv2.Sobel(aligned_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_parent = np.hypot(gx_parent, gy_parent)[valid]
    gradient_current = np.hypot(gx_current, gy_current)[valid]
    if gradient_parent.size < 2 or min(np.std(gradient_parent), np.std(gradient_current)) <= 0:
        gradient_ncc = -1.0
    else:
        gradient_ncc = float(np.corrcoef(gradient_parent, gradient_current)[0, 1])
    eligible = (
        inlier_count >= int(settings["minimum_inliers"])
        and inlier_fraction >= float(settings["minimum_inlier_fraction"])
        and median_reprojection
        <= float(settings["maximum_median_inlier_reprojection_error_pixels"])
        and overlap >= float(settings["minimum_overlap_fraction"])
        and gradient_ncc >= float(settings["minimum_gradient_ncc"])
    )
    return {
        "eligible": bool(eligible),
        "failure": None if eligible else "registration-gates",
        "good_matches": len(good),
        "inliers": inlier_count,
        "inlier_fraction": inlier_fraction,
        "median_reprojection_error": median_reprojection,
        "overlap_fraction": overlap,
        "gradient_ncc": gradient_ncc,
        "parent": parent,
        "current_aligned": aligned,
        "valid_mask": valid,
    }


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.where(
        value >= 0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def fit_diagonal_logit_affine(
    source: np.ndarray,
    target: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    identity_shrinkage: float,
    maximum_evaluations: int,
) -> np.ndarray:
    parameters = np.zeros((3, 2), dtype=np.float64)
    bound_indices = ((0, 1), (2, 4), (6, 10))
    for channel, (scale_index, shift_index) in enumerate(bound_indices):
        x_raw = source[:, channel]
        y_raw = target[:, channel]
        interior = (x_raw > 0) & (x_raw < 1) & (y_raw > 0) & (y_raw < 1)
        if np.count_nonzero(interior) < 4:
            raise C2PAPairedOperatorError("insufficient diagonal fit support")
        x = _logit(x_raw[interior])
        y = _logit(y_raw[interior])
        lower = np.asarray([lower_bounds[scale_index], lower_bounds[shift_index]])
        upper = np.asarray([upper_bounds[scale_index], upper_bounds[shift_index]])
        span = upper - lower

        def residual(
            value: np.ndarray,
            observed_x: np.ndarray = x,
            observed_y: np.ndarray = y,
            parameter_span: np.ndarray = span,
        ) -> np.ndarray:
            prediction = np.exp(value[0]) * observed_x + value[1]
            regularizer = (
                np.sqrt(identity_shrinkage) * value / parameter_span
            )
            return np.concatenate((prediction - observed_y, regularizer))

        result = least_squares(
            residual,
            x0=np.zeros(2),
            bounds=(lower, upper),
            max_nfev=maximum_evaluations,
            method="trf",
        )
        if not result.success:
            raise C2PAPairedOperatorError("diagonal fit failed")
        parameters[channel] = result.x
    return parameters


def apply_diagonal_logit_affine(source: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    output = np.empty_like(source, dtype=np.float64)
    for channel in range(3):
        values = source[:, channel]
        transformed = _sigmoid(
            np.exp(parameters[channel, 0]) * _logit(values)
            + parameters[channel, 1]
        )
        transformed = np.where(values <= 0.0, 0.0, transformed)
        transformed = np.where(values >= 1.0, 1.0, transformed)
        output[:, channel] = transformed
    return output


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def _improvement(candidate: float, baseline: float) -> float:
    return float((baseline - candidate) / max(baseline, 1e-12))


def fit_and_score_registered(
    registration: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    parent = np.asarray(registration["parent"], dtype=np.float64)
    current = np.asarray(registration["current_aligned"], dtype=np.float64)
    valid = np.asarray(registration["valid_mask"], dtype=bool)
    split = contract["fit_and_audit_split"]
    block = int(split["block_side_pixels"])
    yy, xx = np.indices(valid.shape)
    parity = ((yy // block) + (xx // block)) & 1
    fit_mask = valid & (parity == 0)
    audit_mask = valid & (parity == 1)
    if np.count_nonzero(fit_mask) < int(split["minimum_fit_pixels"]):
        raise C2PAPairedOperatorError("insufficient fit pixels")
    if np.count_nonzero(audit_mask) < int(split["minimum_audit_pixels"]):
        raise C2PAPairedOperatorError("insufficient audit pixels")
    fit_source = parent[fit_mask]
    fit_target = current[fit_mask]
    audit_source = parent[audit_mask]
    audit_target = current[audit_mask]
    operator = contract["operator"]
    lower = np.asarray(operator["lower_bounds"], dtype=np.float64)
    upper = np.asarray(operator["upper_bounds"], dtype=np.float64)
    parameters, fit_success = fit_triangular_logit_transport(
        fit_source.reshape(-1, 1, 3),
        fit_target.reshape(-1, 1, 3),
        lower_bounds=lower,
        upper_bounds=upper,
        sample_stride=int(operator["sample_stride"]),
        identity_shrinkage=float(operator["identity_shrinkage"]),
        maximum_evaluations=int(operator["maximum_fit_evaluations"]),
    )
    if not fit_success:
        raise C2PAPairedOperatorError("triangular fit failed")
    transport, diagnostics = select_safe_transport(
        parameters=parameters,
        grid_size=int(operator["safe_dose_grid_size"]),
        finite_difference=float(operator["safe_dose_finite_difference"]),
        minimum_jacobian_determinant=float(operator["minimum_jacobian_determinant"]),
        maximum_jacobian_condition=float(operator["maximum_jacobian_condition"]),
        bisection_iterations=int(operator["safe_dose_bisection_iterations"]),
    )
    diagonal_parameters = fit_diagonal_logit_affine(
        fit_source,
        fit_target,
        lower,
        upper,
        float(operator["identity_shrinkage"]),
        int(operator["maximum_fit_evaluations"]),
    )
    shift = int(contract["controls"]["shift_pixels"])
    shifted_target = np.roll(current, shift, axis=1)
    shifted_valid = valid & np.roll(valid, shift, axis=1)
    shifted_valid[:, :shift] = False
    shifted_fit_mask = fit_mask & shifted_valid
    if np.count_nonzero(shifted_fit_mask) < int(split["minimum_fit_pixels"]):
        raise C2PAPairedOperatorError("insufficient shifted-control fit pixels")
    shifted_parameters, shifted_success = fit_triangular_logit_transport(
        parent[shifted_fit_mask].reshape(-1, 1, 3),
        shifted_target[shifted_fit_mask].reshape(-1, 1, 3),
        lower_bounds=lower,
        upper_bounds=upper,
        sample_stride=int(operator["sample_stride"]),
        identity_shrinkage=float(operator["identity_shrinkage"]),
        maximum_evaluations=int(operator["maximum_fit_evaluations"]),
    )
    if not shifted_success:
        raise C2PAPairedOperatorError("shifted-control fit failed")
    shifted_transport, _ = select_safe_transport(
        parameters=shifted_parameters,
        grid_size=int(operator["safe_dose_grid_size"]),
        finite_difference=float(operator["safe_dose_finite_difference"]),
        minimum_jacobian_determinant=float(operator["minimum_jacobian_determinant"]),
        maximum_jacobian_condition=float(operator["maximum_jacobian_condition"]),
        bisection_iterations=int(operator["safe_dose_bisection_iterations"]),
    )
    candidate = transport.apply(audit_source)
    diagonal = apply_diagonal_logit_affine(audit_source, diagonal_parameters)
    shifted = shifted_transport.apply(audit_source)
    identity_error = _rmse(audit_source, audit_target)
    candidate_error = _rmse(candidate, audit_target)
    diagonal_error = _rmse(diagonal, audit_target)
    shifted_error = _rmse(shifted, audit_target)
    source_boundary = np.any((audit_source <= 0.0) | (audit_source >= 1.0), axis=1)
    candidate_boundary = np.any((candidate <= 0.0) | (candidate >= 1.0), axis=1)
    new_boundary = float(np.mean(candidate_boundary & ~source_boundary))
    out_of_cube = float(np.mean(np.any((candidate < 0.0) | (candidate > 1.0), axis=1)))
    return {
        "fit_pixels": int(np.count_nonzero(fit_mask)),
        "audit_pixels": int(np.count_nonzero(audit_mask)),
        "parameters": transport.parameters.tolist(),
        "safe_dose": float(transport.dose),
        "diagnostics": diagnostics,
        "errors": {
            "identity_rmse": identity_error,
            "candidate_rmse": candidate_error,
            "diagonal_rmse": diagonal_error,
            "shifted_rmse": shifted_error,
            "candidate_improvement_over_identity": _improvement(
                candidate_error, identity_error
            ),
            "candidate_improvement_over_diagonal": _improvement(
                candidate_error, diagonal_error
            ),
            "candidate_improvement_over_shifted": _improvement(
                candidate_error, shifted_error
            ),
            "candidate_p95_absolute_rgb_error": float(
                np.quantile(np.abs(candidate - audit_target), 0.95)
            ),
        },
        "out_of_cube_fraction": out_of_cube,
        "new_boundary_fraction": new_boundary,
        "audit_absolute_errors": np.abs(candidate - audit_target).reshape(-1),
    }


def _aggregate(rows: list[dict[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    eligible = [row for row in rows if row["status"] == "evaluated"]
    gates = contract["scientific_gates"]
    failures: list[str] = []
    if len(eligible) < int(gates["minimum_registration_eligible_rows"]):
        failures.append("minimum registration-eligible rows")
    if not eligible:
        return {
            "eligible_rows": 0,
            "failed_gates": failures,
            "automatic_pass": False,
        }
    identity = np.asarray(
        [row["evaluation"]["errors"]["candidate_improvement_over_identity"] for row in eligible]
    )
    diagonal = np.asarray(
        [row["evaluation"]["errors"]["candidate_improvement_over_diagonal"] for row in eligible]
    )
    shifted = np.asarray(
        [row["evaluation"]["errors"]["candidate_improvement_over_shifted"] for row in eligible]
    )
    all_errors = np.concatenate(
        [np.asarray(row.pop("_audit_absolute_errors"), dtype=np.float64) for row in eligible]
    )
    diagnostics = [row["evaluation"]["diagnostics"] for row in eligible]
    safe_doses = [row["evaluation"]["safe_dose"] for row in eligible]
    out_of_cube = [row["evaluation"]["out_of_cube_fraction"] for row in eligible]
    new_boundary = [row["evaluation"]["new_boundary_fraction"] for row in eligible]
    metrics = {
        "registration_eligible_rows": len(eligible),
        "candidate_improvement_rate_over_identity": float(np.mean(identity > 0.0)),
        "median_improvement_over_identity": float(np.median(identity)),
        "worst_improvement_over_identity": float(np.min(identity)),
        "candidate_win_rate_over_diagonal": float(np.mean(diagonal > 0.0)),
        "median_improvement_over_diagonal": float(np.median(diagonal)),
        "candidate_win_rate_over_shifted": float(np.mean(shifted > 0.0)),
        "median_improvement_over_shifted": float(np.median(shifted)),
        "population_p95_absolute_rgb_error": float(np.quantile(all_errors, 0.95)),
        "minimum_safe_dose": float(min(safe_doses)),
        "minimum_jacobian_determinant": float(
            min(item["minimum_determinant"] for item in diagnostics)
        ),
        "maximum_jacobian_condition": float(
            max(item["maximum_condition"] for item in diagnostics)
        ),
        "maximum_nonpositive_jacobian_count": int(
            max(item["nonpositive_determinant_count"] for item in diagnostics)
        ),
        "maximum_inverse_roundtrip_error": float(
            max(item["maximum_inverse_roundtrip_error"] for item in diagnostics)
        ),
        "maximum_out_of_cube_fraction": float(max(out_of_cube)),
        "maximum_new_boundary_fraction": float(max(new_boundary)),
    }
    comparisons = (
        ("candidate improvement rate", metrics["candidate_improvement_rate_over_identity"], ">=", gates["minimum_candidate_improvement_rate_over_identity"]),
        ("median identity improvement", metrics["median_improvement_over_identity"], ">=", gates["minimum_median_improvement_over_identity"]),
        ("worst identity improvement", metrics["worst_improvement_over_identity"], ">=", gates["minimum_worst_improvement_over_identity"]),
        ("diagonal win rate", metrics["candidate_win_rate_over_diagonal"], ">=", gates["minimum_candidate_win_rate_over_diagonal"]),
        ("median diagonal improvement", metrics["median_improvement_over_diagonal"], ">=", gates["minimum_median_improvement_over_diagonal"]),
        ("shifted win rate", metrics["candidate_win_rate_over_shifted"], ">=", gates["minimum_candidate_win_rate_over_shifted"]),
        ("median shifted improvement", metrics["median_improvement_over_shifted"], ">=", gates["minimum_median_improvement_over_shifted"]),
        ("population p95 error", metrics["population_p95_absolute_rgb_error"], "<=", gates["maximum_population_p95_absolute_rgb_error"]),
        ("minimum safe dose", metrics["minimum_safe_dose"], ">=", gates["minimum_safe_dose"]),
        ("minimum Jacobian", metrics["minimum_jacobian_determinant"], ">=", gates["minimum_jacobian_determinant"]),
        ("maximum condition", metrics["maximum_jacobian_condition"], "<=", gates["maximum_jacobian_condition"]),
        ("nonpositive Jacobian", metrics["maximum_nonpositive_jacobian_count"], "<=", gates["maximum_nonpositive_jacobian_count"]),
        ("inverse roundtrip", metrics["maximum_inverse_roundtrip_error"], "<=", gates["maximum_inverse_roundtrip_error"]),
        ("out of cube", metrics["maximum_out_of_cube_fraction"], "<=", gates["maximum_out_of_cube_fraction"]),
        ("new boundary", metrics["maximum_new_boundary_fraction"], "<=", gates["maximum_new_boundary_fraction"]),
    )
    for name, actual, operator, expected in comparisons:
        passed = actual >= expected if operator == ">=" else actual <= expected
        if not passed:
            failures.append(name)
    return {"metrics": metrics, "failed_gates": failures, "automatic_pass": not failures}


def evaluate(contract_path: Path, root: Path, output_path: Path, order: str) -> dict[str, Any]:
    if order not in {"canonical", "reverse"}:
        raise C2PAPairedOperatorError("order must be canonical or reverse")
    contract, sources = verify_contract(contract_path, root)
    module = sources["module"]
    c2pa = sources["c2pa"]
    evidence_rows = sorted(sources["rows"], key=lambda row: row["current_file_sha256"])
    paths = _locate_current_files(
        module, _resolve(root, contract["runtime_source"]["logical_input_root"]), evidence_rows
    )
    if order == "reverse":
        evidence_rows.reverse()
    rows: list[dict[str, Any]] = []
    for source_row in evidence_rows:
        current_sha = source_row["current_file_sha256"]
        path = paths[current_sha]
        parent_payload = _extract_parent_thumbnail(
            c2pa,
            module,
            path,
            int(contract["runtime_source"]["maximum_parent_thumbnail_bytes"]),
        )
        if sha256_bytes(parent_payload) != source_row["parent_thumbnail_sha256"]:
            raise C2PAPairedOperatorError("parent thumbnail hash mismatch")
        parent = _decode_parent(parent_payload)
        current = _decode_current(path)
        registration = register_parent_current(
            parent, current, contract["decode_and_registration"]
        )
        public_registration = {
            key: value
            for key, value in registration.items()
            if key not in {"parent", "current_aligned", "valid_mask"}
        }
        row: dict[str, Any] = {
            "row_id": source_row["row_id"],
            "current_file_sha256": current_sha,
            "parent_thumbnail_sha256": source_row["parent_thumbnail_sha256"],
            "classification": source_row["classification"],
            "registration": public_registration,
            "status": "registration-rejected",
        }
        if registration["eligible"]:
            try:
                evaluation = fit_and_score_registered(registration, contract)
                row["_audit_absolute_errors"] = evaluation.pop("audit_absolute_errors")
                row["evaluation"] = evaluation
                row["status"] = "evaluated"
            except (C2PAPairedOperatorError, TriangularLogitTransportError) as exc:
                row["status"] = "fit-rejected"
                row["fit_failure"] = str(exc)
        rows.append(row)
    rows.sort(key=lambda row: row["current_file_sha256"])
    aggregate = _aggregate(rows, contract)
    automatic_pass = bool(aggregate["automatic_pass"])
    status = (
        "PASS_PRIVATE_C2PA_PAIRED_OPERATOR_D0"
        if automatic_pass
        else "FAIL_CLOSED_C2PA_PAIRED_OPERATOR_D0"
    )
    report = {
        "schema": "neuro-film.u5-r2c2pa0-parent-current-explicit-operator-result.v1",
        "experiment_id": contract["experiment_id"],
        "status": status,
        "order": order,
        "contract_sha256": sha256_file(contract_path),
        "parent_evidence_sha256": contract["parent_source_audit"]["evidence_sha256"],
        "strict_source_rows": len(evidence_rows),
        "paths_persisted": False,
        "adjustment_values_read": 0,
        "remote_manifest_fetch": False,
        "rows": rows,
        "aggregate": aggregate,
        "claim_ceiling": contract["claim_ceiling"],
    }
    science = dict(report)
    science.pop("order")
    report["scientific_identity"] = canonical_sha256(science)
    _atomic_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("canonical", "reverse"), required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate(
        args.contract.resolve(strict=True),
        args.root.resolve(strict=True),
        args.output,
        args.order,
    )
    print(json.dumps({"status": result["status"], "scientific_identity": result["scientific_identity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
