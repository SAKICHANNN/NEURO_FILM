"""Synthetic identifiability evaluator for the U5.R2H0A Velvia witness.

This module is deliberately isolated from production rendering. It implements
the frozen, approximate Status-A-density-to-separated-dye witness and measures
how strongly its output depends on RGB metamer choice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh, null_space
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linprog, lsq_linear


class VelviaWitnessError(ValueError):
    """Raised when frozen H0A evidence or numerical state is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _affine_value_to_pixel(pairs: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(pairs, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise VelviaWitnessError("axis pairs must be Nx2")
    return np.polyfit(array[:, 0], array[:, 1], 1)


def _axis_residual(pairs: Sequence[Sequence[float]]) -> float:
    array = np.asarray(pairs, dtype=np.float64)
    fit = _affine_value_to_pixel(pairs)
    return float(np.max(np.abs(np.polyval(fit, array[:, 0]) - array[:, 1])))


def _pixel_to_value(pixel: np.ndarray, fit: np.ndarray) -> np.ndarray:
    return (np.asarray(pixel, dtype=np.float64) - fit[1]) / fit[0]


def _axis_pairs(spec: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    x_key = next(key for key in spec if key.startswith("x_") and key.endswith("_pixels"))
    y_key = next(key for key in spec if key.startswith("y_") and key.endswith("_pixels"))
    return spec[x_key], spec[y_key]


def validate_sources(root: Path, config: Mapping[str, Any]) -> None:
    source = config["source"]
    pdf = root / str(source["pdf"])
    if sha256_file(pdf) != str(source["pdf_sha256"]):
        raise VelviaWitnessError("Velvia source PDF hash mismatch")
    for record in source["cie"].values():
        path = root / str(record["path"])
        if sha256_file(path) != str(record["sha256"]):
            raise VelviaWitnessError(f"CIE source hash mismatch: {path}")


def validate_curve_evidence(
    root: Path,
    config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    graph_specs = config["source"]["embedded_graphs"]
    distances: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    axis_residuals: dict[str, float] = {}
    minimum_keys = {
        "characteristic": "minimum_points_per_characteristic_curve",
        "sensitivity": "minimum_points_per_sensitivity_curve",
        "dye_density": "minimum_points_per_dye_curve",
    }
    for family, curves in curve_data["curves"].items():
        image_path = root / str(curve_data["images"][family])
        expected = str(graph_specs[family]["png_sha256"])
        if sha256_file(image_path) != expected:
            raise VelviaWitnessError(f"embedded graph hash mismatch: {family}")
        image = np.asarray(Image.open(image_path).convert("L"))
        if image.shape != (
            int(graph_specs[family]["height"]),
            int(graph_specs[family]["width"]),
        ):
            raise VelviaWitnessError(f"embedded graph dimensions mismatch: {family}")
        axis = config["graph_axes"][family]
        x_pairs, y_pairs = _axis_pairs(axis)
        residual = max(_axis_residual(x_pairs), _axis_residual(y_pairs))
        axis_residuals[family] = residual
        if residual > float(gates["axis_max_residual_px"]):
            raise VelviaWitnessError(f"axis residual failed: {family}")
        ink = image < 128
        for x in axis["known_vertical_grid_pixels"]:
            ink[:, max(0, int(x) - 2) : int(x) + 3] = False
        for y in axis["known_horizontal_grid_pixels"]:
            ink[max(0, int(y) - 2) : int(y) + 3, :] = False
        distance = distance_transform_edt(~ink)
        distances[family] = {}
        counts[family] = {}
        for name, points in curves.items():
            coordinates = np.asarray(points, dtype=np.int64)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise VelviaWitnessError(f"invalid point array: {family}/{name}")
            count = int(coordinates.shape[0])
            counts[family][name] = count
            if count < int(gates[minimum_keys[family]]):
                raise VelviaWitnessError(f"too few points: {family}/{name}")
            if (
                np.any(coordinates[:, 0] < 0)
                or np.any(coordinates[:, 0] >= image.shape[1])
                or np.any(coordinates[:, 1] < 0)
                or np.any(coordinates[:, 1] >= image.shape[0])
            ):
                raise VelviaWitnessError(f"point outside graph: {family}/{name}")
            maximum = float(np.max(distance[coordinates[:, 1], coordinates[:, 0]]))
            distances[family][name] = maximum
            if maximum > float(gates["annotation_max_ink_distance_px"]):
                raise VelviaWitnessError(
                    f"annotation is not on source ink: {family}/{name} {maximum}"
                )
    return {
        "axis_max_residual_px": axis_residuals,
        "annotation_max_ink_distance_px": distances,
        "annotation_counts": counts,
    }


@dataclass(frozen=True)
class CurveBank:
    wavelength_nm: np.ndarray
    sensitivity: Mapping[str, np.ndarray]
    dye_density: Mapping[str, np.ndarray]
    characteristic_x: Mapping[str, np.ndarray]
    characteristic_y: Mapping[str, np.ndarray]


def _physical_points(
    points: Sequence[Sequence[int]], axis: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    x_pairs, y_pairs = _axis_pairs(axis)
    values = np.asarray(points, dtype=np.float64)
    x = _pixel_to_value(values[:, 0], _affine_value_to_pixel(x_pairs))
    y = _pixel_to_value(values[:, 1], _affine_value_to_pixel(y_pairs))
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) <= 0):
        raise VelviaWitnessError("curve x coordinates must be strictly increasing")
    return x, y


def build_curve_bank(
    config: Mapping[str, Any], curve_data: Mapping[str, Any]
) -> CurveBank:
    witness = config["witness"]
    wavelength = np.arange(
        int(witness["wavelength_min_nm"]),
        int(witness["wavelength_max_nm"]) + int(witness["wavelength_step_nm"]),
        int(witness["wavelength_step_nm"]),
        dtype=np.float64,
    )
    curves = curve_data["curves"]
    sensitivity: dict[str, np.ndarray] = {}
    for name, points in curves["sensitivity"].items():
        x, y = _physical_points(points, config["graph_axes"]["sensitivity"])
        log_value = PchipInterpolator(x, y, extrapolate=False)(wavelength)
        sensitivity[name] = np.where(np.isfinite(log_value), 10.0**log_value, 0.0)
    dye_density: dict[str, np.ndarray] = {}
    for name, points in curves["dye_density"].items():
        x, y = _physical_points(points, config["graph_axes"]["dye_density"])
        value = PchipInterpolator(x, y, extrapolate=False)(wavelength)
        value = np.where(wavelength < x[0], y[0], value)
        value = np.where(wavelength > x[-1], y[-1], value)
        dye_density[name] = np.clip(value, 0.0, None)
    characteristic_x: dict[str, np.ndarray] = {}
    characteristic_y: dict[str, np.ndarray] = {}
    for name, points in curves["characteristic"].items():
        x, y = _physical_points(points, config["graph_axes"]["characteristic"])
        characteristic_x[name] = x
        characteristic_y[name] = np.clip(y, 0.0, None)
    return CurveBank(
        wavelength_nm=wavelength,
        sensitivity=sensitivity,
        dye_density=dye_density,
        characteristic_x=characteristic_x,
        characteristic_y=characteristic_y,
    )


def _load_spectrum(path: Path, wavelength: np.ndarray) -> np.ndarray:
    table = np.loadtxt(path, delimiter=",")
    return np.interp(wavelength, table[:, 0], table[:, 1])


def _load_cmf(path: Path, wavelength: np.ndarray) -> np.ndarray:
    table = np.loadtxt(path, delimiter=",")
    return np.column_stack(
        [np.interp(wavelength, table[:, 0], table[:, index]) for index in (1, 2, 3)]
    )


@dataclass(frozen=True)
class SpectralContext:
    wavelength_nm: np.ndarray
    cmf: np.ndarray
    d65: np.ndarray
    d50: np.ndarray
    xyz_from_reflectance_d65: np.ndarray
    film_response: np.ndarray
    layer_gain: np.ndarray


def build_spectral_context(
    root: Path, config: Mapping[str, Any], curves: CurveBank
) -> SpectralContext:
    cie = config["source"]["cie"]
    wavelength = curves.wavelength_nm
    cmf = _load_cmf(root / cie["xyz"]["path"], wavelength)
    d65 = _load_spectrum(root / cie["d65"]["path"], wavelength)
    d50 = _load_spectrum(root / cie["d50"]["path"], wavelength)
    step = float(config["witness"]["wavelength_step_nm"])
    normalizer = 1.0 / float(np.sum(d65 * cmf[:, 1]) * step)
    xyz_matrix = normalizer * (cmf * d65[:, None] * step).T
    film = np.vstack(
        [
            d65 * curves.sensitivity[name] * step
            for name in ("blue", "green", "red")
        ]
    )
    neutral = float(config["witness"]["neutral_reflectance"])
    target_h = 10.0 ** float(config["witness"]["neutral_log_h"])
    layer_gain = target_h / (film @ np.full(wavelength.size, neutral))
    return SpectralContext(
        wavelength_nm=wavelength,
        cmf=cmf,
        d65=d65,
        d50=d50,
        xyz_from_reflectance_d65=xyz_matrix,
        film_response=film,
        layer_gain=layer_gain,
    )


_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_BRADFORD_D50_TO_D65 = np.array(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float64,
)


def encoded_srgb_to_linear(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.where(
        array <= 0.04045,
        array / 12.92,
        ((array + 0.055) / 1.055) ** 2.4,
    )


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    value = np.asarray(xyz, dtype=np.float64) / _D65_WHITE
    delta = 6.0 / 29.0
    f = np.where(value > delta**3, np.cbrt(value), value / (3 * delta**2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def delta_e76(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz_to_lab(first) - xyz_to_lab(second), axis=-1)


def _second_difference(size: int) -> np.ndarray:
    matrix = np.zeros((size - 2, size), dtype=np.float64)
    for row in range(size - 2):
        matrix[row, row : row + 3] = (1.0, -2.0, 1.0)
    return matrix


def reconstruct_reflectances(
    target_xyz: np.ndarray, xyz_matrix: np.ndarray
) -> np.ndarray:
    targets = np.asarray(target_xyz, dtype=np.float64)
    size = xyz_matrix.shape[1]
    smooth = _second_difference(size)
    colour_weight = 100000.0
    smooth_weight = 0.03
    design = np.vstack((colour_weight * xyz_matrix, np.sqrt(smooth_weight) * smooth))
    tail = np.zeros(smooth.shape[0], dtype=np.float64)
    spectra = []
    for xyz in targets:
        result = lsq_linear(
            design,
            np.concatenate((colour_weight * xyz, tail)),
            bounds=(0.0, 1.0),
            method="bvls",
            tol=1e-12,
            max_iter=2000,
        )
        if not result.success:
            raise VelviaWitnessError(f"reflectance solve failed: {result.message}")
        spectra.append(result.x)
    return np.asarray(spectra, dtype=np.float64)


def film_discriminating_metamer_direction(context: SpectralContext) -> np.ndarray:
    observer_null = null_space(context.xyz_from_reflectance_d65)
    if observer_null.shape[1] == 0:
        raise VelviaWitnessError("observer matrix has no null space")
    film = context.film_response / np.linalg.norm(
        context.film_response, axis=1, keepdims=True
    )
    projected = film @ observer_null
    smooth = _second_difference(observer_null.shape[0]) @ observer_null
    objective = projected.T @ projected
    penalty = np.eye(observer_null.shape[1]) + 10.0 * (smooth.T @ smooth)
    values, vectors = eigh(objective, penalty)
    direction = observer_null @ vectors[:, int(np.argmax(values))]
    direction /= np.max(np.abs(direction))
    if np.max(np.abs(context.xyz_from_reflectance_d65 @ direction)) > 1e-10:
        raise VelviaWitnessError("metamer direction escaped observer null space")
    return direction


def metamer_alternatives(
    base: np.ndarray, direction: np.ndarray, fraction: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    spectra = np.asarray(base, dtype=np.float64)

    def bound(sign: float) -> np.ndarray:
        vector = sign * direction
        limits = np.full(spectra.shape, np.inf, dtype=np.float64)
        positive = vector > 1e-15
        negative = vector < -1e-15
        limits[:, positive] = (1.0 - spectra[:, positive]) / vector[positive]
        limits[:, negative] = spectra[:, negative] / (-vector[negative])
        alpha = fraction * np.min(limits, axis=1)
        alpha = np.where(np.isfinite(alpha), np.maximum(alpha, 0.0), 0.0)
        return np.clip(spectra + alpha[:, None] * vector, 0.0, 1.0)

    return bound(1.0), bound(-1.0)


def metamer_extremes(
    base: np.ndarray,
    xyz_matrix: np.ndarray,
    objective: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find per-colour bounded observer-matched extrema for a spectral objective."""

    spectra = np.asarray(base, dtype=np.float64)
    target = spectra @ xyz_matrix.T
    positive = []
    negative = []
    bounds = [(0.0, 1.0)] * spectra.shape[1]
    for xyz in target:
        high = linprog(
            -objective,
            A_eq=xyz_matrix,
            b_eq=xyz,
            bounds=bounds,
            method="highs",
        )
        low = linprog(
            objective,
            A_eq=xyz_matrix,
            b_eq=xyz,
            bounds=bounds,
            method="highs",
        )
        if not high.success or not low.success:
            raise VelviaWitnessError("bounded metamer linear program failed")
        positive.append(high.x)
        negative.append(low.x)
    return np.asarray(positive), np.asarray(negative)


def _characteristic_density(
    curves: CurveBank, channel: str, log_h: np.ndarray
) -> np.ndarray:
    x = curves.characteristic_x[channel]
    y = curves.characteristic_y[channel]
    interpolator = PchipInterpolator(x, y, extrapolate=False)
    density = interpolator(log_h)
    density = np.where(log_h < x[0], y[0], density)
    density = np.where(log_h > x[-1], y[-1], density)
    return np.asarray(density, dtype=np.float64)


def render_witness(
    reflectance: np.ndarray,
    context: SpectralContext,
    curves: CurveBank,
    illuminant: str = "D65",
) -> dict[str, np.ndarray]:
    spectra = np.asarray(reflectance, dtype=np.float64)
    exposure = (spectra @ context.film_response.T) * context.layer_gain
    log_h = np.log10(np.maximum(exposure, 1e-12))
    layers = ("blue", "green", "red")
    density = np.column_stack(
        [_characteristic_density(curves, name, log_h[:, index]) for index, name in enumerate(layers)]
    )
    dmin = np.array(
        [float(np.min(curves.characteristic_y[name])) for name in layers],
        dtype=np.float64,
    )
    amount = np.maximum(density - dmin, 0.0)
    spectral_density = (
        amount[:, 0, None] * curves.dye_density["yellow"]
        + amount[:, 1, None] * curves.dye_density["magenta"]
        + amount[:, 2, None] * curves.dye_density["cyan"]
    )
    transmittance = 10.0 ** (-spectral_density)
    light = context.d65 if illuminant == "D65" else context.d50
    step = float(context.wavelength_nm[1] - context.wavelength_nm[0])
    normalize = 1.0 / float(np.sum(light * context.cmf[:, 1]) * step)
    xyz = normalize * ((transmittance * light) @ context.cmf) * step
    if illuminant == "D50":
        xyz = xyz @ _BRADFORD_D50_TO_D65.T
    return {
        "xyz": xyz,
        "linear_srgb": xyz @ _XYZ_TO_RGB.T,
        "exposure": exposure,
        "log_h": log_h,
        "density": density,
        "dye_amount": amount,
        "transmittance": transmittance,
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _pairwise_output_spread(outputs: Sequence[np.ndarray]) -> np.ndarray:
    pairs = []
    for first in range(len(outputs)):
        for second in range(first + 1, len(outputs)):
            pairs.append(delta_e76(outputs[first], outputs[second]))
    return np.max(np.stack(pairs, axis=0), axis=0)


def evaluate_synthetic(
    root: Path,
    config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_sources(root, config)
    trace = validate_curve_evidence(root, config, curve_data)
    curves = build_curve_bank(config, curve_data)
    context = build_spectral_context(root, config, curves)
    levels = np.linspace(
        0.0, 1.0, int(config["synthetic_population"]["encoded_srgb_levels_per_channel"])
    )
    encoded = np.asarray(list(product(levels, repeat=3)), dtype=np.float64)
    linear = encoded_srgb_to_linear(encoded)
    target_xyz = linear @ _RGB_TO_XYZ.T
    base = reconstruct_reflectances(target_xyz, context.xyz_from_reflectance_d65)
    direction = film_discriminating_metamer_direction(context)
    positive, negative = metamer_extremes(
        base, context.xyz_from_reflectance_d65, direction
    )
    reconstructed = [
        spectra @ context.xyz_from_reflectance_d65.T
        for spectra in (base, positive, negative)
    ]
    reconstruction_de = [delta_e76(target_xyz, xyz) for xyz in reconstructed]
    rendered_d65 = [render_witness(s, context, curves, "D65") for s in (base, positive, negative)]
    rendered_d50 = [render_witness(s, context, curves, "D50") for s in (base, positive, negative)]
    output_spread = _pairwise_output_spread([row["xyz"] for row in rendered_d65])
    effect = delta_e76(target_xyz, rendered_d65[0]["xyz"])
    neutral_mask = np.isclose(encoded[:, 0], encoded[:, 1]) & np.isclose(encoded[:, 1], encoded[:, 2])
    analysis_mask = (~neutral_mask) & (np.max(encoded, axis=1) >= 0.25)
    ratio = output_spread[analysis_mask] / np.maximum(effect[analysis_mask], 1.0)
    spectral_rms = np.maximum(
        np.sqrt(np.mean((positive - base) ** 2, axis=1)),
        np.sqrt(np.mean((negative - base) ** 2, axis=1)),
    )
    neutral_encoded = np.repeat(
        np.linspace(0.0, 1.0, int(config["synthetic_population"]["neutral_ramp_samples"]))[:, None],
        3,
        axis=1,
    )
    neutral_xyz = encoded_srgb_to_linear(neutral_encoded) @ _RGB_TO_XYZ.T
    neutral_spectra = reconstruct_reflectances(neutral_xyz, context.xyz_from_reflectance_d65)
    neutral_render = render_witness(neutral_spectra, context, curves, "D65")
    neutral_lab = xyz_to_lab(neutral_render["xyz"])
    neutral_chroma = np.linalg.norm(neutral_lab[:, 1:], axis=1)
    neutral_y_steps = np.diff(neutral_render["xyz"][:, 1])
    all_transmittance = np.concatenate(
        [row["transmittance"].reshape(-1) for row in rendered_d65]
    )
    all_rgb = np.concatenate([row["linear_srgb"] for row in rendered_d65], axis=0)
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            base,
            positive,
            negative,
            all_transmittance,
            all_rgb,
            neutral_render["xyz"],
        )
    )
    gates = config["gates"]
    base_summary = _summary(reconstruction_de[0])
    metamer_reconstruction = _summary(
        np.maximum(reconstruction_de[1], reconstruction_de[2])
    )
    metamer_spread = _summary(output_spread[analysis_mask])
    ratio_summary = _summary(ratio)
    distinct_mask = analysis_mask
    checks = {
        "finite": finite,
        "spectra_bounded": bool(
            min(np.min(base), np.min(positive), np.min(negative))
            >= -float(gates["spectrum_bound_tolerance"])
            and max(np.max(base), np.max(positive), np.max(negative))
            <= 1.0 + float(gates["spectrum_bound_tolerance"])
        ),
        "base_reconstruction": bool(
            base_summary["median"] <= gates["base_reconstruction_delta_e76_median_max"]
            and base_summary["p95"] <= gates["base_reconstruction_delta_e76_p95_max"]
            and base_summary["maximum"] <= gates["base_reconstruction_delta_e76_max"]
        ),
        "metamer_reconstruction": bool(
            metamer_reconstruction["p95"] <= gates["metamer_reconstruction_delta_e76_p95_max"]
            and metamer_reconstruction["maximum"] <= gates["metamer_reconstruction_delta_e76_max"]
        ),
        "metamers_distinct": bool(
            np.median(spectral_rms[distinct_mask]) >= gates["metamer_spectral_rms_median_min"]
        ),
        "transmittance_bounded": bool(
            np.min(all_transmittance) >= -float(gates["spectrum_bound_tolerance"])
            and np.max(all_transmittance) <= 1.0 + float(gates["spectrum_bound_tolerance"])
        ),
        "neutral_y_monotone": bool(np.min(neutral_y_steps) >= gates["neutral_y_min_step"]),
        "neutral_chroma": bool(np.max(neutral_chroma) <= gates["neutral_chroma_max"]),
        "metamer_output_spread": bool(
            metamer_spread["median"] <= gates["metamer_output_delta_e76_median_max"]
            and metamer_spread["p95"] <= gates["metamer_output_delta_e76_p95_max"]
        ),
        "metamer_to_effect_ratio": bool(
            ratio_summary["median"] <= gates["metamer_to_effect_ratio_median_max"]
            and ratio_summary["p95"] <= gates["metamer_to_effect_ratio_p95_max"]
        ),
    }
    numerical_keys = (
        "finite",
        "spectra_bounded",
        "base_reconstruction",
        "metamer_reconstruction",
        "metamers_distinct",
        "transmittance_bounded",
        "neutral_y_monotone",
        "neutral_chroma",
    )
    if not all(checks[key] for key in numerical_keys):
        decision = "numerically_invalid"
    elif not checks["metamer_output_spread"] or not checks["metamer_to_effect_ratio"]:
        decision = "canonicalizer_sensitive_unidentified"
    else:
        decision = "stable_enough_for_fixed_visual_pilot"
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "trace_evidence": trace,
        "population": {
            "encoded_srgb_samples": int(encoded.shape[0]),
            "analysis_non_neutral_samples": int(np.sum(analysis_mask)),
            "neutral_ramp_samples": int(neutral_encoded.shape[0]),
            "wavelength_samples": int(context.wavelength_nm.size),
        },
        "metrics": {
            "base_reconstruction_delta_e76": base_summary,
            "metamer_reconstruction_delta_e76": metamer_reconstruction,
            "metamer_spectral_rms": _summary(spectral_rms[distinct_mask]),
            "base_witness_effect_delta_e76": _summary(effect[analysis_mask]),
            "metamer_output_delta_e76": metamer_spread,
            "metamer_to_effect_ratio": ratio_summary,
            "d50_vs_d65_base_output_delta_e76": _summary(
                delta_e76(rendered_d50[0]["xyz"], rendered_d65[0]["xyz"])
            ),
            "neutral_y_min_step": float(np.min(neutral_y_steps)),
            "neutral_chroma_max": float(np.max(neutral_chroma)),
            "transmittance_min": float(np.min(all_transmittance)),
            "transmittance_max": float(np.max(all_transmittance)),
            "raw_linear_srgb_min": float(np.min(all_rgb)),
            "raw_linear_srgb_max": float(np.max(all_rgb)),
            "raw_linear_srgb_out_of_gamut_fraction": float(
                np.mean((all_rgb < 0.0) | (all_rgb > 1.0))
            ),
            "metamer_direction_film_response": (
                context.film_response @ direction
            ).tolist(),
            "metamer_direction_observer_residual_max": float(
                np.max(np.abs(context.xyz_from_reflectance_d65 @ direction))
            ),
        },
        "checks": checks,
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "encoded_srgb": encoded,
        "base_reflectance": base,
        "positive_metamer": positive,
        "negative_metamer": negative,
        "base_output_xyz_d65": rendered_d65[0]["xyz"],
        "positive_output_xyz_d65": rendered_d65[1]["xyz"],
        "negative_output_xyz_d65": rendered_d65[2]["xyz"],
        "base_output_xyz_d50_adapted": rendered_d50[0]["xyz"],
        "metamer_direction": direction,
        "neutral_output_xyz_d65": neutral_render["xyz"],
    }
    return report, arrays


def render_curve_overlays(
    root: Path,
    config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    colours = ["#ff2d2d", "#00a650", "#1769ff"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for family, curves in curve_data["curves"].items():
        image = Image.open(root / curve_data["images"][family]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for colour, (name, points) in zip(colours, sorted(curves.items())):
            xy = [(int(x), int(y)) for x, y in points]
            draw.line(xy, fill=colour, width=2)
            for x, y in xy:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=colour, width=2)
            draw.text((12, 12 + 22 * sorted(curves).index(name)), name, fill=colour)
        path = output_dir / f"{family}_overlay.png"
        image.save(path)
        outputs.append(path)
    return outputs
