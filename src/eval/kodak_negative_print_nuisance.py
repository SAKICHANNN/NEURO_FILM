"""U5.R2AA1 synthetic Kodak 250D-to-2383 nuisance audit.

This evaluator is deliberately isolated from production rendering.  It turns
digitized official graph families into one explicit spectral Look
Approximation, then tests whether that look survives the preregistered
unobserved-spectrum, placement, dye, printer and viewing hypotheses.

Nothing in this module estimates a real digital-to-film or film-to-print
operator.  No real image pixels are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq, nnls

from src.eval.velvia_datasheet_witness import (
    _RGB_TO_XYZ,
    _XYZ_TO_RGB,
    _axis_residual,
    _load_cmf,
    _load_spectrum,
    _pairwise_output_spread,
    _physical_points,
    delta_e76,
    encoded_srgb_to_linear,
    film_discriminating_metamer_direction,
    metamer_extremes,
    reconstruct_reflectances,
    sha256_file,
    xyz_to_lab,
)
from src.real_film.gold_transform_consistency import fit_matrix_affine
from src.roll2film.baselines import fit_joint_basic_adjustment


class KodakNuisanceError(ValueError):
    """Raised when frozen AA1 evidence or numerical state is invalid."""


_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]],
    dtype=np.float64,
)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)
_NEGATIVE_LAYERS = ("blue", "green", "red")
_FORMING_LAYERS = ("yellow_forming", "magenta_forming", "cyan_forming")
_DYES = ("yellow", "magenta", "cyan")


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _collapse_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(x, return_inverse=True)
    reduced = np.asarray([np.mean(y[inverse == index]) for index in range(len(unique))])
    if len(unique) < 3:
        raise KodakNuisanceError("digitized curve has fewer than three unique x samples")
    return unique, reduced


def _interp_curve(
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    *,
    outside: str,
) -> np.ndarray:
    x, y = _collapse_x(np.asarray(x), np.asarray(y))
    result = PchipInterpolator(x, y, extrapolate=False)(target)
    if outside == "zero":
        return np.where(np.isfinite(result), result, 0.0)
    if outside != "edge":
        raise KodakNuisanceError(f"unknown interpolation boundary: {outside}")
    result = np.where(target < x[0], y[0], result)
    result = np.where(target > x[-1], y[-1], result)
    return np.asarray(result, dtype=np.float64)


def validate_sources(root: Path, config: Mapping[str, Any]) -> None:
    aa0 = config["source"]["aa0_decision"]
    decision_path = root / str(aa0["path"])
    import json

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision_branch") != aa0["required_branch"]:
        raise KodakNuisanceError("AA0 activation branch mismatch")
    for record in decision["source_files"]:
        path = root / str(record["path"])
        if path.stat().st_size != int(record["bytes"]):
            raise KodakNuisanceError(f"Kodak source size mismatch: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise KodakNuisanceError(f"Kodak source hash mismatch: {path}")
    for family in ("graphs", "cie"):
        for record in config["source"][family].values():
            path = root / str(record["path"])
            if sha256_file(path) != str(record["sha256"]):
                raise KodakNuisanceError(f"AA1 source hash mismatch: {path}")
    curve = config["source"]["curve_data"]
    if sha256_file(root / str(curve["path"])) != str(curve["sha256"]):
        raise KodakNuisanceError("AA1 curve-data hash mismatch")


def validate_curve_evidence(
    root: Path,
    config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    residuals: dict[str, float] = {}
    distances: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for family, curves in curve_data["curves"].items():
        graph = config["source"]["graphs"][family]
        image = np.asarray(Image.open(root / str(graph["path"])).convert("L"))
        axis = curve_data["graph_axes"][family]
        residual = max(
            _axis_residual(axis["x_value_pixels"]),
            _axis_residual(axis["y_value_pixels"]),
        )
        residuals[family] = residual
        if residual > float(gates["axis_max_residual_px"]):
            raise KodakNuisanceError(f"axis residual failed: {family}")
        ink = image < 128
        for x in axis["known_vertical_grid_pixels"]:
            ink[:, max(0, int(x) - 2) : int(x) + 3] = False
        for y in axis["known_horizontal_grid_pixels"]:
            ink[max(0, int(y) - 2) : int(y) + 3, :] = False
        distance = distance_transform_edt(~ink)
        distances[family] = {}
        counts[family] = {}
        if family.endswith("characteristic"):
            minimum = int(gates["minimum_points_per_characteristic_curve"])
        elif family.endswith("sensitivity"):
            minimum = int(gates["minimum_points_per_sensitivity_curve"])
        else:
            minimum = int(gates["minimum_points_per_dye_curve"])
        for name, points in curves.items():
            coordinates = np.asarray(points, dtype=np.int64)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise KodakNuisanceError(f"invalid curve points: {family}/{name}")
            counts[family][name] = int(len(coordinates))
            if len(coordinates) < minimum:
                raise KodakNuisanceError(f"too few curve points: {family}/{name}")
            if (
                np.any(coordinates[:, 0] < 0)
                or np.any(coordinates[:, 0] >= image.shape[1])
                or np.any(coordinates[:, 1] < 0)
                or np.any(coordinates[:, 1] >= image.shape[0])
            ):
                raise KodakNuisanceError(f"curve point outside graph: {family}/{name}")
            maximum = float(np.max(distance[coordinates[:, 1], coordinates[:, 0]]))
            distances[family][name] = maximum
            if maximum > float(gates["annotation_max_ink_distance_px"]):
                raise KodakNuisanceError(f"curve point is not on source ink: {family}/{name}")
    return {
        "axis_max_residual_px": residuals,
        "annotation_max_ink_distance_px": distances,
        "annotation_counts": counts,
    }


@dataclass(frozen=True)
class FilmCurves:
    wavelength_nm: np.ndarray
    negative_sensitivity: Mapping[str, np.ndarray]
    negative_dyes: Mapping[str, np.ndarray]
    negative_midscale: np.ndarray
    negative_dmin: np.ndarray
    negative_characteristic: Mapping[str, tuple[np.ndarray, np.ndarray]]
    print_sensitivity: Mapping[str, np.ndarray]
    print_dyes: Mapping[str, np.ndarray]
    print_visual_neutral: np.ndarray
    print_characteristic: Mapping[str, tuple[np.ndarray, np.ndarray]]


def _physical(
    curve_data: Mapping[str, Any], family: str, points: Sequence[Sequence[int]]
) -> tuple[np.ndarray, np.ndarray]:
    return _physical_points(points, curve_data["graph_axes"][family])


def build_curve_bank(
    config: Mapping[str, Any], curve_data: Mapping[str, Any]
) -> FilmCurves:
    population = config["synthetic_population"]
    wavelength = np.arange(
        int(population["wavelength_min_nm"]),
        int(population["wavelength_max_nm"]) + int(population["wavelength_step_nm"]),
        int(population["wavelength_step_nm"]),
        dtype=np.float64,
    )
    curves = curve_data["curves"]

    def sensitivity(family: str) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for name, points in curves[family].items():
            x, y = _physical(curve_data, family, points)
            log_value = _interp_curve(x, y, wavelength, outside="zero")
            inside = (wavelength >= np.min(x)) & (wavelength <= np.max(x))
            result[name] = np.where(inside, 10.0**log_value, 0.0)
        return result

    def dyes(family: str, extras: set[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        bases: dict[str, np.ndarray] = {}
        auxiliary: dict[str, np.ndarray] = {}
        for name, points in curves[family].items():
            x, y = _physical(curve_data, family, points)
            value = np.clip(_interp_curve(x, y, wavelength, outside="edge"), 0.0, None)
            (auxiliary if name in extras else bases)[name] = value
        return bases, auxiliary

    def characteristic(family: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, points in curves[family].items():
            x, y = _physical(curve_data, family, points)
            x, y = _collapse_x(x, np.clip(y, 0.0, None))
            # Characteristic curves are physically monotone. Source-raster
            # antialiasing can introduce sub-pixel reversals that would make
            # neutral LAD inversion ambiguous.
            y = np.maximum.accumulate(y)
            result[name] = (x, y)
        return result

    negative_dyes, negative_aux = dyes(
        "250d_dye_density", {"midscale_neutral", "minimum_density"}
    )
    print_dyes, print_aux = dyes("2383_dye_density", {"visual_neutral"})
    return FilmCurves(
        wavelength,
        sensitivity("250d_sensitivity"),
        negative_dyes,
        negative_aux["midscale_neutral"],
        negative_aux["minimum_density"],
        characteristic("250d_characteristic"),
        sensitivity("2383_sensitivity"),
        print_dyes,
        print_aux["visual_neutral"],
        characteristic("2383_characteristic"),
    )


@dataclass(frozen=True)
class SpectralContext:
    wavelength_nm: np.ndarray
    cmf: np.ndarray
    illuminants: Mapping[str, np.ndarray]
    xyz_from_reflectance_d65: np.ndarray
    film_response: np.ndarray


def build_spectral_context(
    root: Path, config: Mapping[str, Any], curves: FilmCurves
) -> SpectralContext:
    cie = config["source"]["cie"]
    wavelength = curves.wavelength_nm
    cmf = _load_cmf(root / str(cie["xyz"]["path"]), wavelength)
    illuminants = {
        name.upper(): _load_spectrum(root / str(record["path"]), wavelength)
        for name, record in cie.items()
        if name != "xyz"
    }
    step = float(config["synthetic_population"]["wavelength_step_nm"])
    d65 = illuminants["D65"]
    normalizer = 1.0 / float(np.sum(d65 * cmf[:, 1]) * step)
    xyz_matrix = normalizer * (cmf * d65[:, None] * step).T
    film = np.vstack(
        [d65 * curves.negative_sensitivity[name] * step for name in _FORMING_LAYERS]
    )
    return SpectralContext(wavelength, cmf, illuminants, xyz_matrix, film)


def _characteristic(
    bank: Mapping[str, tuple[np.ndarray, np.ndarray]],
    channel: str,
    log_exposure: np.ndarray,
) -> np.ndarray:
    x, y = bank[channel]
    return _interp_curve(x, y, np.asarray(log_exposure), outside="edge")


def _inverse_characteristic(
    bank: Mapping[str, tuple[np.ndarray, np.ndarray]], channel: str, density: float
) -> float:
    x, y = bank[channel]
    if density <= y[0]:
        return float(x[0])
    if density >= y[-1]:
        return float(x[-1])
    interpolator = PchipInterpolator(x, y, extrapolate=False)
    return float(brentq(lambda value: float(interpolator(value) - density), x[0], x[-1]))


def _negative_dye_scales(curves: FilmCurves, mapping: str) -> np.ndarray:
    basis = np.column_stack(
        [
            curves.negative_dyes[name]
            / max(float(np.max(curves.negative_dyes[name])), 1e-8)
            for name in _DYES
        ]
    )
    if mapping == "diagonal_peak_normalized":
        return np.ones(3, dtype=np.float64)
    if mapping != "midscale_nnls_scaled":
        raise KodakNuisanceError(f"unknown negative dye mapping: {mapping}")
    target = np.maximum(curves.negative_midscale - curves.negative_dmin, 0.0)
    scales, _ = nnls(basis, target)
    if np.any(scales <= 0.0):
        raise KodakNuisanceError("negative midscale NNLS produced a zero dye scale")
    return scales


def negative_transmittance(
    reflectance: np.ndarray,
    context: SpectralContext,
    curves: FilmCurves,
    placement: float,
    mapping: str,
) -> np.ndarray:
    spectra = np.asarray(reflectance, dtype=np.float64)
    neutral = float(0.18)
    neutral_response = context.film_response @ np.full(context.wavelength_nm.size, neutral)
    response = spectra @ context.film_response.T
    log_exposure = float(placement) + np.log10(
        np.maximum(response / neutral_response[None, :], 1e-12)
    )
    density = np.column_stack(
        [
            _characteristic(curves.negative_characteristic, channel, log_exposure[:, index])
            for index, channel in enumerate(_NEGATIVE_LAYERS)
        ]
    )
    neutral_density = np.asarray(
        [
            _characteristic(
                curves.negative_characteristic, channel, np.asarray([placement])
            )[0]
            for channel in _NEGATIVE_LAYERS
        ]
    )
    floor = np.asarray(
        [np.min(curves.negative_characteristic[channel][1]) for channel in _NEGATIVE_LAYERS]
    )
    ratio = np.maximum(density - floor, 0.0) / np.maximum(neutral_density - floor, 1e-8)
    bases = np.vstack(
        [
            curves.negative_dyes[name] / max(float(np.max(curves.negative_dyes[name])), 1e-8)
            for name in _DYES
        ]
    )
    amount = ratio * _negative_dye_scales(curves, mapping)[None, :]
    spectral_density = amount @ bases
    return 10.0 ** (-spectral_density)


def _printer_primaries(
    context: SpectralContext, curves: FilmCurves, hypothesis: str
) -> np.ndarray:
    wavelength = context.wavelength_nm
    sensitivity = np.vstack([curves.print_sensitivity[name] for name in _FORMING_LAYERS])
    centres = np.asarray(
        [wavelength[int(np.argmax(row))] for row in sensitivity], dtype=np.float64
    )
    if hypothesis.startswith("peak_gaussian_sigma"):
        sigma = float(hypothesis.removeprefix("peak_gaussian_sigma"))
        primaries = np.exp(-0.5 * ((wavelength[None, :] - centres[:, None]) / sigma) ** 2)
    elif hypothesis == "tungsten_sensitivity_weighted":
        primaries = context.illuminants["A"][None, :] * sensitivity
    else:
        raise KodakNuisanceError(f"unknown printer hypothesis: {hypothesis}")
    return primaries / np.maximum(np.max(primaries, axis=1, keepdims=True), 1e-12)


def _print_dye_scales(curves: FilmCurves) -> np.ndarray:
    basis = np.column_stack([curves.print_dyes[name] for name in _DYES])
    scales, _ = nnls(basis, curves.print_visual_neutral)
    if np.any(scales <= 0.0):
        raise KodakNuisanceError("print visual-neutral NNLS produced a zero dye scale")
    return scales


def _adapt_to_d65(
    xyz: np.ndarray,
    source_light: np.ndarray,
    context: SpectralContext,
    step: float,
) -> np.ndarray:
    source_white = (source_light[:, None] * context.cmf).sum(axis=0) * step
    target_light = context.illuminants["D65"]
    target_white = (target_light[:, None] * context.cmf).sum(axis=0) * step
    source_white /= source_white[1]
    target_white /= target_white[1]
    source_lms = _BRADFORD @ source_white
    target_lms = _BRADFORD @ target_white
    matrix = _BRADFORD_INV @ np.diag(target_lms / source_lms) @ _BRADFORD
    return np.asarray(xyz) @ matrix.T


def render_print(
    negative: np.ndarray,
    neutral_negative: np.ndarray,
    context: SpectralContext,
    curves: FilmCurves,
    printer: str,
    viewer: str,
    lad_density: Sequence[float],
) -> dict[str, np.ndarray]:
    step = float(context.wavelength_nm[1] - context.wavelength_nm[0])
    primaries = _printer_primaries(context, curves, printer)
    sensitivity = np.vstack([curves.print_sensitivity[name] for name in _FORMING_LAYERS])
    kernels = primaries * sensitivity * step
    raw = np.asarray(negative) @ kernels.T
    neutral_raw = np.asarray(neutral_negative).reshape(1, -1) @ kernels.T
    target_log = np.asarray(
        [
            _inverse_characteristic(curves.print_characteristic, channel, density)
            for channel, density in zip(_NEGATIVE_LAYERS, lad_density)
        ]
    )
    gains = 10.0**target_log / np.maximum(neutral_raw[0], 1e-20)
    log_exposure = np.log10(np.maximum(raw * gains[None, :], 1e-20))
    density = np.column_stack(
        [
            _characteristic(curves.print_characteristic, channel, log_exposure[:, index])
            for index, channel in enumerate(_NEGATIVE_LAYERS)
        ]
    )
    ratio = density / np.asarray(lad_density, dtype=np.float64)[None, :]
    amount = ratio * _print_dye_scales(curves)[None, :]
    bases = np.vstack([curves.print_dyes[name] for name in _DYES])
    transmittance = 10.0 ** (-(amount @ bases))
    light = context.illuminants[viewer.upper()]
    normalizer = 1.0 / float(np.sum(light * context.cmf[:, 1]) * step)
    xyz = normalizer * ((transmittance * light[None, :]) @ context.cmf) * step
    if viewer.upper() != "D65":
        xyz = _adapt_to_d65(xyz, light, context, step)
    return {
        "xyz": xyz,
        "linear_srgb": xyz @ _XYZ_TO_RGB.T,
        "density": density,
        "transmittance": transmittance,
    }


def render_chain(
    reflectance: np.ndarray,
    context: SpectralContext,
    curves: FilmCurves,
    config: Mapping[str, Any],
    *,
    placement: float,
    mapping: str,
    printer: str,
    viewer: str,
) -> dict[str, np.ndarray]:
    negative = negative_transmittance(reflectance, context, curves, placement, mapping)
    neutral_reflectance = np.full(
        (1, context.wavelength_nm.size),
        float(config["synthetic_population"]["neutral_reflectance"]),
    )
    neutral_negative = negative_transmittance(
        neutral_reflectance, context, curves, placement, mapping
    )[0]
    lad_rgb = np.asarray(
        config["chain"]["print_lad_status_a_density_rgb"], dtype=np.float64
    )
    if lad_rgb.shape != (3,):
        raise KodakNuisanceError("print LAD RGB aim must contain three values")
    return render_print(
        negative,
        neutral_negative,
        context,
        curves,
        printer,
        viewer,
        # Characteristic/dye rows are B/G/R (yellow/magenta/cyan forming),
        # while H-61B and the AA0 observed anchor record aims as R/G/B.
        lad_rgb[::-1],
    )


def _per_channel_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = np.empty_like(target)
    for channel in range(3):
        design = np.column_stack([source[:, channel], np.ones(len(source))])
        parameters = np.linalg.lstsq(design, target[:, channel], rcond=None)[0]
        result[:, channel] = design @ parameters
    return result


def _control_metrics(
    source_rgb: np.ndarray, source_xyz: np.ndarray, target_rgb: np.ndarray, target_xyz: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    identity_de = delta_e76(source_xyz, target_xyz)
    basic = fit_joint_basic_adjustment(source_rgb, target_rgb).apply(source_rgb)
    basic_xyz = basic @ _RGB_TO_XYZ.T
    affine = _per_channel_affine(source_rgb, target_rgb)
    affine_xyz = affine @ _RGB_TO_XYZ.T
    basic_de = delta_e76(basic_xyz, target_xyz)
    affine_de = delta_e76(affine_xyz, target_xyz)
    controls = {
        "identity_delta_e76": _summary(identity_de),
        "joint_basic_delta_e76": _summary(basic_de),
        "per_channel_affine_delta_e76": _summary(affine_de),
        "joint_basic_delta_e76_ge_2_fraction": float(np.mean(basic_de >= 2.0)),
    }
    arrays = {"basic_rgb": basic, "per_channel_affine_rgb": affine}
    positive = config["controls"]["positive_matrix"]
    if positive["enabled"]:
        operator = fit_matrix_affine(
            source_rgb,
            target_rgb,
            np.ones(len(source_rgb), dtype=np.float64),
            positive,
            "linear_srgb",
        )
        predicted = operator.apply(source_rgb)
        controls["bounded_positive_3x3_delta_e76"] = _summary(
            delta_e76(predicted @ _RGB_TO_XYZ.T, target_xyz)
        )
        controls["bounded_positive_3x3_parameters"] = operator.to_dict()
        arrays["bounded_positive_3x3_rgb"] = predicted
    return controls, arrays


def _jacobian_metrics(encoded: np.ndarray, output: np.ndarray, levels: int) -> dict[str, float]:
    grid = output.reshape(levels, levels, levels, 3)
    step = 1.0 / (levels - 1)
    derivatives = []
    for axis in range(3):
        plus = np.take(grid, range(2, levels), axis=axis)
        minus = np.take(grid, range(0, levels - 2), axis=axis)
        derivative = (plus - minus) / (2.0 * step)
        slices = [slice(1, -1), slice(1, -1), slice(1, -1)]
        slices[axis] = slice(None)
        derivatives.append(derivative[tuple(slices)])
    jacobian = np.stack(derivatives, axis=-1).reshape(-1, 3, 3)
    return {
        "determinant_min": float(np.min(np.linalg.det(jacobian))),
        "spectral_norm_max": float(np.max(np.linalg.svd(jacobian, compute_uv=False)[:, 0])),
        "sample_count": int(len(jacobian)),
    }


def _member_key(
    spectrum: str, placement: float, mapping: str, printer: str, viewer: str
) -> str:
    return f"{spectrum}|{placement:.1f}|{mapping}|{printer}|{viewer}"


def _spread_and_medoid(
    outputs: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, str, dict[str, float]]:
    names = list(outputs)
    labs = {name: xyz_to_lab(outputs[name]) for name in names}
    maximum = np.zeros(len(next(iter(outputs.values()))), dtype=np.float64)
    sums = dict.fromkeys(names, 0.0)
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            distance = np.linalg.norm(labs[names[first]] - labs[names[second]], axis=1)
            maximum = np.maximum(maximum, distance)
            mean = float(np.mean(distance))
            sums[names[first]] += mean
            sums[names[second]] += mean
    medoid = min(names, key=lambda name: (sums[name], name))
    return maximum, medoid, {name: float(value) for name, value in sums.items()}


def evaluate_synthetic(
    root: Path,
    config: Mapping[str, Any],
    curve_data: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_sources(root, config)
    trace = validate_curve_evidence(root, config, curve_data)
    curves = build_curve_bank(config, curve_data)
    context = build_spectral_context(root, config, curves)
    population = config["synthetic_population"]
    levels_count = int(population["encoded_srgb_levels_per_channel"])
    levels = np.linspace(0.0, 1.0, levels_count)
    encoded = np.asarray(list(product(levels, repeat=3)), dtype=np.float64)
    source_rgb = encoded_srgb_to_linear(encoded)
    source_xyz = source_rgb @ _RGB_TO_XYZ.T
    smooth = reconstruct_reflectances(source_xyz, context.xyz_from_reflectance_d65)
    metamer_context = type(
        "MetamerContext",
        (),
        {
            "xyz_from_reflectance_d65": context.xyz_from_reflectance_d65,
            "film_response": context.film_response,
        },
    )()
    direction = film_discriminating_metamer_direction(metamer_context)
    positive, negative = metamer_extremes(
        smooth, context.xyz_from_reflectance_d65, direction
    )
    spectra = {
        "smooth": smooth,
        "positive_metamer": positive,
        "negative_metamer": negative,
    }
    reconstruction = {
        name: delta_e76(source_xyz, value @ context.xyz_from_reflectance_d65.T)
        for name, value in spectra.items()
    }
    spectral_rms = np.maximum(
        np.sqrt(np.mean((positive - smooth) ** 2, axis=1)),
        np.sqrt(np.mean((negative - smooth) ** 2, axis=1)),
    )
    chain = config["chain"]
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for spectrum, placement, mapping, printer, viewer in product(
        chain["spectra"],
        chain["negative_neutral_placements"],
        chain["negative_dye_mappings"],
        chain["printer_hypotheses"],
        chain["viewing_illuminants"],
    ):
        key = _member_key(spectrum, float(placement), mapping, printer, viewer)
        outputs[key] = render_chain(
            spectra[spectrum],
            context,
            curves,
            config,
            placement=float(placement),
            mapping=mapping,
            printer=printer,
            viewer=viewer,
        )
    nominal = chain["nominal"]
    nominal_key = _member_key(
        nominal["spectrum"],
        float(nominal["negative_neutral_placement"]),
        nominal["negative_dye_mapping"],
        nominal["printer_hypothesis"],
        nominal["viewing_illuminant"],
    )
    nominal_output = outputs[nominal_key]
    controls, control_arrays = _control_metrics(
        source_rgb, source_xyz, nominal_output["linear_srgb"], nominal_output["xyz"], config
    )
    effect = delta_e76(source_xyz, nominal_output["xyz"])

    def axis_spread(axis: str) -> np.ndarray:
        values = {
            "spectrum": chain["spectra"],
            "placement": chain["negative_neutral_placements"],
            "dye_mapping": chain["negative_dye_mappings"],
            "printer": chain["printer_hypotheses"],
            "viewer": chain["viewing_illuminants"],
        }[axis]
        selected = []
        for value in values:
            params = {
                "spectrum": nominal["spectrum"],
                "placement": float(nominal["negative_neutral_placement"]),
                "dye_mapping": nominal["negative_dye_mapping"],
                "printer": nominal["printer_hypothesis"],
                "viewer": nominal["viewing_illuminant"],
            }
            params[axis] = value
            selected.append(
                outputs[
                    _member_key(
                        params["spectrum"],
                        float(params["placement"]),
                        params["dye_mapping"],
                        params["printer"],
                        params["viewer"],
                    )
                ]["xyz"]
            )
        return _pairwise_output_spread(selected)

    axis_spreads = {axis: axis_spread(axis) for axis in (
        "spectrum", "placement", "dye_mapping", "printer", "viewer"
    )}
    ensemble_spread, medoid, medoid_sums = _spread_and_medoid(
        {name: row["xyz"] for name, row in outputs.items()}
    )
    denominator = np.maximum(effect, 1.0)
    nuisance_ratio = {
        **{axis: spread / denominator for axis, spread in axis_spreads.items()},
        "ensemble": ensemble_spread / denominator,
    }

    neutral_encoded = np.repeat(
        np.linspace(0.0, 1.0, int(population["neutral_ramp_samples"]))[:, None],
        3,
        axis=1,
    )
    neutral_rgb = encoded_srgb_to_linear(neutral_encoded)
    neutral_xyz = neutral_rgb @ _RGB_TO_XYZ.T
    neutral_smooth = reconstruct_reflectances(
        neutral_xyz, context.xyz_from_reflectance_d65
    )
    neutral_positive, neutral_negative = metamer_extremes(
        neutral_smooth, context.xyz_from_reflectance_d65, direction
    )
    neutral_spectra = {
        "smooth": neutral_smooth,
        "positive_metamer": neutral_positive,
        "negative_metamer": neutral_negative,
    }
    neutral_outputs: dict[str, np.ndarray] = {}
    for name in outputs:
        spectrum, placement, mapping, printer, viewer = name.split("|")
        neutral_outputs[name] = render_chain(
            neutral_spectra[spectrum],
            context,
            curves,
            config,
            placement=float(placement),
            mapping=mapping,
            printer=printer,
            viewer=viewer,
        )["xyz"]
    neutral_y_min_step = min(
        float(np.min(np.diff(value[:, 1]))) for value in neutral_outputs.values()
    )
    neutral_chroma = {
        name: np.linalg.norm(xyz_to_lab(value)[:, 1:], axis=1)
        for name, value in neutral_outputs.items()
    }
    nominal_neutral_chroma = float(np.max(neutral_chroma[nominal_key]))
    ensemble_neutral_chroma = float(
        max(np.max(value) for value in neutral_chroma.values())
    )
    all_rgb = np.concatenate([row["linear_srgb"] for row in outputs.values()])
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            smooth,
            positive,
            negative,
            all_rgb,
            *neutral_outputs.values(),
        )
    )
    jacobian = _jacobian_metrics(encoded, nominal_output["linear_srgb"], levels_count)
    gates = config["gates"]
    base_reconstruction = _summary(reconstruction["smooth"])
    metamer_reconstruction = _summary(
        np.maximum(reconstruction["positive_metamer"], reconstruction["negative_metamer"])
    )
    nuisance_summaries = {name: _summary(value) for name, value in nuisance_ratio.items()}
    checks = {
        "finite": bool(finite),
        "base_reconstruction": bool(
            base_reconstruction["median"] <= gates["base_reconstruction_delta_e76_median_max"]
            and base_reconstruction["p95"] <= gates["base_reconstruction_delta_e76_p95_max"]
            and base_reconstruction["maximum"] <= gates["base_reconstruction_delta_e76_max"]
        ),
        "metamer_reconstruction": bool(
            metamer_reconstruction["p95"] <= gates["metamer_reconstruction_delta_e76_p95_max"]
            and metamer_reconstruction["maximum"] <= gates["metamer_reconstruction_delta_e76_max"]
        ),
        "metamers_distinct": bool(
            np.median(spectral_rms) >= gates["metamer_spectral_rms_median_min"]
        ),
        "identity_salience": bool(
            controls["identity_delta_e76"]["median"]
            >= gates["nominal_identity_delta_e76_median_min"]
        ),
        "best_basic_residual": bool(
            controls["joint_basic_delta_e76"]["median"]
            >= gates["nominal_best_basic_residual_delta_e76_median_min"]
            and controls["joint_basic_delta_e76_ge_2_fraction"]
            >= gates["best_basic_residual_delta_e76_ge_2_fraction_min"]
        ),
        "per_channel_affine_residual": bool(
            controls["per_channel_affine_delta_e76"]["median"]
            >= gates["nominal_per_channel_affine_residual_delta_e76_median_min"]
        ),
        "neutral_y_monotone": bool(neutral_y_min_step >= gates["neutral_y_min_step"]),
        "neutral_chroma": bool(
            nominal_neutral_chroma <= gates["nominal_neutral_chroma_max"]
            and ensemble_neutral_chroma <= gates["ensemble_neutral_chroma_max"]
        ),
        "jacobian": bool(
            jacobian["determinant_min"] > gates["nominal_jacobian_determinant_min"]
            and jacobian["spectral_norm_max"] <= gates["nominal_jacobian_spectral_norm_max"]
        ),
        "raw_range": bool(
            np.max(np.abs(all_rgb)) <= gates["raw_linear_srgb_abs_max"]
            and np.mean((all_rgb < 0.0) | (all_rgb > 1.0))
            <= gates["raw_outside_cube_component_fraction_max"]
        ),
    }
    for axis, summary in nuisance_summaries.items():
        threshold = gates["nuisance_ratio"][axis]
        checks[f"nuisance_{axis}"] = bool(
            summary["median"] <= threshold["median_max"]
            and summary["p95"] <= threshold["p95_max"]
        )
    if not all(checks[key] for key in (
        "finite", "base_reconstruction", "metamer_reconstruction", "metamers_distinct"
    )):
        decision = "colourimetric_invalid"
    elif not all(checks[key] for key in (
        "identity_salience", "best_basic_residual", "per_channel_affine_residual"
    )):
        decision = "basic_collapse"
    elif not all(checks[key] for key in checks if key.startswith("nuisance_")):
        decision = "nuisance_unidentified"
    elif not all(checks[key] for key in ("neutral_y_monotone", "neutral_chroma", "jacobian", "raw_range")):
        decision = "numerically_invalid"
    else:
        decision = "stable_nonbasic_candidate"
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "trace_evidence": trace,
        "population": {
            "encoded_srgb_samples": int(len(encoded)),
            "neutral_ramp_samples": int(len(neutral_encoded)),
            "wavelength_samples": int(len(context.wavelength_nm)),
            "ensemble_members": int(len(outputs)),
        },
        "nominal_member": nominal_key,
        "metrics": {
            "base_reconstruction_delta_e76": base_reconstruction,
            "metamer_reconstruction_delta_e76": metamer_reconstruction,
            "metamer_spectral_rms": _summary(spectral_rms),
            "controls": controls,
            "nuisance_spread_delta_e76": {
                **{axis: _summary(value) for axis, value in axis_spreads.items()},
                "ensemble": _summary(ensemble_spread),
            },
            "nuisance_to_nominal_effect_ratio": nuisance_summaries,
            "ensemble_medoid": medoid,
            "ensemble_medoid_distance_sum": medoid_sums[medoid],
            "neutral_y_min_step": neutral_y_min_step,
            "nominal_neutral_chroma_max": nominal_neutral_chroma,
            "ensemble_neutral_chroma_max": ensemble_neutral_chroma,
            "jacobian": jacobian,
            "raw_linear_srgb_min": float(np.min(all_rgb)),
            "raw_linear_srgb_max": float(np.max(all_rgb)),
            "raw_linear_srgb_abs_max": float(np.max(np.abs(all_rgb))),
            "raw_outside_cube_component_fraction": float(
                np.mean((all_rgb < 0.0) | (all_rgb > 1.0))
            ),
            "metamer_direction_observer_residual_max": float(
                np.max(np.abs(context.xyz_from_reflectance_d65 @ direction))
            ),
        },
        "checks": checks,
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "encoded_srgb": encoded,
        "source_linear_srgb": source_rgb,
        "smooth_reflectance": smooth,
        "positive_metamer": positive,
        "negative_metamer": negative,
        "nominal_output_xyz": nominal_output["xyz"],
        "nominal_output_linear_srgb": nominal_output["linear_srgb"],
        "ensemble_spread_delta_e76": ensemble_spread,
        "metamer_direction": direction,
        **control_arrays,
    }
    return report, arrays


def array_fingerprints(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    import hashlib

    result: dict[str, str] = {}
    for name, value in arrays.items():
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


__all__ = [
    "FilmCurves",
    "KodakNuisanceError",
    "SpectralContext",
    "array_fingerprints",
    "build_curve_bank",
    "build_spectral_context",
    "evaluate_synthetic",
    "negative_transmittance",
    "render_chain",
    "render_print",
    "validate_curve_evidence",
    "validate_sources",
]
