from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.color_engine.oklch_local_minde import linear_rec2020_to_oklab, oklab_to_linear_rec2020
from src.color_engine.srgb_transfer import encoded_srgb_to_linear, linear_srgb_to_encoded
from src.preprocess.color_management import linear_rgb_matrix


def encoded_to_oklab(encoded: np.ndarray) -> np.ndarray:
    shape = encoded.shape
    linear = encoded_srgb_to_linear(np.asarray(encoded).reshape(-1, 1, 3))
    rec2020 = linear @ linear_rgb_matrix("linear_srgb", "linear_rec2020").T
    return linear_rec2020_to_oklab(rec2020.astype(np.float32)).reshape(shape)


def oklab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    return oklab_to_linear_rec2020(lab) @ linear_rgb_matrix("linear_rec2020", "linear_srgb").T


def bernstein(lightness: np.ndarray, degree: int = 8) -> np.ndarray:
    value = np.asarray(lightness, dtype=np.float64)
    return np.stack([math.comb(degree, k) * value**k * (1.0-value)**(degree-k) for k in range(degree+1)], axis=-1)


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def basis_weights(lightness: np.ndarray, hue: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    planes = np.asarray(config["lightness_planes"])
    lightness = np.asarray(lightness).reshape(-1)
    hue = np.asarray(hue).reshape(-1)
    band = np.zeros((len(lightness), len(planes)), dtype=np.float64)
    index = np.clip(np.searchsorted(planes, lightness, side="right")-1, 0, len(planes)-2)
    blend = smoothstep((lightness-planes[index]) / (planes[index+1]-planes[index]))
    band[np.arange(len(lightness)), index] = 1.0-blend
    band[np.arange(len(lightness)), index+1] = blend
    count = config["hue_controls"]
    position = np.mod(hue, 2*np.pi) * count / (2*np.pi)
    lower = np.floor(position).astype(int) % count
    blend = smoothstep(position-np.floor(position))
    hue_weight = np.zeros((len(hue), count), dtype=np.float64)
    hue_weight[np.arange(len(hue)), lower] = 1.0-blend
    hue_weight[np.arange(len(hue)), (lower+1) % count] = blend
    return band, (band[:, :, None]*hue_weight[:, None, :]).reshape(len(hue), -1)


def color_design(lab: np.ndarray, tone_lightness: np.ndarray, config: dict) -> np.ndarray:
    values = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    chroma = np.linalg.norm(values[:, 1:], axis=1)
    bands, weights = basis_weights(tone_lightness, np.arctan2(values[:, 2], values[:, 1]), config)
    n, cells = weights.shape
    design = np.zeros((n, 2, 2*cells + 2*bands.shape[1]), dtype=np.float64)
    design[:, 0, :cells] = weights * values[:, 1, None]
    design[:, 1, :cells] = weights * values[:, 2, None]
    design[:, 0, cells:2*cells] = -weights * values[:, 2, None]
    design[:, 1, cells:2*cells] = weights * values[:, 1, None]
    low, high = config["tint_falloff"]
    neutral = 1.0-smoothstep((chroma-low)/(high-low))
    design[:, 0, 2*cells::2] = bands * neutral[:, None]
    design[:, 1, 2*cells+1::2] = bands * neutral[:, None]
    return design


def _quadratic_solve(q, b, constant, initial, lower, upper, constraints, config, scale=None):
    scale = np.ones(len(initial)) if scale is None else np.asarray(scale)
    scaled_q = scale[:, None]*q*scale[None, :]
    scaled_b = scale*b
    linear = [LinearConstraint(matrix*scale[None, :], lo, hi) for matrix, lo, hi in constraints]
    result = minimize(
        lambda z: float(z @ scaled_q @ z - 2*scaled_b @ z + constant), initial/scale,
        jac=lambda z: 2*(scaled_q @ z-scaled_b), method=config["solver"]["method"],
        bounds=Bounds(lower/scale, upper/scale), constraints=linear,
        options={"maxiter": config["solver"]["maxiter"], "ftol": config["solver"]["ftol"]},
    )
    values = result.x*scale
    violation = max(float(np.max(lower-values)), float(np.max(values-upper)), 0.0)
    for matrix, lo, hi in constraints:
        output = matrix @ values
        violation = max(violation, float(np.max(lo-output)), float(np.max(output-hi)))
    receipt = {
        "success": bool(result.success), "status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit), "objective": float(result.fun), "maximum_constraint_violation": violation,
        "finite": bool(np.isfinite(values).all() and np.isfinite(result.fun)),
        "at_box_bound": np.flatnonzero((np.abs(values-lower)<1e-6) | (np.abs(values-upper)<1e-6)).tolist(),
    }
    receipt["valid"] = receipt["success"] and receipt["finite"] and violation <= config["solver"]["constraint_tolerance"]
    return values, receipt


def fit_tone(source_lab: np.ndarray, reference_lab: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    source = source_lab[:, 0]
    reference = reference_lab[:, 0]
    probabilities = config["tone_quantiles"]
    source_quantiles = np.quantile(source, probabilities)
    reference_quantiles = np.quantile(reference, probabilities)
    source_median, reference_median = np.median(source), np.median(reference)
    source_range = max(float(np.quantile(source, .9)-np.quantile(source, .1)), config["tone_min_range"])
    reference_range = max(float(np.quantile(reference, .9)-np.quantile(reference, .1)), config["tone_min_range"])
    desired_range = source_range*np.clip(reference_range/source_range, *config["tone_range_ratio"])
    median_delta = np.clip(reference_median-source_median, -config["tone_median_limit"], config["tone_median_limit"])
    target = source_median+median_delta+desired_range*(reference_quantiles-reference_median)/reference_range
    degree = config["tone_degree"]
    identity = np.linspace(0.0, 1.0, degree+1)
    data = bernstein(source_quantiles, degree)/config["tone_data_scale"]/np.sqrt(len(probabilities))
    right = target/config["tone_data_scale"]/np.sqrt(len(probabilities))
    second = np.diff(np.eye(degree+1), n=2, axis=0)
    smooth = second*np.sqrt(config["tone_smooth_weight"]/len(second))/config["tone_smooth_scale"]
    q = data.T@data+smooth.T@smooth
    lower = identity-config["tone_coefficient_deviation"]
    upper = identity+config["tone_coefficient_deviation"]
    lower[0], upper[0] = config["tone_endpoint_bounds"][0]
    lower[-1], upper[-1] = config["tone_endpoint_bounds"][1]
    differences = np.diff(np.eye(degree+1), axis=0)
    median_row = bernstein(np.asarray([source_median]), degree)
    constraints = [
        (differences, np.full(degree, config["tone_derivative_bounds"][0]/degree), np.full(degree, config["tone_derivative_bounds"][1]/degree)),
        (median_row, np.asarray([source_median-config["tone_median_limit"]]), np.asarray([source_median+config["tone_median_limit"]])),
    ]
    values, receipt = _quadratic_solve(q, data.T@right, float(right@right), identity, lower, upper, constraints, config)
    receipt.update({"source_quantiles": source_quantiles.tolist(), "reference_quantiles": reference_quantiles.tolist(), "target_quantiles": target.tolist(), "source_median": float(source_median)})
    return values, receipt


def _weighted_distribution(values, weights):
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    unique, starts, inverse = np.unique(ordered, return_index=True, return_inverse=True)
    totals = np.add.reduceat(weights[order], starts)
    mid = (np.cumsum(totals)-0.5*totals)/totals.sum()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = mid[inverse]
    return unique, mid, ranks


def _supported(weights, config):
    total = float(weights.sum())
    effective = total*total/max(float(weights@weights), 1e-300)
    return total >= config["minimum_weight"] and effective >= config["minimum_effective_samples"], total, effective


def conditional_targets(source_lab: np.ndarray, reference_lab: np.ndarray, alpha: np.ndarray, config: dict) -> dict:
    source = np.asarray(source_lab, dtype=np.float64)
    reference = np.asarray(reference_lab, dtype=np.float64)
    tone_lightness = bernstein(source[:, 0], config["tone_degree"])@alpha
    source_chroma = np.linalg.norm(source[:, 1:], axis=1)
    reference_chroma = np.linalg.norm(reference[:, 1:], axis=1)
    source_hue = np.arctan2(source[:, 2], source[:, 1])
    reference_hue = np.arctan2(reference[:, 2], reference[:, 1])
    source_bands, source_weights = basis_weights(tone_lightness, source_hue, config)
    reference_bands, reference_weights = basis_weights(reference[:, 0], reference_hue, config)
    source_weights *= (source_chroma >= config["chromatic_threshold"])[:, None]
    reference_weights *= (reference_chroma >= config["chromatic_threshold"])[:, None]
    cells = []
    support = []
    limit = np.deg2rad(config["rotation_degrees"])
    for cell in range(source_weights.shape[1]):
        sw, rw = source_weights[:, cell], reference_weights[:, cell]
        valid_s, sum_s, neff_s = _supported(sw, config)
        valid_r, sum_r, neff_r = _supported(rw, config)
        support.append({"cell": cell, "source_weight": sum_s, "reference_weight": sum_r, "source_neff": neff_s, "reference_neff": neff_r, "supported": bool(valid_s and valid_r)})
        if not (valid_s and valid_r):
            continue
        si, ri = np.flatnonzero(sw > 0), np.flatnonzero(rw > 0)
        _, _, ranks = _weighted_distribution(source_chroma[si], sw[si])
        knots, probabilities, _ = _weighted_distribution(reference_chroma[ri], rw[ri])
        target_chroma = np.interp(np.clip(ranks, *config["rank_clip"]), probabilities, knots)
        source_mean = np.angle(np.sum(sw[si]*np.exp(1j*source_hue[si])))
        reference_mean = np.angle(np.sum(rw[ri]*np.exp(1j*reference_hue[ri])))
        delta = float(np.clip(np.angle(np.exp(1j*(reference_mean-source_mean))), -limit, limit))
        target = target_chroma[:, None]*np.stack([np.cos(source_hue[si]+delta), np.sin(source_hue[si]+delta)], axis=1)
        cells.append((si, sw[si]/sum_s, target))
        support[-1]["target_hue_delta"] = delta
    weight = np.zeros(len(source))
    target_sum = np.zeros((len(source), 2))
    constant = 0.0
    for indexes, cell_weight, target in cells:
        cell_weight = cell_weight/len(cells)
        weight[indexes] += cell_weight
        target_sum[indexes] += cell_weight[:, None]*target
        constant += float(np.sum(cell_weight*np.sum(target*target, axis=1)))
    neutral = []
    for band in range(source_bands.shape[1]):
        sw = source_bands[:, band]*(source_chroma <= config["neutral_threshold"])
        rw = reference_bands[:, band]*(reference_chroma <= config["neutral_threshold"])
        valid_s, sum_s, neff_s = _supported(sw, config)
        valid_r, sum_r, neff_r = _supported(rw, config)
        if valid_s and valid_r:
            neutral.append({"band": band, "weights": sw/sum_s, "target": np.sum(reference[:, 1:]*rw[:, None], axis=0)/sum_r, "source_neff": neff_s, "reference_neff": neff_r})
    return {"source_lab": source, "tone_lightness": tone_lightness, "weight": weight, "target_sum": target_sum, "constant": constant, "neutral": neutral, "support": support, "chromatic_cells": len(cells), "source_supported_pixel_fraction": float((weight > 0).mean())}


def color_data_loss(chroma: np.ndarray, targets: dict, config: dict) -> dict:
    jc = float(np.sum(targets["weight"]*np.sum(chroma*chroma, axis=1))-2*np.sum(chroma*targets["target_sum"])+targets["constant"])/config["color_data_scale"]**2
    jn = 0.0
    for band in targets["neutral"]:
        difference = np.sum(chroma*band["weights"][:, None], axis=0)-band["target"]
        jn += float(difference@difference)/config["neutral_data_scale"]**2/len(targets["neutral"])
    return {"chromatic": jc, "neutral": jn, "total": jc+jn}


def fit_color(targets: dict, config: dict, *, simple: bool, initial: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    design = color_design(targets["source_lab"], targets["tone_lightness"], config)
    n, _, dimension = design.shape
    cells = len(config["lightness_planes"])*config["hue_controls"]
    flat = design.reshape(2*n, dimension)
    weights = np.repeat(targets["weight"], 2)/config["color_data_scale"]**2
    q = flat.T@(flat*weights[:, None])
    b = flat.T@targets["target_sum"].reshape(-1)/config["color_data_scale"]**2
    constant = targets["constant"]/config["color_data_scale"]**2
    for band in targets["neutral"]:
        row = np.einsum("n,ncp->cp", band["weights"], design)/config["neutral_data_scale"]/np.sqrt(len(targets["neutral"]))
        right = band["target"]/config["neutral_data_scale"]/np.sqrt(len(targets["neutral"]))
        q += row.T@row
        b += row.T@right
        constant += float(right@right)
    edge_pairs = []
    hues = config["hue_controls"]
    bands = len(config["lightness_planes"])
    for j in range(bands):
        for k in range(hues):
            edge_pairs.append((j*hues+k, j*hues+(k+1)%hues))
            if j+1 < bands:
                edge_pairs.append((j*hues+k, (j+1)*hues+k))
    for offset, scale_key in ((0, "gain_smooth_scale"), (cells, "rotation_smooth_scale")):
        for left, right in edge_pairs:
            row = np.zeros(dimension)
            row[offset+left], row[offset+right] = 1.0, -1.0
            q += np.outer(row, row)*config["color_smooth_weight"]/len(edge_pairs)/config[scale_key]**2
    for band in range(bands-1):
        for channel in range(2):
            row = np.zeros(dimension)
            row[2*cells+2*band+channel], row[2*cells+2*(band+1)+channel] = 1.0, -1.0
            q += np.outer(row, row)*config["tint_smooth_weight"]/(bands-1)/config["tint_smooth_scale"]**2
    projection = np.eye(dimension)
    if simple:
        projection = np.zeros((dimension, 4))
        projection[:cells, 0] = 1.0
        projection[cells:2*cells, 1] = 1.0
        projection[2*cells::2, 2] = 1.0
        projection[2*cells+1::2, 3] = 1.0
    q, b = projection.T@q@projection, projection.T@b
    gains = 1 if simple else cells
    tint_count = 2 if simple else 2*bands
    dim = 2*gains+tint_count
    tangent = np.tan(np.deg2rad(config["rotation_degrees"]))
    lower = np.r_[np.full(gains, config["gain_bounds"][0]), np.full(gains, -config["gain_bounds"][1]*tangent), np.full(tint_count, -config["tint_bound"])]
    upper = np.r_[np.full(gains, config["gain_bounds"][1]), np.full(gains, config["gain_bounds"][1]*tangent), np.full(tint_count, config["tint_bound"])]
    if targets["chromatic_cells"] == 0:
        lower[:gains] = upper[:gains] = 1.0
        lower[gains:2*gains] = upper[gains:2*gains] = 0.0
    if not targets["neutral"]:
        lower[2*gains:] = upper[2*gains:] = 0.0
    matrix = np.zeros((2*gains, dim))
    for index in range(gains):
        matrix[2*index, index], matrix[2*index, gains+index] = tangent, -1.0
        matrix[2*index+1, index], matrix[2*index+1, gains+index] = tangent, 1.0
    initial = np.r_[np.ones(gains), np.zeros(gains+tint_count)] if initial is None else initial
    scale = np.r_[np.ones(gains), np.full(gains, 0.1), np.full(tint_count, 0.01)]
    values, receipt = _quadratic_solve(q, b, constant, initial, lower, upper, [(matrix, np.zeros(2*gains), np.full(2*gains, np.inf))], config, scale)
    expanded = projection@values
    receipt.update({"simple": simple, "coefficients": values.tolist(), "chromatic_cells": targets["chromatic_cells"], "neutral_bands": len(targets["neutral"]), "no_chromatic_fit": targets["chromatic_cells"] == 0})
    receipt["rotation_constraints_near_active"] = int(np.sum(np.abs(matrix@values)<1e-6))
    receipt["data_loss"] = color_data_loss(np.einsum("ncp,p->nc", design, expanded), targets, config)
    return expanded, receipt


def _ray_coefficients(lightness, direction):
    samples = []
    for c in (0.0, 1.0, -1.0, 2.0):
        samples.append(oklab_to_linear_srgb(np.column_stack([lightness, direction*c])))
    p0, plus, minus, twice = samples
    p2 = (plus+minus)*0.5-p0
    mixed = (plus-minus)*0.5
    p3 = (twice-p0-4*p2-2*mixed)/6
    p1 = mixed-p3
    return np.stack([p0, p1, p2, p3], axis=-1)


def gamut_boundary(lightness: np.ndarray, direction: np.ndarray, iterations: int) -> np.ndarray:
    lightness = np.asarray(lightness, dtype=np.float64).reshape(-1)
    direction = np.asarray(direction, dtype=np.float64).reshape(-1, 2)
    coefficients = _ray_coefficients(lightness, direction)
    p0, p1, p2, p3 = np.moveaxis(coefficients, -1, 0)
    if np.any(p0 < -1e-10) or np.any(p0 > 1.0+1e-10):
        raise ValueError("neutral gamut origin invalid")
    high = np.ones(len(lightness))
    for _ in range(5):
        c = high[:, None]
        value = ((p3*c+p2)*c+p1)*c+p0
        needs = np.all((value >= 0.0) & (value <= 1.0), axis=1)
        if not needs.any():
            break
        high[needs] *= 2.0
    else:
        raise ValueError("gamut upper bracket not found")
    discriminant = (2*p2)**2-12*p3*p1
    quadratic = np.abs(p3) > 1e-14
    valid = quadratic & (discriminant >= 0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = np.where(quadratic, 6*p3, 1.0)
    critical_a = np.where(valid, (-2*p2-root)/denominator, np.inf)
    critical_b = np.where(valid, (-2*p2+root)/denominator, np.inf)
    linear = (~quadratic) & (np.abs(p2)>1e-14)
    critical_a = np.where(linear, -p1/np.where(linear, 2*p2, 1.0), critical_a)
    for critical in (critical_a, critical_b):
        critical[(critical <= 0) | (critical >= high[:, None])] = np.inf
    critical = np.sort(np.stack([critical_a, critical_b, np.broadcast_to(high[:, None], p0.shape)], axis=-1), axis=-1)
    lo = np.zeros_like(p0)
    hi = np.full_like(p0, np.inf)
    previous = np.zeros_like(p0)
    for index in range(3):
        endpoint = critical[..., index]
        finite = np.isfinite(endpoint)
        c = np.where(finite, endpoint, 0.0)
        value = ((p3*c+p2)*c+p1)*c+p0
        first_bad = finite & ~np.isfinite(hi) & ((value < 0.0) | (value > 1.0))
        lo[first_bad] = previous[first_bad]
        hi[first_bad] = c[first_bad]
        previous = np.where(finite, c, previous)
    has_boundary = np.isfinite(hi)
    hi = np.where(has_boundary, hi, high[:, None])
    for _ in range(iterations):
        mid = (lo+hi)*0.5
        value = ((p3*mid+p2)*mid+p1)*mid+p0
        inside = (value >= 0.0) & (value <= 1.0)
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    boundary = np.min(np.where(has_boundary, lo, np.inf), axis=1)
    boundary[(lightness <= 1e-10) | (lightness >= 1-1e-10)] = 0.0
    if not np.isfinite(boundary).all():
        raise ValueError("nonfinite first gamut boundary")
    return boundary


def soft_gamut(lab: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    shape = lab.shape
    values = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    chroma = np.linalg.norm(values[:, 1:], axis=1)
    direction = np.zeros((len(values), 2))
    chromatic = chroma > 1e-12
    direction[chromatic] = values[chromatic, 1:]/chroma[chromatic, None]
    direction[~chromatic, 0] = 1.0
    maximum = gamut_boundary(values[:, 0], direction, config["gamut_iterations"])
    knee = config["gamut_knee"]*maximum
    width = maximum-knee
    compressed = chroma.copy()
    active = (chroma > knee) & (width > 0)
    compressed[active] = knee[active]-width[active]*np.expm1(-(chroma[active]-knee[active])/width[active])
    compressed[width <= 0] = 0.0
    mapped = np.column_stack([values[:, 0], direction*compressed[:, None]])
    linear = oklab_to_linear_srgb(mapped)
    clamped = np.clip(linear, 0.0, 1.0)
    clamp = np.abs(linear-clamped)
    if not np.isfinite(linear).all() or float(clamp.max()) > config["gamut_roundoff_tolerance"]:
        raise ValueError("soft gamut requires substantive channel clipping")
    before = oklab_to_linear_srgb(values)
    in_gamut = np.all((before >= 0) & (before <= 1), axis=1)
    ratio = np.ones_like(chroma)
    ratio[chromatic] = compressed[chromatic]/chroma[chromatic]
    metrics = {
        "pre_gamut_outside_fraction": float((~in_gamut).mean()),
        "compressed_fraction": float((chroma-compressed > 1e-8).mean()),
        "in_gamut_compressed_fraction": float((in_gamut & (chroma-compressed > 1e-8)).mean()),
        "mean_chroma_reduction": float(np.mean(chroma-compressed)),
        "minimum_chroma_ratio": float(ratio.min()), "mean_chroma_ratio": float(ratio.mean()),
        "maximum_roundoff_clamp": float(clamp.max()),
        "roundoff_clipped_channel_fraction": float((clamp > 0).mean()),
    }
    return clamped.reshape(shape).astype(np.float32), metrics


def render_lab(source_lab: np.ndarray, alpha: np.ndarray, coefficients: np.ndarray, config: dict) -> np.ndarray:
    shape = source_lab.shape
    values = np.asarray(source_lab).reshape(-1, 3)
    lightness = bernstein(values[:, 0], config["tone_degree"])@alpha
    chroma = np.einsum("ncp,p->nc", color_design(values, lightness, config), coefficients)
    return np.column_stack([lightness, chroma]).reshape(shape)


def render_image(encoded: np.ndarray, alpha: np.ndarray, coefficients: np.ndarray, config: dict, *, rows: int | None = None) -> tuple[np.ndarray, dict]:
    step = rows or config["render_rows"]
    output = np.empty(encoded.shape, dtype=np.float32)
    summaries = []
    for top in range(0, encoded.shape[0], step):
        source = encoded_to_oklab(encoded[top:top+step])
        target = render_lab(source, alpha, coefficients, config)
        linear, metrics = soft_gamut(target, config)
        output[top:top+step] = linear
        summaries.append((linear.shape[0]*linear.shape[1], metrics))
    merged = {}
    for key in summaries[0][1]:
        if key.startswith("maximum"):
            merged[key] = max(item[key] for _, item in summaries)
        elif key.startswith("minimum"):
            merged[key] = min(item[key] for _, item in summaries)
        else:
            merged[key] = sum(count*item[key] for count, item in summaries)/sum(count for count, _ in summaries)
    return output, merged


def encode_preview(linear: np.ndarray) -> np.ndarray:
    return np.floor(linear_srgb_to_encoded(linear)*255.0+0.5).astype(np.uint8)
