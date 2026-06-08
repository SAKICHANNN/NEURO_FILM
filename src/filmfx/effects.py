"""Deterministic film-effect layer generators."""

from __future__ import annotations

import numpy as np

from .fast_blur import gaussian_filter_safe
from .layers import FilmLayer


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    x = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _softplus(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return np.log1p(np.exp(value))


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4)).astype(np.float32)


def grain_residual_layer(
    base_rgb: np.ndarray,
    *,
    strength: float = 0.018,
    seed: int = 7,
    color: bool = True,
    name: str = "grain",
) -> FilmLayer:
    rng = np.random.default_rng(seed)
    lum = luminance(base_rgb)[..., None]
    channels = 3 if color else 1
    noise = rng.normal(0.0, 1.0, size=base_rgb.shape[:2] + (channels,)).astype(np.float32)
    if not color:
        noise = np.repeat(noise, 3, axis=2)
    noise = noise - gaussian_filter_safe(noise, sigma=(1.2, 1.2, 0.0))
    noise = noise - noise.mean(axis=(0, 1), keepdims=True)
    noise = noise / max(float(noise.std()), 1e-6)
    envelope = 0.45 + 0.75 * (1.0 - lum)
    residual = noise * envelope * strength
    residual = residual - residual.mean(axis=(0, 1), keepdims=True)
    return FilmLayer(name=name, mode="residual", residual=residual.astype(np.float32))


def halation_layer(
    base_rgb: np.ndarray,
    *,
    strength: float = 0.16,
    threshold: float = 0.78,
    edge_threshold: float = 0.08,
    min_radius: float = 1.1,
    max_radius: float = 10.0,
    radius_gamma: float = 1.35,
    scale_count: int = 6,
    name: str = "halation",
) -> FilmLayer:
    lum = luminance(base_rgb)
    gy, gx = np.gradient(lum)
    edge = np.hypot(gx, gy)
    highlight = _smoothstep(threshold, 1.0, lum)
    edge_mask = _smoothstep(0.0, edge_threshold, edge)
    support = highlight * edge_mask

    # Approximate a spatially varying scattering radius. Each source pixel gets a
    # continuous target radius from its exposure, then is softly assigned across
    # Gaussian scale-space before blurring.
    scale_count = max(3, int(scale_count))
    min_radius = max(0.4, float(min_radius))
    max_radius = max(min_radius + 0.1, float(max_radius))
    sigmas = np.geomspace(min_radius, max_radius, scale_count).astype(np.float32)
    radius = min_radius + (max_radius - min_radius) * np.power(highlight, radius_gamma)
    log_radius = np.log(np.maximum(radius, 1e-4))[..., None]
    log_sigmas = np.log(sigmas).reshape(1, 1, scale_count)
    bandwidth = max(float(np.log(max_radius / min_radius) / max(scale_count - 1, 1)) * 0.9, 1e-3)
    scale_weights = np.exp(-0.5 * ((log_sigmas - log_radius) / bandwidth) ** 2).astype(np.float32)
    scale_weights /= np.maximum(scale_weights.sum(axis=2, keepdims=True), 1e-6)

    halo = np.zeros_like(lum, dtype=np.float32)
    for index, sigma in enumerate(sigmas):
        source = support * scale_weights[..., index]
        halo += gaussian_filter_safe(source, sigma=float(sigma))

    alpha = np.clip(halo * strength, 0.0, min(0.22, strength))
    rgb = np.zeros_like(base_rgb, dtype=np.float32)
    rgb[..., 0] = 1.0
    rgb[..., 1] = 0.34
    rgb[..., 2] = 0.12
    return FilmLayer(name=name, mode="screen", rgb=rgb, alpha=alpha[..., None].astype(np.float32))


def physical_halation_layer(
    base_rgb: np.ndarray,
    *,
    source_linear_rgb: np.ndarray | None = None,
    source_normalization: str = "percentile",
    source_reference_percentile: float = 99.7,
    profile: str = "cinestill_800t",
    amplify: float = 1.0,
    impact: float = 0.85,
    source_limiter_stops: float = 2.0,
    source_softness: float = 0.42,
    source_gamma: float = 1.45,
    local_diffusion: float = 1.0,
    global_diffusion: float = 0.18,
    hue_green: float = 0.28,
    background_gain: float = 1.25,
    background_luma_target: float = 0.20,
    no_remjet: float | None = None,
    skin_protect: float = 0.55,
    output_alpha_cap: float = 0.32,
    name: str = "physical_halation",
) -> FilmLayer:
    """Build a physically inspired red/orange halation screen layer.

    ``amplify`` changes the secondary exposure coupling. ``impact`` changes the
    final display mix. Keeping these separate avoids treating halation as only a
    red opacity slider.
    """
    base_rgb = np.clip(base_rgb.astype(np.float32), 0.0, 1.0)
    if source_linear_rgb is None:
        linear = _srgb_to_linear(base_rgb)
    else:
        linear = np.maximum(source_linear_rgb.astype(np.float32), 0.0)
        if linear.shape != base_rgb.shape:
            raise ValueError(f"source_linear_rgb shape {linear.shape} does not match base image {base_rgb.shape}")
    y = np.maximum(luminance(linear), 1e-6)
    log_e = np.log2(y / 0.18 + 1e-6)

    if no_remjet is None:
        no_remjet = 1.0 if profile == "cinestill_800t" else 0.35
    if profile == "vision3_500t":
        red_backscatter = 0.40
        green_backscatter = 0.12
    elif profile == "cinestill_800t":
        red_backscatter = 1.00
        green_backscatter = 0.34
    else:
        red_backscatter = 0.72
        green_backscatter = 0.22

    source_raw = _softplus((log_e - float(source_limiter_stops)) / max(float(source_softness), 1e-4))
    source = np.power(source_raw, float(source_gamma))
    if source_normalization == "percentile":
        source = source / max(float(np.percentile(source, float(source_reference_percentile))), 1e-6)
        source = np.clip(source, 0.0, 2.5)
    elif source_normalization != "none":
        raise ValueError(f"Unsupported source_normalization: {source_normalization}")

    maxc = base_rgb.max(axis=2)
    minc = base_rgb.min(axis=2)
    chroma = maxc - minc
    white_hot = _smoothstep(0.62, 0.94, maxc) * (1.0 - _smoothstep(0.22, 0.55, chroma))
    color_hot = _smoothstep(0.58, 0.92, maxc) * _smoothstep(0.05, 0.45, chroma)
    specular_confidence = np.clip(0.45 + 0.70 * white_hot + 0.45 * color_hot, 0.0, 1.35)

    gy, gx = np.gradient(y)
    edge = np.hypot(gx, gy)
    edge_confidence = 0.35 + 0.65 * _smoothstep(0.002, 0.045, edge)
    source = source * specular_confidence * edge_confidence

    bg_sigma = max(4.0, 24.0 * float(local_diffusion))
    # Estimate the surrounding background, not the light source core itself.
    # Otherwise stronger sources can incorrectly suppress their own halo by
    # raising the local mean used by the dark-background gate.
    background_probe = np.minimum(y, 0.35)
    local_mean = gaussian_filter_safe(background_probe, sigma=bg_sigma)
    local_abs = gaussian_filter_safe(np.abs(background_probe - local_mean), sigma=max(2.0, bg_sigma * 0.35))
    dark_visibility = _sigmoid((float(background_luma_target) - local_mean) * float(background_gain) * 10.0)
    contrast_visibility = _smoothstep(0.008, 0.16, local_abs + edge * 2.0)

    r, g, b = base_rgb[..., 0], base_rgb[..., 1], base_rgb[..., 2]
    skin = (
        _smoothstep(0.16, 0.42, r)
        * _smoothstep(0.09, 0.34, g)
        * (1.0 - _smoothstep(0.02, 0.30, b - g))
        * _smoothstep(0.02, 0.20, r - b)
        * (1.0 - _smoothstep(0.42, 0.72, chroma))
    )
    visibility = dark_visibility * (0.45 + 0.55 * contrast_visibility) * (1.0 - np.clip(skin * skin_protect, 0.0, 0.9))

    diffusion = max(0.15, float(local_diffusion))
    red_near = gaussian_filter_safe(source, sigma=2.0 * diffusion)
    red_mid = gaussian_filter_safe(source, sigma=8.0 * diffusion)
    red_tail = gaussian_filter_safe(source, sigma=18.0 * diffusion)
    red_glare = gaussian_filter_safe(source, sigma=max(24.0, 52.0 * diffusion)) * float(global_diffusion)
    red_exposure = 0.50 * red_near + 0.34 * red_mid + 0.16 * red_tail + red_glare

    source_high = _softplus((log_e - (float(source_limiter_stops) + 1.35)) / max(float(source_softness) * 1.15, 1e-4))
    source_high = np.power(source_high, float(source_gamma) + 0.35)
    if source_normalization == "percentile":
        source_high = source_high / max(float(np.percentile(source_high, 99.8)), 1e-6)
        source_high = np.clip(source_high, 0.0, 2.0)
    source_high = source_high * specular_confidence * edge_confidence
    green_near = gaussian_filter_safe(source_high, sigma=0.75 * diffusion)
    green_mid = gaussian_filter_safe(source_high, sigma=3.0 * diffusion)
    green_exposure = 0.68 * green_near + 0.32 * green_mid

    coupling = float(amplify) * float(no_remjet)
    h_r = coupling * red_backscatter * visibility * red_exposure
    h_g = coupling * green_backscatter * float(hue_green) * visibility * green_exposure
    h_b = np.zeros_like(h_r, dtype=np.float32)

    halation = np.stack([h_r, h_g, h_b], axis=2).astype(np.float32)
    energy = np.maximum(halation.max(axis=2, keepdims=True), 1e-6)
    color = np.clip(halation / energy, 0.0, 1.0)
    color[..., 0] = np.maximum(color[..., 0], 0.72)
    color[..., 1] = np.clip(color[..., 1] + 0.10, 0.08, 0.72)
    color[..., 2] = np.minimum(color[..., 2], 0.03)

    alpha = 1.0 - np.exp(-energy[..., 0] * 0.42)
    alpha = np.clip(alpha * float(impact), 0.0, min(float(output_alpha_cap), 1.0))
    return FilmLayer(name=name, mode="screen", rgb=color.astype(np.float32), alpha=alpha[..., None].astype(np.float32))


def density_halation_layer(
    base_rgb: np.ndarray,
    *,
    source_linear_rgb: np.ndarray | None = None,
    source_normalization: str = "percentile",
    source_reference_percentile: float = 99.7,
    amplify: float = 1.0,
    impact: float = 0.85,
    source_limiter_stops: float = 2.2,
    source_softness: float = 0.46,
    source_gamma: float = 1.35,
    local_diffusion: float = 1.0,
    global_diffusion: float = 0.20,
    background_gain: float = 1.25,
    background_luma_target: float = 0.20,
    density_tint: tuple[float, float, float] = (1.0, 0.96, 0.86),
    skin_protect: float = 0.35,
    output_alpha_cap: float = 0.26,
    name: str = "density_halation",
) -> FilmLayer:
    """Build a monochrome/density-domain halation layer.

    This is a separate rule family from color-negative backscatter. It models
    classic black-and-white or clear-base glow as a neutral density response
    instead of red-layer-dominant secondary exposure.
    """

    base_rgb = np.clip(base_rgb.astype(np.float32), 0.0, 1.0)
    if source_linear_rgb is None:
        linear = _srgb_to_linear(base_rgb)
    else:
        linear = np.maximum(source_linear_rgb.astype(np.float32), 0.0)
        if linear.shape != base_rgb.shape:
            raise ValueError(f"source_linear_rgb shape {linear.shape} does not match base image {base_rgb.shape}")
    y = np.maximum(luminance(linear), 1e-6)
    log_e = np.log2(y / 0.18 + 1e-6)

    source_raw = _softplus((log_e - float(source_limiter_stops)) / max(float(source_softness), 1e-4))
    source = np.power(source_raw, float(source_gamma))
    if source_normalization == "percentile":
        source = source / max(float(np.percentile(source, float(source_reference_percentile))), 1e-6)
        source = np.clip(source, 0.0, 2.5)
    elif source_normalization != "none":
        raise ValueError(f"Unsupported source_normalization: {source_normalization}")

    maxc = base_rgb.max(axis=2)
    minc = base_rgb.min(axis=2)
    chroma = maxc - minc
    white_hot = _smoothstep(0.60, 0.94, maxc) * (1.0 - _smoothstep(0.20, 0.60, chroma))
    color_hot = _smoothstep(0.64, 0.96, maxc) * _smoothstep(0.08, 0.55, chroma)
    specular_confidence = np.clip(0.50 + 0.65 * white_hot + 0.25 * color_hot, 0.0, 1.20)

    gy, gx = np.gradient(y)
    edge = np.hypot(gx, gy)
    edge_confidence = 0.30 + 0.70 * _smoothstep(0.002, 0.045, edge)
    source = source * specular_confidence * edge_confidence

    diffusion = max(0.15, float(local_diffusion))
    bg_sigma = max(4.0, 28.0 * diffusion)
    background_probe = np.minimum(y, 0.36)
    local_mean = gaussian_filter_safe(background_probe, sigma=bg_sigma)
    local_abs = gaussian_filter_safe(np.abs(background_probe - local_mean), sigma=max(2.0, bg_sigma * 0.35))
    dark_visibility = _sigmoid((float(background_luma_target) - local_mean) * float(background_gain) * 8.0)
    contrast_visibility = _smoothstep(0.006, 0.14, local_abs + edge * 1.8)

    r, g, b = base_rgb[..., 0], base_rgb[..., 1], base_rgb[..., 2]
    skin = (
        _smoothstep(0.16, 0.42, r)
        * _smoothstep(0.09, 0.34, g)
        * (1.0 - _smoothstep(0.02, 0.30, b - g))
        * _smoothstep(0.02, 0.20, r - b)
        * (1.0 - _smoothstep(0.42, 0.72, chroma))
    )
    visibility = dark_visibility * (0.40 + 0.60 * contrast_visibility) * (1.0 - np.clip(skin * skin_protect, 0.0, 0.75))

    near = gaussian_filter_safe(source, sigma=2.4 * diffusion)
    mid = gaussian_filter_safe(source, sigma=10.0 * diffusion)
    tail = gaussian_filter_safe(source, sigma=26.0 * diffusion)
    glare = gaussian_filter_safe(source, sigma=max(32.0, 70.0 * diffusion)) * float(global_diffusion)
    density_exposure = float(amplify) * visibility * (0.42 * near + 0.33 * mid + 0.25 * tail + glare)

    alpha = 1.0 - np.exp(-density_exposure * 0.34)
    alpha = np.clip(alpha * float(impact), 0.0, min(float(output_alpha_cap), 1.0))
    rgb = np.ones_like(base_rgb, dtype=np.float32)
    tint = np.asarray(density_tint, dtype=np.float32).reshape(1, 1, 3)
    rgb *= np.clip(tint, 0.0, 1.0)
    return FilmLayer(name=name, mode="screen", rgb=rgb.astype(np.float32), alpha=alpha[..., None].astype(np.float32))


def dust_scratch_layer(
    shape: tuple[int, int, int],
    *,
    strength: float = 0.08,
    seed: int = 7,
    name: str = "dust_scratch",
) -> FilmLayer:
    height, width, _ = shape
    rng = np.random.default_rng(seed)
    alpha = np.zeros((height, width, 1), dtype=np.float32)
    speck_count = max(1, int(height * width * 0.00018 * strength * 10.0))
    for _ in range(speck_count):
        y = rng.integers(0, height)
        x = rng.integers(0, width)
        radius = rng.integers(1, 3)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        alpha[y0:y1, x0:x1, 0] = np.maximum(alpha[y0:y1, x0:x1, 0], rng.uniform(0.08, 0.20) * strength)
    scratch_count = max(0, int(width * 0.012 * strength))
    for _ in range(scratch_count):
        x = rng.integers(0, width)
        y0 = rng.integers(0, max(1, height // 3))
        length = rng.integers(max(4, height // 6), max(5, height))
        y1 = min(height, y0 + length)
        alpha[y0:y1, max(0, x - 1) : min(width, x + 1), 0] = np.maximum(
            alpha[y0:y1, max(0, x - 1) : min(width, x + 1), 0],
            rng.uniform(0.03, 0.10) * strength,
        )
    rgb = np.ones(shape, dtype=np.float32)
    return FilmLayer(name=name, mode="alpha", rgb=rgb, alpha=np.clip(alpha, 0.0, 0.25))
