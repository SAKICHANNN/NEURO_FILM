"""Source-bound negative controls for the NegClone film fingerprint method."""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
import tarfile
from types import ModuleType
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sdist(
    archive: Path, *, expected_sha256: str, expected_member_count: int
) -> tuple[str, ...]:
    path = archive.resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("NegClone sdist is absent or has the wrong SHA-256")
    with tarfile.open(path, mode="r:gz") as handle:
        members = tuple(member.name for member in handle.getmembers())
    if (
        len(members) != expected_member_count
        or len(set(members)) != len(members)
        or any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in members
        )
    ):
        raise ValueError("NegClone sdist member inventory is invalid")
    return members


def validate_source_files(
    source_root: Path, expected_hashes: dict[str, str]
) -> dict[str, str]:
    root = source_root.resolve()
    observed: dict[str, str] = {}
    for relative, expected in expected_hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"NegClone source path is invalid: {relative}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            raise ValueError(f"NegClone source hash changed: {relative}")
    return observed


def inspect_source_contract(source_root: Path) -> dict[str, bool]:
    root = source_root.resolve()
    fingerprint = (root / "negclone" / "fingerprint.py").read_text(
        encoding="utf-8"
    )
    scanner = (root / "negclone" / "scanner_profiles.py").read_text(
        encoding="utf-8"
    )
    lightroom = (
        root / "negclone" / "presets" / "lightroom.py"
    ).read_text(encoding="utf-8")
    return {
        "unseeded_image_sampling": "random.sample(image_paths, sample_size)"
        in fingerprint,
        "unseeded_grain_patch_sampling": "random.randint(" in fingerprint,
        "within_image_percentile_colour_masks": (
            "np.percentile(luminance, SHADOW_PERCENTILE)" in fingerprint
            and "np.percentile(luminance, HIGHLIGHT_PERCENTILE)" in fingerprint
        ),
        "scene_histogram_tone_curve": (
            "np.histogram(flat, bins=256" in fingerprint
            and "PchipInterpolator(x_unique, y_unique)" in fingerprint
        ),
        "random_texture_grain_measurement": (
            "local_std = float(np.std(patch))" in fingerprint
            and "np.fft.fft2(windowed)" in fingerprint
        ),
        "median_aggregation_only": (
            "_aggregate_grain(grain_profiles)" in fingerprint
            and "_aggregate_color(color_biases)" in fingerprint
            and "_aggregate_tone(tonal_rolloffs)" in fingerprint
        ),
        "scanner_offsets_declared_approximate": (
            "Values are approximate compensation offsets" in scanner
        ),
        "preset_has_tone_colour_grain": all(
            token in lightroom
            for token in (
                "ToneCurvePV2012",
                "ColorGradeShadowHue",
                "GrainAmount",
            )
        ),
        "preset_has_3d_lut": "3D LUT" in lightroom or ".cube" in lightroom,
    }


def build_synthetic_probes() -> dict[str, np.ndarray]:
    red = np.broadcast_to(
        np.array([0.8, 0.2, 0.2], dtype=np.float64),
        (256, 256, 3),
    ).copy()
    blue = np.broadcast_to(
        np.array([0.2, 0.2, 0.8], dtype=np.float64),
        (256, 256, 3),
    ).copy()
    dark_values = np.linspace(0.0, 0.5, 256, dtype=np.float64)
    bright_values = np.linspace(0.5, 1.0, 256, dtype=np.float64)
    dark = np.broadcast_to(
        dark_values[None, :, None], (256, 256, 3)
    ).copy()
    bright = np.broadcast_to(
        bright_values[None, :, None], (256, 256, 3)
    ).copy()
    flat = np.full((256, 256, 3), 0.5, dtype=np.float64)
    yy, xx = np.indices((256, 256))
    checker_luma = 0.1 + 0.8 * ((yy + xx) % 2)
    checker = np.repeat(checker_luma[:, :, None], 3, axis=2)
    return {
        "red_flat": red,
        "blue_flat": blue,
        "dark_gradient": dark,
        "bright_gradient": bright,
        "flat_grey": flat,
        "checker_texture": checker,
    }


def _tuple3(value: Any) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise ValueError("NegClone colour probe returned invalid RGB bias")
    return result


def _curve_array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != 2
        or len(result) == 0
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("NegClone tone probe returned an invalid curve")
    return result


def run_exact_module_probes(
    fingerprint_module: ModuleType | Any, *, random_seed: int
) -> dict[str, Any]:
    probes = build_synthetic_probes()
    colour_red = fingerprint_module._analyze_color_bias(probes["red_flat"])
    colour_blue = fingerprint_module._analyze_color_bias(probes["blue_flat"])
    red_midtones = _tuple3(colour_red.midtones)
    blue_midtones = _tuple3(colour_blue.midtones)

    tone_dark = fingerprint_module._analyze_tonal_rolloff(
        probes["dark_gradient"]
    )
    tone_bright = fingerprint_module._analyze_tonal_rolloff(
        probes["bright_gradient"]
    )
    dark_curve = _curve_array(tone_dark.curve_points)
    bright_curve = _curve_array(tone_bright.curve_points)
    if not np.array_equal(dark_curve[:, 0], bright_curve[:, 0]):
        raise ValueError("NegClone tone probes use inconsistent input knots")

    random.seed(random_seed)
    grain_flat = fingerprint_module._analyze_grain(probes["flat_grey"])
    random.seed(random_seed)
    grain_checker = fingerprint_module._analyze_grain(
        probes["checker_texture"]
    )

    return {
        "colour": {
            "red_flat_midtones": list(red_midtones),
            "blue_flat_midtones": list(blue_midtones),
            "midtone_bias_l2_distance": float(
                np.linalg.norm(np.asarray(red_midtones) - blue_midtones)
            ),
        },
        "tone": {
            "dark": {
                "shadow_lift": float(tone_dark.shadow_lift),
                "highlight_compression": float(
                    tone_dark.highlight_compression
                ),
                "midtone_contrast": float(tone_dark.midtone_contrast),
            },
            "bright": {
                "shadow_lift": float(tone_bright.shadow_lift),
                "highlight_compression": float(
                    tone_bright.highlight_compression
                ),
                "midtone_contrast": float(tone_bright.midtone_contrast),
            },
            "curve_output_maximum_absolute_difference": float(
                np.max(np.abs(dark_curve[:, 1] - bright_curve[:, 1]))
            ),
        },
        "grain": {
            "flat_grey": {
                "mean_intensity": float(grain_flat.mean_intensity),
                "size_estimate": float(grain_flat.size_estimate),
                "clumping_factor": float(grain_flat.clumping_factor),
                "peak_frequency": float(grain_flat.peak_frequency),
                "spectral_slope": float(grain_flat.spectral_slope),
                "spectral_centroid": float(grain_flat.spectral_centroid),
            },
            "checker_texture": {
                "mean_intensity": float(grain_checker.mean_intensity),
                "size_estimate": float(grain_checker.size_estimate),
                "clumping_factor": float(grain_checker.clumping_factor),
                "peak_frequency": float(grain_checker.peak_frequency),
                "spectral_slope": float(grain_checker.spectral_slope),
                "spectral_centroid": float(
                    grain_checker.spectral_centroid
                ),
            },
        },
    }


def evaluate_shortcut_gates(
    probes: dict[str, Any], gates: dict[str, float]
) -> dict[str, bool]:
    return {
        "colour_content_shortcut_detected": probes["colour"][
            "midtone_bias_l2_distance"
        ]
        >= float(gates["minimum_colour_bias_l2_shortcut"]),
        "exposure_histogram_shortcut_detected": probes["tone"][
            "curve_output_maximum_absolute_difference"
        ]
        >= float(gates["minimum_tone_curve_output_difference"]),
        "texture_as_grain_shortcut_detected": (
            probes["grain"]["flat_grey"]["mean_intensity"]
            <= float(gates["maximum_flat_grain_intensity"])
            and probes["grain"]["checker_texture"]["mean_intensity"]
            >= float(gates["minimum_checker_grain_intensity"])
            and probes["grain"]["checker_texture"]["clumping_factor"]
            >= float(gates["minimum_checker_clumping"])
        ),
    }
