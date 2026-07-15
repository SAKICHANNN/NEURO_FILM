"""Sampled FilmSet CT5 metrics and paired evaluator-only oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def evaluate_sampled_candidate(
    input_pixels: np.ndarray,
    target_pixels: np.ndarray,
    output_pixels: np.ndarray,
) -> tuple[dict[str, float | int | None], list[dict[str, float]]]:
    source = _image_pixels(input_pixels)
    target = _image_pixels(target_pixels)
    output = _image_pixels(output_pixels)
    if source.shape != target.shape or source.shape != output.shape:
        raise ValueError("sampled input, target, and output shapes must match")
    if not np.all(np.isfinite(output)):
        raise ValueError("candidate output contains non-finite values")
    source_lab = _to_lab(source)
    target_lab = _to_lab(target)
    output_lab = _to_lab(output)
    fidelity_delta = deltaE_ciede2000(target_lab, output_lab)
    style_delta = deltaE_ciede2000(source_lab, output_lab)
    output_encoded = _linear_to_srgb(output)
    target_encoded = _linear_to_srgb(target)
    per_image: list[dict[str, float]] = []
    for index in range(len(source)):
        rgb_rmse = float(np.sqrt(np.mean((output[index] - target[index]) ** 2)))
        encoded_mse = float(np.mean((output_encoded[index] - target_encoded[index]) ** 2))
        source_chroma = np.linalg.norm(source_lab[index, :, 1:3], axis=-1)
        output_chroma = np.linalg.norm(output_lab[index, :, 1:3], axis=-1)
        source_luma = source[index] @ LUMA
        output_luma = output[index] @ LUMA
        luma_displacement = float(
            np.mean(
                np.abs(
                    np.quantile(output_luma, [0.05, 0.25, 0.5, 0.75, 0.95])
                    - np.quantile(source_luma, [0.05, 0.25, 0.5, 0.75, 0.95])
                )
            )
        )
        per_image.append(
            {
                "mean_delta_e00_to_target": float(np.mean(fidelity_delta[index])),
                "linear_rgb_rmse_to_target": rgb_rmse,
                "srgb_psnr_to_target": float(-10.0 * np.log10(max(encoded_mse, 1e-12))),
                "median_delta_e00_from_input": float(np.median(style_delta[index])),
                "mean_delta_e00_from_input": float(np.mean(style_delta[index])),
                "mean_chroma_delta_from_input": float(np.mean(output_chroma - source_chroma)),
                "luma_quantile_displacement": luma_displacement,
                "rgb_displacement_rms": float(
                    np.sqrt(np.mean((output[index] - source[index]) ** 2))
                ),
                "raw_out_of_range_fraction": float(
                    np.mean((output[index] < 0.0) | (output[index] > 1.0))
                ),
            }
        )
    keys = tuple(per_image[0])
    summary: dict[str, float | int | None] = {
        key: float(np.mean([row[key] for row in per_image])) for key in keys
    }
    summary["images"] = len(per_image)
    summary["sampled_ssim"] = None
    summary["severe_visual_adjudication"] = None
    return summary, per_image


def paired_per_image_affine_oracle(
    input_pixels: np.ndarray,
    target_pixels: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    source = _image_pixels(input_pixels)
    target = _image_pixels(target_pixels)
    if source.shape != target.shape:
        raise ValueError("paired oracle input and target shapes must match")
    rendered = np.empty_like(source)
    regularizer = np.diag([ridge, ridge, ridge, 0.0])
    for index in range(len(source)):
        design = np.concatenate((source[index], np.ones((source.shape[1], 1))), axis=1)
        gram = design.T @ design + regularizer
        parameters = np.linalg.solve(gram, design.T @ target[index])
        rendered[index] = design @ parameters
    return rendered


def summarize_operator_output(operator: Any, input_pixels: np.ndarray) -> np.ndarray:
    source = _image_pixels(input_pixels)
    rendered = operator.apply(source.reshape(-1, 3))
    return rendered.reshape(source.shape)


def _image_pixels(values: np.ndarray) -> np.ndarray:
    pixels = np.asarray(values, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.shape[1] < 16:
        raise ValueError("sampled image pixels must have shape (images, pixels, 3)")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("sampled image pixels must be finite")
    return pixels


def _to_lab(linear_rgb: np.ndarray) -> np.ndarray:
    encoded = _linear_to_srgb(linear_rgb)
    shape = encoded.shape
    return rgb2lab(encoded.reshape(-1, 1, 3)).reshape(shape)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
    )
