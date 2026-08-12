"""Bounded-row, receipt-replayed histogram copula for developed texture."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import gamma, norm

from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.thomas_dc_projection import (
    ThomasDcReceipt,
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)


def _blocks(height: int, block_height: int) -> list[tuple[int, int]]:
    return [(y0, min(height, y0 + block_height)) for y0 in range(0, height, block_height)]


def _quantize(values: np.ndarray, minimum: float, maximum: float, bins: int) -> np.ndarray:
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.uint16)
    return np.rint((values - minimum) * float(bins - 1) / (maximum - minimum)).astype(
        np.uint16
    )


def _midpoint_table(counts: np.ndarray, sample_count: int) -> np.ndarray:
    cumulative = np.cumsum(counts, dtype=np.uint64)
    before = cumulative - counts
    return (before.astype(np.float64) + 0.5 * counts.astype(np.float64)) / float(
        sample_count
    )


def _render_block(receipt: ThomasDcReceipt, y0: int, y1: int, width: int) -> np.ndarray:
    return render_dc_projected_thomas_region(
        receipt, origin_yx=(y0, 0), shape=(y1 - y0, width)
    )


def apply_bounded_multipass_histogram_copula(
    neutral_base: np.ndarray,
    *,
    profile: DensityConditionedThomasProfile,
    prior: ManufacturerCharacteristicPrior,
    layer_seeds: tuple[int, int, int],
    correlation_matrix: np.ndarray,
    canonical_receipt_row_block_height: int,
    rank_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the P4HG copula using only bounded row intermediates."""
    base = np.asarray(neutral_base, dtype=np.float64)
    correlation = np.asarray(correlation_matrix, dtype=np.float64)
    if (
        base.ndim != 3
        or base.shape[-1] != 3
        or not np.all(np.isfinite(base))
        or np.any(base < 0.0)
        or np.any(base > 1.0)
        or correlation.shape != (3, 3)
        or not np.array_equal(correlation, correlation.T)
        or not np.array_equal(np.diag(correlation), np.ones(3))
        or isinstance(rank_bins, bool)
        or not isinstance(rank_bins, int)
        or rank_bins < 2
        or rank_bins > 65536
        or isinstance(canonical_receipt_row_block_height, bool)
        or not isinstance(canonical_receipt_row_block_height, int)
        or canonical_receipt_row_block_height <= 0
    ):
        raise ValueError("invalid bounded histogram copula request")
    cholesky = np.linalg.cholesky(correlation)
    height, width = base.shape[:2]
    blocks = _blocks(height, canonical_receipt_row_block_height)
    receipts = tuple(
        build_thomas_dc_receipt(
            (height, width),
            profile_id=profile.spatial_profile_id,
            particle_sigma_pixels=profile.particle_sigma_samples,
            cluster_sigma_pixels=profile.cluster_sigma_samples,
            mean_offspring=profile.mean_offspring,
            component_seeds=profile.component_seeds,
            realization_seed=seed,
            truncate=profile.truncate,
            canonical_row_block_height=canonical_receipt_row_block_height,
        )
        for seed in layer_seeds
    )
    peak_temporary_bytes = 0

    raw_min = np.full(3, np.inf, dtype=np.float64)
    raw_max = np.full(3, -np.inf, dtype=np.float64)
    for y0, y1 in blocks:
        for index, receipt in enumerate(receipts):
            field = _render_block(receipt, y0, y1, width)
            peak_temporary_bytes = max(peak_temporary_bytes, field.nbytes)
            raw_min[index] = min(raw_min[index], float(np.min(field)))
            raw_max[index] = max(raw_max[index], float(np.max(field)))

    raw_counts = np.zeros((3, rank_bins), dtype=np.uint64)
    for y0, y1 in blocks:
        for index, receipt in enumerate(receipts):
            field = _render_block(receipt, y0, y1, width)
            quantized = _quantize(field, raw_min[index], raw_max[index], rank_bins)
            raw_counts[index] += np.bincount(
                quantized.reshape(-1), minlength=rank_bins
            ).astype(np.uint64)
            peak_temporary_bytes = max(
                peak_temporary_bytes, field.nbytes + quantized.nbytes
            )
    sample_count = height * width
    raw_tables = np.stack(
        [_midpoint_table(raw_counts[index], sample_count) for index in range(3)]
    )

    normal_sum = np.zeros(3, dtype=np.float64)
    normal_cross = np.zeros((3, 3), dtype=np.float64)
    for y0, y1 in blocks:
        normal = np.empty((y1 - y0, width, 3), dtype=np.float64)
        for index, receipt in enumerate(receipts):
            field = _render_block(receipt, y0, y1, width)
            quantized = _quantize(field, raw_min[index], raw_max[index], rank_bins)
            normal[..., index] = norm.ppf(raw_tables[index][quantized])
        flat = normal.reshape(-1, 3)
        normal_sum += np.sum(flat, axis=0, dtype=np.float64)
        normal_cross += flat.T @ flat
        peak_temporary_bytes = max(
            peak_temporary_bytes, normal.nbytes + flat.shape[0] * 2
        )
    normal_mean = normal_sum / sample_count
    centered_cross = normal_cross - sample_count * np.outer(normal_mean, normal_mean)
    variance = np.diag(centered_cross)
    input_correlation = centered_cross / np.sqrt(np.outer(variance, variance))
    eigenvalues, eigenvectors = np.linalg.eigh(input_correlation)
    if np.min(eigenvalues) <= np.finfo(np.float64).eps:
        raise RuntimeError("bounded histogram copula inputs are degenerate")
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    def correlated_block(y0: int, y1: int) -> np.ndarray:
        normal = np.empty((y1 - y0, width, 3), dtype=np.float64)
        for index, receipt in enumerate(receipts):
            field = _render_block(receipt, y0, y1, width)
            quantized = _quantize(field, raw_min[index], raw_max[index], rank_bins)
            normal[..., index] = norm.ppf(raw_tables[index][quantized])
        whitened = np.einsum("...j,jk->...k", normal - normal_mean, whitening)
        return np.einsum("ij,...j->...i", cholesky, whitened)

    correlated_min = np.full(3, np.inf, dtype=np.float64)
    correlated_max = np.full(3, -np.inf, dtype=np.float64)
    for y0, y1 in blocks:
        correlated = correlated_block(y0, y1)
        correlated_min = np.minimum(correlated_min, np.min(correlated, axis=(0, 1)))
        correlated_max = np.maximum(correlated_max, np.max(correlated, axis=(0, 1)))
        peak_temporary_bytes = max(peak_temporary_bytes, correlated.nbytes * 3)

    correlated_counts = np.zeros((3, rank_bins), dtype=np.uint64)
    for y0, y1 in blocks:
        correlated = correlated_block(y0, y1)
        for index in range(3):
            quantized = _quantize(
                correlated[..., index],
                correlated_min[index],
                correlated_max[index],
                rank_bins,
            )
            correlated_counts[index] += np.bincount(
                quantized.reshape(-1), minlength=rank_bins
            ).astype(np.uint64)
        peak_temporary_bytes = max(peak_temporary_bytes, correlated.nbytes * 3)
    correlated_tables = np.stack(
        [
            _midpoint_table(correlated_counts[index], sample_count)
            for index in range(3)
        ]
    )

    output = np.empty(base.shape, dtype=np.float32)
    minimum_sigma = math.inf
    maximum_sigma = 0.0
    degenerate_count = 0
    minimum_density = math.inf
    residual_square_sum = 0.0
    for y0, y1 in blocks:
        base_block = base[y0:y1]
        correlated = correlated_block(y0, y1)
        block_output = np.empty(base_block.shape, dtype=np.float64)
        for index, channel in enumerate(("red", "green", "blue")):
            values = base_block[..., index]
            lower, upper = prior.curves[index].domain
            exposure = lower + values * (upper - lower)
            sigma_d = profile.amplitude_profile.evaluate_channel(prior, channel, exposure)
            sigma = sigma_d * (4.0 * values * (1.0 - values))
            available = np.full(values.shape, np.inf, dtype=np.float64)
            positive = values > 0.0
            available[positive] = -np.log10(values[positive])
            active = (available > np.finfo(np.float64).eps) & (
                sigma > np.finfo(np.float64).tiny
            )
            quantized = _quantize(
                correlated[..., index],
                correlated_min[index],
                correlated_max[index],
                rank_bins,
            )
            uniforms = correlated_tables[index][quantized]
            delta = np.zeros(values.shape, dtype=np.float64)
            if np.any(active):
                selected_density = available[active]
                selected_sigma = sigma[active]
                shape = np.square(selected_density / selected_sigma)
                scale = np.square(selected_sigma) / selected_density
                delta[active] = (
                    gamma.ppf(uniforms[active], a=shape, scale=scale)
                    - selected_density
                )
                minimum_density = min(
                    minimum_density, float(np.min(selected_density + delta[active]))
                )
            block_output[..., index] = values * np.power(10.0, -delta)
            minimum_sigma = min(minimum_sigma, float(np.min(sigma)))
            maximum_sigma = max(maximum_sigma, float(np.max(sigma)))
            degenerate_count += int(np.count_nonzero(~active))
        output[y0:y1] = block_output.astype(np.float32)
        difference = output[y0:y1].astype(np.float64) - base_block
        residual_square_sum += float(np.sum(difference * difference, dtype=np.float64))
        peak_temporary_bytes = max(
            peak_temporary_bytes,
            correlated.nbytes * 3 + block_output.nbytes + difference.nbytes,
        )
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise RuntimeError("bounded histogram copula escaped the unit cube")
    return output, {
        "receipt_ids": [receipt.receipt_id for receipt in receipts],
        "rank_bins": rank_bins,
        "pass_count": 6,
        "canonical_row_block_height": canonical_receipt_row_block_height,
        "peak_live_temporary_bytes": int(peak_temporary_bytes),
        "full_frame_intermediate_count_excluding_input_output": 0,
        "empirical_input_correlation": input_correlation.tolist(),
        "minimum_target_sigma_d": minimum_sigma,
        "maximum_target_sigma_d": maximum_sigma,
        "support_degenerate_fraction": degenerate_count / float(base.size),
        "support_degenerate_residual_absolute": 0.0,
        "minimum_developed_density": minimum_density,
        "bounded_residual_rms": math.sqrt(residual_square_sum / float(base.size)),
        "limited_fraction": 0.0,
        "hard_clipping_used": 0.0,
    }


__all__ = ["apply_bounded_multipass_histogram_copula"]
