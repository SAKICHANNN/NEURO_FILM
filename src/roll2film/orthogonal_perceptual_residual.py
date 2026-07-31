"""Bounded Lab-orthogonal residual over a validated explicit base look."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.color_engine.gamut import compress_chroma_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.color_engine.srgb_transfer import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)


@dataclass(frozen=True)
class OrthogonalPerceptualResidual:
    """Add only the candidate direction not already expressed by the base.

    The three inputs are fixed display-sRGB renders of the same pixels:
    ``source``, a separately validated ``base``, and a fixed ``candidate``.
    The operation is local, deterministic, and does not infer arm weights.
    """

    residual_strength: float = 0.8
    maximum_orthogonal_delta_e76: float = 12.0
    minimum_lightness: float = 0.1
    maximum_lightness: float = 99.9
    projection_epsilon: float = 1e-12
    gamut_iterations: int = 24
    encoded_margin: float = 1.0 / 65535.0
    row_chunk: int = 128
    numeric_gamut_tolerance: float = 2e-6

    def __post_init__(self) -> None:
        finite = (
            self.residual_strength,
            self.maximum_orthogonal_delta_e76,
            self.minimum_lightness,
            self.maximum_lightness,
            self.projection_epsilon,
            self.encoded_margin,
            self.numeric_gamut_tolerance,
        )
        if (
            not all(np.isfinite(finite))
            or not 0.0 <= self.residual_strength <= 1.0
            or self.maximum_orthogonal_delta_e76 <= 0.0
            or not 0.0 < self.minimum_lightness < self.maximum_lightness
            or self.maximum_lightness >= 100.0
            or self.projection_epsilon <= 0.0
            or self.gamut_iterations <= 0
            or not 0.0 < self.encoded_margin < 0.01
            or self.row_chunk <= 0
            or not 0.0 < self.numeric_gamut_tolerance <= 1e-4
        ):
            raise ValueError("invalid orthogonal perceptual residual")

    @staticmethod
    def _lab(encoded: np.ndarray) -> np.ndarray:
        return linear_rgb_to_lab(
            encoded_srgb_to_linear(encoded).astype(np.float32),
            working_space="linear_srgb",
        ).astype(np.float64)

    def _orthogonal_lab_residual(
        self,
        source: np.ndarray,
        base: np.ndarray,
        candidate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_lab = self._lab(source)
        base_lab = self._lab(base)
        candidate_lab = self._lab(candidate)
        base_residual = base_lab - source_lab
        candidate_residual = candidate_lab - source_lab
        denominator = np.sum(
            base_residual * base_residual,
            axis=-1,
            keepdims=True,
        )
        projection_scale = np.divide(
            np.sum(
                candidate_residual * base_residual,
                axis=-1,
                keepdims=True,
            ),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > self.projection_epsilon,
        )
        orthogonal = candidate_residual - projection_scale * base_residual
        norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
        cap = np.minimum(
            1.0,
            np.divide(
                self.maximum_orthogonal_delta_e76,
                norm,
                out=np.ones_like(norm),
                where=norm > self.projection_epsilon,
            ),
        )
        return base_lab, orthogonal * cap

    @staticmethod
    def _analytical_scale(
        origin: np.ndarray,
        residual: np.ndarray,
        *,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            upper_scale = np.where(
                residual > 0.0,
                (upper - origin) / residual,
                np.inf,
            )
            lower_scale = np.where(
                residual < 0.0,
                (lower - origin) / residual,
                np.inf,
            )
        return np.clip(
            np.min(np.minimum(upper_scale, lower_scale), axis=-1),
            0.0,
            1.0,
        )

    def _apply_chunk(
        self,
        source: np.ndarray,
        base: np.ndarray,
        candidate: np.ndarray,
        *,
        strength: float,
    ) -> np.ndarray:
        base_lab, orthogonal = self._orthogonal_lab_residual(
            source,
            base,
            candidate,
        )
        target_lab = base_lab + strength * orthogonal
        target_lab[..., 0] = np.clip(
            target_lab[..., 0],
            self.minimum_lightness,
            self.maximum_lightness,
        )
        target_lab = compress_chroma_to_working_gamut(
            target_lab.astype(np.float32),
            working_space="linear_srgb",
            iterations=self.gamut_iterations,
            tolerance=self.numeric_gamut_tolerance,
        )

        candidate_linear = lab_to_linear_rgb(
            target_lab,
            working_space="linear_srgb",
        ).astype(np.float64)
        base_linear = encoded_srgb_to_linear(base).astype(np.float64)
        linear_residual = candidate_linear - base_linear
        linear_scale = self._analytical_scale(
            base_linear,
            linear_residual,
            lower=0.0,
            upper=1.0,
        )
        safe_linear = (
            base_linear + linear_residual * linear_scale[..., None]
        )
        excursion = max(
            float(np.max(-safe_linear)),
            float(np.max(safe_linear - 1.0)),
        )
        if excursion > self.numeric_gamut_tolerance:
            raise RuntimeError("analytical linear safety escaped gamut")
        # Numerical closure only after the analytical scale above. This is not
        # used as gamut mapping and is bounded by numeric_gamut_tolerance.
        safe_linear = np.minimum(np.maximum(safe_linear, 0.0), 1.0)
        encoded_candidate = linear_srgb_to_encoded(safe_linear)

        encoded_residual = encoded_candidate - base
        encoded_scale = self._analytical_scale(
            base,
            encoded_residual,
            lower=self.encoded_margin,
            upper=1.0 - self.encoded_margin,
        )
        output = base + encoded_residual * encoded_scale[..., None]
        endpoint = np.all(source == 0.0, axis=-1) | np.all(
            source == 1.0,
            axis=-1,
        )
        return np.where(endpoint[..., None], source, output)

    def apply(
        self,
        source_rgb: np.ndarray,
        base_rgb: np.ndarray,
        candidate_rgb: np.ndarray,
        *,
        strength: float | None = None,
    ) -> np.ndarray:
        arrays = tuple(np.asarray(value) for value in (
            source_rgb,
            base_rgb,
            candidate_rgb,
        ))
        source, base, candidate = arrays
        amount = (
            self.residual_strength if strength is None else float(strength)
        )
        if (
            any(
                value.dtype
                not in {np.dtype(np.float32), np.dtype(np.float64)}
                for value in arrays
            )
            or any(value.shape != source.shape for value in arrays[1:])
            or source.ndim != 3
            or source.shape[-1] != 3
            or source.size == 0
            or any(not np.all(np.isfinite(value)) for value in arrays)
            or any(np.any(value < 0.0) or np.any(value > 1.0) for value in arrays)
            or not np.isfinite(amount)
            or not 0.0 <= amount <= 1.0
        ):
            raise ValueError(
                "expected matching finite HxWx3 encoded RGB and strength in [0,1]"
            )
        if amount == 0.0:
            return np.array(base, copy=True)
        output = np.empty_like(base)
        for y0 in range(0, source.shape[0], self.row_chunk):
            y1 = min(y0 + self.row_chunk, source.shape[0])
            output[y0:y1] = self._apply_chunk(
                source[y0:y1].astype(np.float64),
                base[y0:y1].astype(np.float64),
                candidate[y0:y1].astype(np.float64),
                strength=amount,
            ).astype(base.dtype)
        if (
            not np.all(np.isfinite(output))
            or np.any(output < 0.0)
            or np.any(output > 1.0)
        ):
            raise RuntimeError("orthogonal perceptual residual escaped RGB")
        return output


__all__ = ["OrthogonalPerceptualResidual"]
