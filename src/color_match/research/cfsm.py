"""Canonical Factorized Safe Match (CFSM) explicit-operator challenger.

This first falsifiable candidate isolates three responsibilities:

1. compare one uploaded reference with a fixed, image-independent canonical
   colour prior;
2. estimate a global orientation-preserving Gaussian transport;
3. project that estimate onto a bounded, replayable tetrahedral LUT.

The fitted LUT is fixed for the entire source batch.  It therefore cannot
change a shared input colour merely because surrounding pixels change.  The
canonical prior is deliberately simple in this first leaf; passing numerical
and batch gates does not establish photographic identifiability or aesthetics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import numpy as np

from src.preprocess import WorkingImage
from src.roll2film.constrained import (
    LUTConstraintReport,
    LUTConstraintSpec,
    audit_lut_constraints,
)
from src.roll2film.identification import (
    estimate_affine_spline_transport_operator,
    estimate_gaussian_transport_operator,
)
from src.roll2film.lut import DenseLUT3D
from src.roll2film.operators import AffineColorOperator
from src.roll2film.splines import AffineMonotoneSplineOperator

from ..canonical import canonical_sha256
from ..contracts import ReferenceMatchContractError


CFSM_ALGORITHM_ID = "canonical-factorized-safe-match.gaussian-v0"
CFSM_BATCH_ALGORITHM_ID = "canonical-factorized-safe-match.batch-gaussian-v1"
CFSM_ANALYTIC_ALGORITHM_ID = (
    "canonical-factorized-safe-match.analytic-photo-prior-v1"
)
CFSM_EMPIRICAL_ALGORITHM_ID = (
    "canonical-factorized-safe-match.empirical-photo-prior-v1"
)
CFSM_QUANTILE_ALGORITHM_ID = (
    "canonical-factorized-safe-match.monotone-quantile-v1"
)
CFSM_CANDIDATE_SCHEMA_ID = "neuro-film.cfsm-candidate.v0"
_SUPPORTED_ALGORITHM_IDS = frozenset(
    {
        CFSM_ALGORITHM_ID,
        CFSM_BATCH_ALGORITHM_ID,
        CFSM_ANALYTIC_ALGORITHM_ID,
        CFSM_EMPIRICAL_ALGORITHM_ID,
        CFSM_QUANTILE_ALGORITHM_ID,
    }
)
_ANALYTIC_PRIOR_SEED = 2026072701
_ANALYTIC_PRIOR_KINDS = frozenset(
    {
        "analytic-low-key-neutral-v1",
        "analytic-mid-key-neutral-v1",
        "analytic-wide-chroma-v1",
    }
)


@dataclass(frozen=True)
class CFSMProjectionPolicy:
    """Frozen numerical limits for the isolated v0 challenger."""

    prior_axis_size: int = 17
    lut_size: int = 17
    maximum_reference_samples: int = 65536
    projection_iterations: int = 28
    maximum_residual_amplitude: float = 0.25
    maximum_first_axis_step: float = 0.08
    maximum_second_axis_difference: float = 0.04
    maximum_neutral_axis_error: float = 0.04
    minimum_tetrahedron_jacobian_determinant: float = 0.05
    minimum_useful_strength: float = 1e-4


@dataclass(frozen=True)
class CFSMFitDiagnostics:
    """Evidence needed to distinguish a real candidate from identity fallback."""

    algorithm_id: str
    projected_strength: float
    used_identity_fallback: bool
    fallback_reason: str | None
    reference_sample_count: int
    canonical_prior_sample_count: int
    canonical_prior_mode: str
    source_image_count: int
    constraint_report: LUTConstraintReport


@dataclass(frozen=True)
class CFSMCandidate:
    """One fixed explicit operator derived from exactly one reference."""

    schema_id: str
    algorithm_id: str
    candidate_id: str
    policy: CFSMProjectionPolicy
    lut: DenseLUT3D
    diagnostics: CFSMFitDiagnostics


def _validate_policy(policy: CFSMProjectionPolicy) -> None:
    if not isinstance(policy, CFSMProjectionPolicy):
        raise ReferenceMatchContractError(
            "CFSM policy must be CFSMProjectionPolicy"
        )
    integer_values = (
        ("prior_axis_size", policy.prior_axis_size, 5, 65),
        ("lut_size", policy.lut_size, 5, 65),
        ("maximum_reference_samples", policy.maximum_reference_samples, 64, 1_000_000),
        ("projection_iterations", policy.projection_iterations, 1, 64),
    )
    for label, value, minimum, maximum in integer_values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > maximum
        ):
            raise ReferenceMatchContractError(
                f"CFSM {label} must be an integer within [{minimum}, {maximum}]"
            )
    limits = (
        policy.maximum_residual_amplitude,
        policy.maximum_first_axis_step,
        policy.maximum_second_axis_difference,
        policy.maximum_neutral_axis_error,
        policy.minimum_tetrahedron_jacobian_determinant,
        policy.minimum_useful_strength,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or float(value) <= 0.0
        for value in limits
    ):
        raise ReferenceMatchContractError(
            "CFSM projection limits must be finite and positive"
        )
    if policy.minimum_tetrahedron_jacobian_determinant > 1.0:
        raise ReferenceMatchContractError(
            "CFSM minimum Jacobian determinant cannot exceed identity"
        )
    if policy.minimum_useful_strength >= 1.0:
        raise ReferenceMatchContractError(
            "CFSM minimum useful strength must be below one"
        )


def _validate_image(image: WorkingImage, label: str) -> None:
    if not isinstance(image, WorkingImage):
        raise ReferenceMatchContractError(f"{label} must be WorkingImage")
    if (
        image.working_space != "linear_srgb"
        or image.transfer_state != "display_linear"
    ):
        raise ReferenceMatchContractError(
            f"{label} must be display-linear linear_srgb for CFSM v0"
        )
    pixels = np.asarray(image.pixels)
    if (
        pixels.ndim != 3
        or pixels.shape[-1] != 3
        or not np.isfinite(pixels).all()
        or float(np.min(pixels)) < 0.0
        or float(np.max(pixels)) > 1.0
    ):
        raise ReferenceMatchContractError(
            f"{label} pixels must be finite HxWx3 within [0, 1]"
        )


def _cube_grid(axis_size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, axis_size, dtype=np.float64)
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"),
        axis=-1,
    )


def _analytic_photographic_prior(kind: str, sample_count: int) -> np.ndarray:
    """Generate one versioned natural-image-like RGB moment prior."""

    if kind not in _ANALYTIC_PRIOR_KINDS:
        raise ReferenceMatchContractError(
            f"unsupported CFSM analytic prior: {kind}"
        )
    digest = hashlib.sha256(
        f"{_ANALYTIC_PRIOR_SEED}:{kind}".encode("ascii")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    if kind == "analytic-low-key-neutral-v1":
        alpha, beta, chroma = 1.35, 3.8, 0.13
    elif kind == "analytic-mid-key-neutral-v1":
        alpha, beta, chroma = 2.1, 2.8, 0.15
    else:
        alpha, beta, chroma = 1.55, 2.25, 0.24
    luminance = 0.01 + 0.98 * rng.beta(alpha, beta, sample_count)
    opponent_a = rng.normal(0.0, 1.0, sample_count)
    opponent_b = rng.normal(0.0, 1.0, sample_count)
    scale = chroma * (
        0.25 + 0.75 * 4.0 * luminance * (1.0 - luminance)
    )
    rgb = np.column_stack(
        (
            luminance + scale * (0.55 * opponent_a + 0.25 * opponent_b),
            luminance + scale * (-0.35 * opponent_a + 0.15 * opponent_b),
            luminance + scale * (-0.20 * opponent_a - 0.40 * opponent_b),
        )
    )
    return np.clip(rgb, 0.0, 1.0)


def _reference_samples(
    reference: WorkingImage,
    maximum_samples: int,
) -> np.ndarray:
    flattened = np.asarray(reference.pixels, dtype=np.float64).reshape(-1, 3)
    if len(flattened) <= maximum_samples:
        return flattened
    indices = np.linspace(
        0,
        len(flattened) - 1,
        maximum_samples,
        dtype=np.int64,
    )
    return flattened[indices]


def _boundary_envelope(grid: np.ndarray) -> np.ndarray:
    """Smoothly pin every cube face to identity without hard clipping."""

    channel_weights = 4.0 * grid * (1.0 - grid)
    return np.prod(channel_weights, axis=-1, keepdims=True)


def _constraint_spec(policy: CFSMProjectionPolicy) -> LUTConstraintSpec:
    return LUTConstraintSpec(
        output_minimum=0.0,
        output_maximum=1.0,
        maximum_residual_amplitude=policy.maximum_residual_amplitude,
        maximum_first_axis_step=policy.maximum_first_axis_step,
        maximum_second_axis_difference=policy.maximum_second_axis_difference,
        maximum_neutral_axis_error=policy.maximum_neutral_axis_error,
        minimum_tetrahedron_jacobian_determinant=(
            policy.minimum_tetrahedron_jacobian_determinant
        ),
    )


def _diagnostics_to_dict(diagnostics: CFSMFitDiagnostics) -> dict[str, object]:
    return {
        "algorithm_id": diagnostics.algorithm_id,
        "projected_strength": diagnostics.projected_strength,
        "used_identity_fallback": diagnostics.used_identity_fallback,
        "fallback_reason": diagnostics.fallback_reason,
        "reference_sample_count": diagnostics.reference_sample_count,
        "canonical_prior_sample_count": diagnostics.canonical_prior_sample_count,
        "canonical_prior_mode": diagnostics.canonical_prior_mode,
        "source_image_count": diagnostics.source_image_count,
        "constraint_report": diagnostics.constraint_report.to_dict(),
    }


def _candidate_payload(candidate: CFSMCandidate) -> dict[str, object]:
    return {
        "schema_id": candidate.schema_id,
        "algorithm_id": candidate.algorithm_id,
        "policy": asdict(candidate.policy),
        "lut": candidate.lut.to_dict(),
        "diagnostics": _diagnostics_to_dict(candidate.diagnostics),
    }


def compute_cfsm_candidate_id(candidate: CFSMCandidate) -> str:
    """Return the canonical identity of all executable candidate state."""

    return canonical_sha256(_candidate_payload(candidate))


def validate_cfsm_candidate(candidate: CFSMCandidate) -> None:
    """Fail closed on metadata, LUT, report or identity tampering."""

    if not isinstance(candidate, CFSMCandidate):
        raise ReferenceMatchContractError(
            "candidate must be CFSMCandidate"
        )
    if candidate.schema_id != CFSM_CANDIDATE_SCHEMA_ID:
        raise ReferenceMatchContractError("unsupported CFSM candidate schema")
    if (
        candidate.algorithm_id not in _SUPPORTED_ALGORITHM_IDS
        or candidate.diagnostics.algorithm_id != candidate.algorithm_id
    ):
        raise ReferenceMatchContractError("unsupported CFSM candidate algorithm")
    _validate_policy(candidate.policy)
    if not isinstance(candidate.lut, DenseLUT3D):
        raise ReferenceMatchContractError("CFSM candidate LUT is invalid")
    if (
        candidate.lut.size != candidate.policy.lut_size
        or candidate.lut.interpolation != "tetrahedral"
        or not np.array_equal(candidate.lut.domain_min, np.zeros(3))
        or not np.array_equal(candidate.lut.domain_max, np.ones(3))
    ):
        raise ReferenceMatchContractError(
            "CFSM candidate LUT metadata does not match its policy"
        )
    strength = candidate.diagnostics.projected_strength
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not np.isfinite(float(strength))
        or float(strength) < 0.0
        or float(strength) > 1.0
    ):
        raise ReferenceMatchContractError(
            "CFSM projected strength must be within [0, 1]"
        )
    if candidate.diagnostics.reference_sample_count < 64:
        raise ReferenceMatchContractError(
            "CFSM candidate has insufficient reference samples"
        )
    if candidate.algorithm_id == CFSM_ALGORITHM_ID and (
        candidate.diagnostics.canonical_prior_mode != "fixed-uniform-cube"
        or candidate.diagnostics.source_image_count != 0
        or candidate.diagnostics.canonical_prior_sample_count
        != candidate.policy.prior_axis_size**3
    ):
        raise ReferenceMatchContractError(
            "CFSM fixed canonical-prior diagnostics mismatch"
        )
    if candidate.algorithm_id == CFSM_BATCH_ALGORITHM_ID and (
        candidate.diagnostics.canonical_prior_mode
        != "uploaded-source-batch"
        or candidate.diagnostics.source_image_count < 1
        or candidate.diagnostics.canonical_prior_sample_count < 64
        or candidate.diagnostics.canonical_prior_sample_count
        > candidate.policy.maximum_reference_samples
    ):
        raise ReferenceMatchContractError(
            "CFSM batch canonical-prior diagnostics mismatch"
        )
    if candidate.algorithm_id == CFSM_ANALYTIC_ALGORITHM_ID and (
        candidate.diagnostics.canonical_prior_mode
        not in _ANALYTIC_PRIOR_KINDS
        or candidate.diagnostics.source_image_count != 0
        or candidate.diagnostics.canonical_prior_sample_count
        != candidate.policy.prior_axis_size**3
    ):
        raise ReferenceMatchContractError(
            "CFSM analytic canonical-prior diagnostics mismatch"
        )
    if candidate.algorithm_id == CFSM_EMPIRICAL_ALGORITHM_ID and (
        not candidate.diagnostics.canonical_prior_mode.startswith(
            "empirical-neutral-photo-v1:"
        )
        or len(candidate.diagnostics.canonical_prior_mode)
        != len("empirical-neutral-photo-v1:") + 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate.diagnostics.canonical_prior_mode[
                len("empirical-neutral-photo-v1:") :
            ]
        )
        or candidate.diagnostics.source_image_count < 4
        or candidate.diagnostics.canonical_prior_sample_count
        < candidate.diagnostics.source_image_count * 64
    ):
        raise ReferenceMatchContractError(
            "CFSM empirical canonical-prior diagnostics mismatch"
        )
    if candidate.algorithm_id == CFSM_QUANTILE_ALGORITHM_ID and (
        candidate.diagnostics.canonical_prior_mode
        != "fixed-uniform-cube-monotone-quantile-v1"
        or candidate.diagnostics.source_image_count != 0
        or candidate.diagnostics.canonical_prior_sample_count
        != candidate.policy.prior_axis_size**3
    ):
        raise ReferenceMatchContractError(
            "CFSM quantile canonical-prior diagnostics mismatch"
        )
    expected_fallback = float(strength) == 0.0
    if (
        candidate.diagnostics.used_identity_fallback != expected_fallback
        or candidate.diagnostics.fallback_reason
        != ("safe-projection-collapsed" if expected_fallback else None)
    ):
        raise ReferenceMatchContractError(
            "CFSM fallback diagnostics are inconsistent"
        )
    report = audit_lut_constraints(
        candidate.lut,
        _constraint_spec(candidate.policy),
    )
    if not report.passes or report != candidate.diagnostics.constraint_report:
        raise ReferenceMatchContractError(
            "CFSM candidate constraint report does not match its LUT"
        )
    if (
        len(candidate.candidate_id) != 64
        or any(char not in "0123456789abcdef" for char in candidate.candidate_id)
        or candidate.candidate_id != compute_cfsm_candidate_id(candidate)
    ):
        raise ReferenceMatchContractError(
            "CFSM candidate_id does not match canonical payload"
        )


def _project_lut(
    raw_values: np.ndarray,
    grid: np.ndarray,
    policy: CFSMProjectionPolicy,
) -> tuple[DenseLUT3D, float, LUTConstraintReport]:
    residual = (raw_values - grid) * _boundary_envelope(grid)
    spec = _constraint_spec(policy)

    def candidate(strength: float) -> tuple[DenseLUT3D, LUTConstraintReport]:
        lut = DenseLUT3D(
            values=grid + float(strength) * residual,
            domain_min=np.zeros(3, dtype=np.float64),
            domain_max=np.ones(3, dtype=np.float64),
            interpolation="tetrahedral",
        )
        return lut, audit_lut_constraints(lut, spec)

    identity, identity_report = candidate(0.0)
    if not identity_report.passes:
        raise ReferenceMatchContractError(
            "CFSM projection policy rejects the identity operator"
        )
    full, full_report = candidate(1.0)
    if full_report.passes:
        return full, 1.0, full_report

    lower = 0.0
    upper = 1.0
    best_lut = identity
    best_report = identity_report
    for _ in range(policy.projection_iterations):
        middle = (lower + upper) * 0.5
        trial_lut, trial_report = candidate(middle)
        if trial_report.passes:
            lower = middle
            best_lut = trial_lut
            best_report = trial_report
        else:
            upper = middle
    return best_lut, lower, best_report


def _fit_cfsm_from_transport(
    reference: WorkingImage,
    estimate: AffineColorOperator | AffineMonotoneSplineOperator,
    *,
    policy: CFSMProjectionPolicy,
    algorithm_id: str,
    prior_mode: str,
    source_image_count: int,
    canonical_prior_sample_count: int,
) -> CFSMCandidate:
    """Project one frozen affine estimate into a replayable safe LUT."""

    resolved = policy
    _validate_policy(resolved)
    _validate_image(reference, "reference")
    target = _reference_samples(reference, resolved.maximum_reference_samples)
    if len(target) < 64:
        raise ReferenceMatchContractError(
            "CFSM reference must contain at least 64 pixels"
        )
    if not isinstance(
        estimate,
        (AffineColorOperator, AffineMonotoneSplineOperator),
    ):
        raise ReferenceMatchContractError(
            "CFSM transport estimate type is unsupported"
        )
    lut_grid = _cube_grid(resolved.lut_size)
    raw_values = estimate.apply(lut_grid)
    lut, strength, report = _project_lut(raw_values, lut_grid, resolved)
    fallback = strength < resolved.minimum_useful_strength
    if fallback:
        identity_grid = _cube_grid(resolved.lut_size)
        lut = DenseLUT3D(
            identity_grid,
            np.zeros(3, dtype=np.float64),
            np.ones(3, dtype=np.float64),
            "tetrahedral",
        )
        report = audit_lut_constraints(lut, _constraint_spec(resolved))
        strength = 0.0
    diagnostics = CFSMFitDiagnostics(
        algorithm_id=algorithm_id,
        projected_strength=float(strength),
        used_identity_fallback=fallback,
        fallback_reason="safe-projection-collapsed" if fallback else None,
        reference_sample_count=len(target),
        canonical_prior_sample_count=canonical_prior_sample_count,
        canonical_prior_mode=prior_mode,
        source_image_count=source_image_count,
        constraint_report=report,
    )
    provisional = CFSMCandidate(
        schema_id=CFSM_CANDIDATE_SCHEMA_ID,
        algorithm_id=algorithm_id,
        candidate_id="0" * 64,
        policy=resolved,
        lut=lut,
        diagnostics=diagnostics,
    )
    candidate = CFSMCandidate(
        schema_id=provisional.schema_id,
        algorithm_id=provisional.algorithm_id,
        candidate_id=compute_cfsm_candidate_id(provisional),
        policy=provisional.policy,
        lut=provisional.lut,
        diagnostics=provisional.diagnostics,
    )
    validate_cfsm_candidate(candidate)
    return candidate


def _fit_cfsm_from_prior(
    reference: WorkingImage,
    prior: np.ndarray,
    *,
    policy: CFSMProjectionPolicy,
    algorithm_id: str,
    prior_mode: str,
    source_image_count: int,
) -> CFSMCandidate:
    """Fit shared CFSM machinery after a pixel-sample prior is frozen."""

    resolved = policy
    _validate_policy(resolved)
    _validate_image(reference, "reference")
    canonical_prior = np.asarray(prior, dtype=np.float64)
    if (
        canonical_prior.ndim != 2
        or canonical_prior.shape[1] != 3
        or len(canonical_prior) < 64
        or not np.isfinite(canonical_prior).all()
        or float(np.min(canonical_prior)) < 0.0
        or float(np.max(canonical_prior)) > 1.0
    ):
        raise ReferenceMatchContractError(
            "CFSM canonical prior must be finite Nx3 within [0, 1]"
        )
    target = _reference_samples(reference, resolved.maximum_reference_samples)
    try:
        estimate = estimate_gaussian_transport_operator(
            canonical_prior,
            [target],
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ReferenceMatchContractError(
            "CFSM could not estimate a finite orientation-preserving transport"
        ) from exc
    return _fit_cfsm_from_transport(
        reference,
        estimate,
        policy=resolved,
        algorithm_id=algorithm_id,
        prior_mode=prior_mode,
        source_image_count=source_image_count,
        canonical_prior_sample_count=len(canonical_prior),
    )


def _symmetric_matrix_power(
    matrix: np.ndarray,
    power: float,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(eigenvalues.min()) <= 0.0:
        raise ReferenceMatchContractError(
            "CFSM empirical covariance must be positive definite"
        )
    return (eigenvectors * np.power(eigenvalues, power)) @ eigenvectors.T


def fit_cfsm_empirical_candidate(
    reference: WorkingImage,
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    prior_id: str,
    source_image_count: int,
    source_pixel_count: int,
    policy: CFSMProjectionPolicy | None = None,
) -> CFSMCandidate:
    """Fit CFSM from a frozen research-only neutral-photo moment artifact."""

    resolved = policy or CFSMProjectionPolicy()
    _validate_policy(resolved)
    _validate_image(reference, "reference")
    if (
        not isinstance(prior_id, str)
        or len(prior_id) != 64
        or any(character not in "0123456789abcdef" for character in prior_id)
    ):
        raise ReferenceMatchContractError(
            "CFSM empirical prior_id must be canonical SHA-256"
        )
    if (
        isinstance(source_image_count, bool)
        or not isinstance(source_image_count, int)
        or source_image_count < 4
        or isinstance(source_pixel_count, bool)
        or not isinstance(source_pixel_count, int)
        or source_pixel_count < source_image_count * 64
    ):
        raise ReferenceMatchContractError(
            "CFSM empirical prior source counts are invalid"
        )
    mean = np.asarray(prior_mean, dtype=np.float64)
    covariance = np.asarray(prior_covariance, dtype=np.float64)
    if (
        mean.shape != (3,)
        or covariance.shape != (3, 3)
        or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
        or np.any(mean < 0.0)
        or np.any(mean > 1.0)
        or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0)
        or np.any(np.diag(covariance) > 0.250000000001)
    ):
        raise ReferenceMatchContractError(
            "CFSM empirical prior moments are invalid"
        )
    target = _reference_samples(reference, resolved.maximum_reference_samples)
    target_mean = target.mean(axis=0)
    regularization = 1e-6
    source_cov = covariance + regularization * np.eye(3)
    target_cov = np.cov(target, rowvar=False) + regularization * np.eye(3)
    source_sqrt = _symmetric_matrix_power(source_cov, 0.5)
    source_inv_sqrt = _symmetric_matrix_power(source_cov, -0.5)
    middle = source_sqrt @ target_cov @ source_sqrt
    matrix = (
        source_inv_sqrt
        @ _symmetric_matrix_power(middle, 0.5)
        @ source_inv_sqrt
    )
    matrix = 0.5 * (matrix + matrix.T)
    estimate = AffineColorOperator(
        matrix=matrix,
        bias=target_mean - matrix @ mean,
    )
    return _fit_cfsm_from_transport(
        reference,
        estimate,
        policy=resolved,
        algorithm_id=CFSM_EMPIRICAL_ALGORITHM_ID,
        prior_mode=f"empirical-neutral-photo-v1:{prior_id}",
        source_image_count=source_image_count,
        canonical_prior_sample_count=source_pixel_count,
    )


def fit_cfsm_quantile_candidate(
    reference: WorkingImage,
    *,
    policy: CFSMProjectionPolicy | None = None,
) -> CFSMCandidate:
    """Fit a fixed affine-plus-monotone-quantile explicit operator."""

    resolved = policy or CFSMProjectionPolicy()
    _validate_policy(resolved)
    _validate_image(reference, "reference")
    prior = _cube_grid(resolved.prior_axis_size).reshape(-1, 3)
    target = _reference_samples(reference, resolved.maximum_reference_samples)
    try:
        estimate = estimate_affine_spline_transport_operator(
            prior,
            [target],
            knot_quantiles=(
                0.001,
                0.03,
                0.12,
                0.35,
                0.65,
                0.88,
                0.97,
                0.999,
            ),
            iterations=8,
            regularization=1e-6,
            normalize_frame_photometric=False,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ReferenceMatchContractError(
            "CFSM could not estimate a finite monotone-quantile transport"
        ) from exc
    return _fit_cfsm_from_transport(
        reference,
        estimate,
        policy=resolved,
        algorithm_id=CFSM_QUANTILE_ALGORITHM_ID,
        prior_mode="fixed-uniform-cube-monotone-quantile-v1",
        source_image_count=0,
        canonical_prior_sample_count=len(prior),
    )


def fit_cfsm_candidate(
    reference: WorkingImage,
    *,
    policy: CFSMProjectionPolicy | None = None,
) -> CFSMCandidate:
    """Fit the v0 candidate against an image-independent uniform cube."""

    resolved = policy or CFSMProjectionPolicy()
    _validate_policy(resolved)
    _validate_image(reference, "reference")
    prior = _cube_grid(resolved.prior_axis_size).reshape(-1, 3)
    return _fit_cfsm_from_prior(
        reference,
        prior,
        policy=resolved,
        algorithm_id=CFSM_ALGORITHM_ID,
        prior_mode="fixed-uniform-cube",
        source_image_count=0,
    )


def fit_cfsm_analytic_candidate(
    reference: WorkingImage,
    *,
    prior_kind: str,
    policy: CFSMProjectionPolicy | None = None,
) -> CFSMCandidate:
    """Fit one pre-registered analytic photographic-prior challenger."""

    resolved = policy or CFSMProjectionPolicy()
    _validate_policy(resolved)
    _validate_image(reference, "reference")
    sample_count = resolved.prior_axis_size**3
    prior = _analytic_photographic_prior(prior_kind, sample_count)
    return _fit_cfsm_from_prior(
        reference,
        prior,
        policy=resolved,
        algorithm_id=CFSM_ANALYTIC_ALGORITHM_ID,
        prior_mode=prior_kind,
        source_image_count=0,
    )


def fit_cfsm_batch_candidate(
    reference: WorkingImage,
    sources: Iterable[WorkingImage],
    *,
    policy: CFSMProjectionPolicy | None = None,
) -> CFSMCandidate:
    """Fit one fixed LUT from the reference and the complete uploaded batch."""

    resolved = policy or CFSMProjectionPolicy()
    _validate_policy(resolved)
    _validate_image(reference, "reference")
    if isinstance(sources, (WorkingImage, np.ndarray, str, bytes)):
        raise ReferenceMatchContractError(
            "CFSM batch prior sources must be an iterable of WorkingImage"
        )
    sampled: list[tuple[str, np.ndarray]] = []
    source_count = 0
    try:
        for source in sources:
            if source_count >= 64:
                raise ReferenceMatchContractError(
                    "CFSM batch prior supports at most 64 source images"
                )
            _validate_image(source, f"source[{source_count}]")
            source_samples = _reference_samples(
                source,
                resolved.maximum_reference_samples,
            )
            sample_bytes = np.asarray(
                source_samples,
                dtype="<f4",
                order="C",
            ).tobytes(order="C")
            sampled.append(
                (hashlib.sha256(sample_bytes).hexdigest(), source_samples)
            )
            source_count += 1
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "CFSM batch prior sources must be an iterable of WorkingImage"
        ) from exc
    if source_count < 1:
        raise ReferenceMatchContractError(
            "CFSM batch prior requires at least one source image"
        )
    sampled.sort(key=lambda item: item[0])
    pooled = np.concatenate([item[1] for item in sampled], axis=0)
    if len(pooled) > resolved.maximum_reference_samples:
        indices = np.linspace(
            0,
            len(pooled) - 1,
            resolved.maximum_reference_samples,
            dtype=np.int64,
        )
        pooled = pooled[indices]
    return _fit_cfsm_from_prior(
        reference,
        pooled,
        policy=resolved,
        algorithm_id=CFSM_BATCH_ALGORITHM_ID,
        prior_mode="uploaded-source-batch",
        source_image_count=source_count,
    )


def cfsm_candidate_to_json(candidate: CFSMCandidate) -> str:
    """Serialize the research candidate without losing executable state."""

    validate_cfsm_candidate(candidate)
    payload = {
        **_candidate_payload(candidate),
        "candidate_id": candidate.candidate_id,
    }
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ) + "\n"


def cfsm_candidate_from_json(encoded: str) -> CFSMCandidate:
    """Parse and fully revalidate one serialized research candidate."""

    if not isinstance(encoded, str):
        raise ReferenceMatchContractError(
            "encoded CFSM candidate must be a string"
        )
    try:
        payload = json.loads(encoded)
        if not isinstance(payload, dict) or set(payload) != {
            "schema_id",
            "algorithm_id",
            "candidate_id",
            "policy",
            "lut",
            "diagnostics",
        }:
            raise ValueError("candidate keys mismatch")
        if set(payload["policy"]) != set(asdict(CFSMProjectionPolicy())):
            raise ValueError("candidate policy keys mismatch")
        if set(payload["lut"]) != {
            "schema",
            "interpolation",
            "domain_min",
            "domain_max",
            "values",
        }:
            raise ValueError("candidate LUT keys mismatch")
        diagnostics_payload = payload["diagnostics"]
        if set(diagnostics_payload) != {
            "algorithm_id",
            "projected_strength",
            "used_identity_fallback",
            "fallback_reason",
            "reference_sample_count",
            "canonical_prior_sample_count",
            "canonical_prior_mode",
            "source_image_count",
            "constraint_report",
        }:
            raise ValueError("candidate diagnostics keys mismatch")
        report_payload = dict(diagnostics_payload["constraint_report"])
        if set(report_payload) != {
            "output_minimum",
            "output_maximum",
            "maximum_residual_amplitude",
            "maximum_first_axis_step",
            "maximum_second_axis_difference",
            "maximum_neutral_axis_error",
            "minimum_tetrahedron_jacobian_determinant",
            "maximum_tetrahedron_jacobian_determinant",
            "output_range_pass",
            "residual_pass",
            "first_difference_pass",
            "second_difference_pass",
            "neutral_axis_pass",
            "jacobian_pass",
            "passes",
        }:
            raise ValueError("candidate constraint-report keys mismatch")
        claimed_passes = report_payload.pop("passes")
        report = LUTConstraintReport(**report_payload)
        if bool(claimed_passes) != report.passes:
            raise ValueError("constraint report pass state mismatch")
        diagnostics = CFSMFitDiagnostics(
            algorithm_id=diagnostics_payload["algorithm_id"],
            projected_strength=diagnostics_payload["projected_strength"],
            used_identity_fallback=diagnostics_payload[
                "used_identity_fallback"
            ],
            fallback_reason=diagnostics_payload["fallback_reason"],
            reference_sample_count=diagnostics_payload[
                "reference_sample_count"
            ],
            canonical_prior_sample_count=diagnostics_payload[
                "canonical_prior_sample_count"
            ],
            canonical_prior_mode=diagnostics_payload[
                "canonical_prior_mode"
            ],
            source_image_count=diagnostics_payload["source_image_count"],
            constraint_report=report,
        )
        candidate = CFSMCandidate(
            schema_id=payload["schema_id"],
            algorithm_id=payload["algorithm_id"],
            candidate_id=payload["candidate_id"],
            policy=CFSMProjectionPolicy(**payload["policy"]),
            lut=DenseLUT3D.from_dict(payload["lut"]),
            diagnostics=diagnostics,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "encoded CFSM candidate payload is invalid"
        ) from exc
    validate_cfsm_candidate(candidate)
    return candidate


def render_cfsm_candidate(
    candidate: CFSMCandidate,
    source: WorkingImage,
) -> WorkingImage:
    """Apply the same fitted explicit operator without source-context fitting."""

    validate_cfsm_candidate(candidate)
    _validate_image(source, "source")
    output_pixels = candidate.lut.apply(
        np.asarray(source.pixels, dtype=np.float64)
    )
    if (
        not np.isfinite(output_pixels).all()
        or float(np.min(output_pixels)) < 0.0
        or float(np.max(output_pixels)) > 1.0
    ):
        raise ReferenceMatchContractError(
            "CFSM output violates the projected display gamut"
        )
    return WorkingImage(
        pixels=np.asarray(output_pixels, dtype=np.float32),
        working_space=source.working_space,
        transfer_state=source.transfer_state,
        source_transfer_state=source.source_transfer_state,
        source_profile=source.source_profile,
        hdr_metadata=dict(source.hdr_metadata),
        orientation_applied=source.orientation_applied,
        alpha_policy=source.alpha_policy,
        bit_depth_in=source.bit_depth_in,
        source_path=source.source_path,
        warnings=list(source.warnings),
    )


def render_cfsm_batch(
    candidate: CFSMCandidate,
    sources: Iterable[WorkingImage],
) -> tuple[WorkingImage, ...]:
    """Apply one fixed CFSM operator to a non-empty ordered source batch."""

    if isinstance(sources, (WorkingImage, np.ndarray, str, bytes)):
        raise ReferenceMatchContractError(
            "CFSM sources must be an iterable of WorkingImage"
        )
    try:
        batch = tuple(sources)
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "CFSM sources must be an iterable of WorkingImage"
        ) from exc
    if not batch:
        raise ReferenceMatchContractError(
            "CFSM source batch must not be empty"
        )
    return tuple(render_cfsm_candidate(candidate, source) for source in batch)


__all__ = [
    "CFSM_ALGORITHM_ID",
    "CFSM_ANALYTIC_ALGORITHM_ID",
    "CFSM_BATCH_ALGORITHM_ID",
    "CFSM_CANDIDATE_SCHEMA_ID",
    "CFSMCandidate",
    "CFSMFitDiagnostics",
    "CFSMProjectionPolicy",
    "cfsm_candidate_from_json",
    "cfsm_candidate_to_json",
    "compute_cfsm_candidate_id",
    "fit_cfsm_analytic_candidate",
    "fit_cfsm_batch_candidate",
    "fit_cfsm_candidate",
    "render_cfsm_batch",
    "render_cfsm_candidate",
    "validate_cfsm_candidate",
]
