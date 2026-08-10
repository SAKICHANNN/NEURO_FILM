"""Canonical research-only profile for the experimental scanner chain."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .contracts import PhysicalDomain
from .scanner import ScannerProfile
from .scanner_glare import MultiscaleScannerGlareProfile, ScannerGlareComponent

SCHEMA = "neuro_film.generic_scanner_chain_profile.v1"
CLAIM_LEVEL = "generic-physical-inspired"
ARITHMETIC = "float64-reference"
STAGE_ORDER = (
    "spectral",
    "p6za-multiscale-glare",
    "dmax",
    "mtf",
    "noise",
)
GLARE_ALGORITHM = "channel-serial-block-fft"

_TOP_LEVEL_FIELDS = {
    "schema",
    "claim_level",
    "input_domain",
    "output_domain",
    "arithmetic",
    "stage_order",
    "pixel_pitch_um",
    "scanner_profile",
    "glare_profile",
    "glare_algorithm",
    "glare_kernel_size",
    "glare_row_chunk",
    "downstream_tile_rows",
    "product_runtime_allowed",
    "scanner_calibrated",
}
_SCANNER_FIELDS = {
    "profile_id",
    "illuminant_rgb",
    "spectral_matrix",
    "local_flare_fraction",
    "global_flare_fraction",
    "flare_sigma_um",
    "dmax_density_rgb",
    "mtf_sigma_um_rgb",
    "shot_noise_variance_scale",
    "read_noise_variance",
    "seed",
}
_GLARE_FIELDS = {"components", "flare_fraction", "truncate_sigma"}
_GLARE_COMPONENT_FIELDS = {"weight", "sigma_pixels"}


def _exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields must match v1 exactly")
    return value


def _float(value: Any, label: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite JSON float")
    return value


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a JSON integer")
    return value


def _float_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return tuple(_float(item, label) for item in value)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


@dataclass(frozen=True)
class ScannerChainProfile:
    scanner_profile: ScannerProfile
    glare_profile: MultiscaleScannerGlareProfile
    pixel_pitch_um: float = 1.0
    glare_kernel_size: int = 177
    glare_row_chunk: int = 512
    downstream_tile_rows: int = 512
    schema: str = SCHEMA
    claim_level: str = CLAIM_LEVEL
    input_domain: str = PhysicalDomain.TRANSMITTANCE.value
    output_domain: str = PhysicalDomain.SCAN_LINEAR.value
    arithmetic: str = ARITHMETIC
    stage_order: tuple[str, ...] = STAGE_ORDER
    glare_algorithm: str = GLARE_ALGORITHM
    product_runtime_allowed: bool = False
    scanner_calibrated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scanner_profile, ScannerProfile):
            raise TypeError("scanner_profile must be ScannerProfile")
        if not isinstance(self.glare_profile, MultiscaleScannerGlareProfile):
            raise TypeError("glare_profile must be MultiscaleScannerGlareProfile")
        fixed = (
            (self.schema, SCHEMA, "schema"),
            (self.claim_level, CLAIM_LEVEL, "claim level"),
            (self.input_domain, PhysicalDomain.TRANSMITTANCE.value, "input domain"),
            (self.output_domain, PhysicalDomain.SCAN_LINEAR.value, "output domain"),
            (self.arithmetic, ARITHMETIC, "arithmetic"),
            (tuple(self.stage_order), STAGE_ORDER, "stage order"),
            (self.glare_algorithm, GLARE_ALGORITHM, "glare algorithm"),
            (self.product_runtime_allowed, False, "product runtime authority"),
            (self.scanner_calibrated, False, "scanner calibration claim"),
        )
        for actual, expected, label in fixed:
            if actual != expected:
                raise ValueError(f"unsupported scanner chain {label}")
        if not math.isfinite(self.pixel_pitch_um) or self.pixel_pitch_um <= 0.0:
            raise ValueError("pixel_pitch_um must be finite and positive")
        for value, label in (
            (self.glare_kernel_size, "glare_kernel_size"),
            (self.glare_row_chunk, "glare_row_chunk"),
            (self.downstream_tile_rows, "downstream_tile_rows"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.glare_kernel_size % 2 == 0:
            raise ValueError("glare_kernel_size must be odd")

    def to_dict(self) -> dict[str, Any]:
        scanner = self.scanner_profile
        glare = self.glare_profile
        return {
            "schema": self.schema,
            "claim_level": self.claim_level,
            "input_domain": self.input_domain,
            "output_domain": self.output_domain,
            "arithmetic": self.arithmetic,
            "stage_order": list(self.stage_order),
            "pixel_pitch_um": self.pixel_pitch_um,
            "scanner_profile": {
                "profile_id": scanner.profile_id,
                "illuminant_rgb": list(scanner.illuminant_rgb),
                "spectral_matrix": [list(row) for row in scanner.spectral_matrix],
                "local_flare_fraction": scanner.local_flare_fraction,
                "global_flare_fraction": scanner.global_flare_fraction,
                "flare_sigma_um": scanner.flare_sigma_um,
                "dmax_density_rgb": None
                if scanner.dmax_density_rgb is None
                else list(scanner.dmax_density_rgb),
                "mtf_sigma_um_rgb": list(scanner.mtf_sigma_um_rgb),
                "shot_noise_variance_scale": scanner.shot_noise_variance_scale,
                "read_noise_variance": scanner.read_noise_variance,
                "seed": scanner.seed,
            },
            "glare_profile": {
                "components": [
                    {
                        "weight": component.weight,
                        "sigma_pixels": component.sigma_pixels,
                    }
                    for component in glare.components
                ],
                "flare_fraction": glare.flare_fraction,
                "truncate_sigma": glare.truncate_sigma,
            },
            "glare_algorithm": self.glare_algorithm,
            "glare_kernel_size": self.glare_kernel_size,
            "glare_row_chunk": self.glare_row_chunk,
            "downstream_tile_rows": self.downstream_tile_rows,
            "product_runtime_allowed": self.product_runtime_allowed,
            "scanner_calibrated": self.scanner_calibrated,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @property
    def profile_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScannerChainProfile:
        root = _exact_fields(payload, _TOP_LEVEL_FIELDS, "scanner chain profile")
        scanner = _exact_fields(
            root["scanner_profile"], _SCANNER_FIELDS, "scanner profile"
        )
        glare = _exact_fields(root["glare_profile"], _GLARE_FIELDS, "glare profile")
        glare_components = glare["components"]
        if not isinstance(glare_components, list) or not glare_components:
            raise ValueError("glare components must be a non-empty array")
        components = []
        for raw in glare_components:
            item = _exact_fields(raw, _GLARE_COMPONENT_FIELDS, "glare component")
            components.append(
                ScannerGlareComponent(
                    weight=_float(item["weight"], "glare component weight"),
                    sigma_pixels=_float(
                        item["sigma_pixels"], "glare component sigma_pixels"
                    ),
                )
            )
        matrix = scanner["spectral_matrix"]
        if not isinstance(matrix, list) or len(matrix) != 3:
            raise ValueError("spectral_matrix must be 3x3")
        dmax = scanner["dmax_density_rgb"]
        if dmax is not None:
            dmax = _float_vector(dmax, 3, "dmax_density_rgb")
        stage_order = root["stage_order"]
        if not isinstance(stage_order, list) or not all(
            isinstance(value, str) for value in stage_order
        ):
            raise ValueError("stage_order must be a string array")
        for label in ("product_runtime_allowed", "scanner_calibrated"):
            if not isinstance(root[label], bool):
                raise TypeError(f"{label} must be a JSON boolean")
        return cls(
            scanner_profile=ScannerProfile(
                profile_id=scanner["profile_id"],
                illuminant_rgb=_float_vector(
                    scanner["illuminant_rgb"], 3, "illuminant_rgb"
                ),
                spectral_matrix=tuple(
                    _float_vector(row, 3, "spectral_matrix row") for row in matrix
                ),
                local_flare_fraction=_float(
                    scanner["local_flare_fraction"], "local_flare_fraction"
                ),
                global_flare_fraction=_float(
                    scanner["global_flare_fraction"], "global_flare_fraction"
                ),
                flare_sigma_um=_float(scanner["flare_sigma_um"], "flare_sigma_um"),
                dmax_density_rgb=dmax,
                mtf_sigma_um_rgb=_float_vector(
                    scanner["mtf_sigma_um_rgb"], 3, "mtf_sigma_um_rgb"
                ),
                shot_noise_variance_scale=_float(
                    scanner["shot_noise_variance_scale"],
                    "shot_noise_variance_scale",
                ),
                read_noise_variance=_float(
                    scanner["read_noise_variance"], "read_noise_variance"
                ),
                seed=_int(scanner["seed"], "seed"),
            ),
            glare_profile=MultiscaleScannerGlareProfile(
                components=tuple(components),
                flare_fraction=_float(glare["flare_fraction"], "flare_fraction"),
                truncate_sigma=_float(glare["truncate_sigma"], "truncate_sigma"),
            ),
            pixel_pitch_um=_float(root["pixel_pitch_um"], "pixel_pitch_um"),
            glare_kernel_size=_int(root["glare_kernel_size"], "glare_kernel_size"),
            glare_row_chunk=_int(root["glare_row_chunk"], "glare_row_chunk"),
            downstream_tile_rows=_int(
                root["downstream_tile_rows"], "downstream_tile_rows"
            ),
            schema=root["schema"],
            claim_level=root["claim_level"],
            input_domain=root["input_domain"],
            output_domain=root["output_domain"],
            arithmetic=root["arithmetic"],
            stage_order=tuple(stage_order),
            glare_algorithm=root["glare_algorithm"],
            product_runtime_allowed=root["product_runtime_allowed"],
            scanner_calibrated=root["scanner_calibrated"],
        )

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> ScannerChainProfile:
        if not isinstance(raw, bytes):
            raise TypeError("serialized scanner profile must be bytes")
        try:
            text = raw.decode("utf-8", errors="strict")
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid scanner profile JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("scanner profile JSON root must be an object")
        return cls.from_dict(payload)


__all__ = [
    "ARITHMETIC",
    "CLAIM_LEVEL",
    "GLARE_ALGORITHM",
    "SCHEMA",
    "STAGE_ORDER",
    "ScannerChainProfile",
]
