"""Canonical profile ingress for the native Thomas RGB16 export core."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from .native_gauge_profile import (
    NativeGaugeProfileF32V1,
    native_gauge_profile_struct,
    validate_native_gauge_payload,
)
from .native_granularity_amplitude import (
    MAX_KNOTS,
    NativeGranularityAmplitudeProfileV1,
)
from .native_thomas_field import (
    NativeThomasFieldProfile,
    NativeThomasFieldProfileV1,
)

PROFILE_SCHEMA = "neuro_film.native_thomas_export_profile.v1"


def canonical_profile_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_hex(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid {field} SHA-256")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"invalid {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}")
    return result


def _amplitude_payload(
    profile: NativeGranularityAmplitudeProfileV1,
) -> dict[str, Any]:
    channels = []
    for channel in range(3):
        count = int(profile.knot_count[channel])
        if count < 2 or count > MAX_KNOTS:
            raise ValueError("native amplitude knot count is outside capacity")
        channels.append(
            {
                "log_exposure_knots": [
                    float(profile.log_exposure_knots[channel][index])
                    for index in range(count)
                ],
                "density_knots": [
                    float(profile.density_knots[channel][index])
                    for index in range(count)
                ],
                "floor_variance": float(profile.channel_floor_variance[channel]),
            }
        )
    source_sha = bytes(profile.source_profile_sha256).decode("ascii")
    return {
        "source_profile_sha256": source_sha,
        "channels": channels,
        "shared_amplitude": float(profile.shared_amplitude),
        "measurement_energy": float(profile.measurement_energy),
    }


def _field_payload(profile: NativeThomasFieldProfile) -> dict[str, Any]:
    return {
        "particle_sigma_pixels": float(profile.particle_sigma_pixels),
        "cluster_sigma_pixels": float(profile.cluster_sigma_pixels),
        "mean_offspring": float(profile.mean_offspring),
        "truncate": float(profile.truncate),
        "component_seeds": [int(value) for value in profile.component_seeds],
        "realization_seed": int(profile.realization_seed),
    }


def compile_native_thomas_export_profile(
    amplitude: NativeGranularityAmplitudeProfileV1,
    fields: Sequence[NativeThomasFieldProfile],
    gauge_payload: dict[str, Any],
    *,
    source_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Serialize exact ABI inputs without fitting or changing parameters."""

    if len(fields) != 3:
        raise ValueError("native Thomas export requires exactly three fields")
    validate_native_gauge_payload(gauge_payload)
    bindings = {
        str(name): _sha256_hex(value, field=f"binding {name}")
        for name, value in sorted(source_bindings.items())
    }
    if not bindings:
        raise ValueError("native Thomas export bindings must not be empty")
    body = {
        "schema": PROFILE_SCHEMA,
        "source_bindings": bindings,
        "amplitude": _amplitude_payload(amplitude),
        "fields": [_field_payload(profile) for profile in fields],
        "gauge": gauge_payload,
        "domain_order": [
            "relative-layer-log-exposure",
            "developed-density",
            "film-transmittance",
            "display-linear-neutral-gauge",
            "srgb-oetf",
            "uint16-quantization",
            "png-with-fixed-srgb-icc",
        ],
        "claim_level": "generic-physical-inspired",
    }
    return {
        **body,
        "profile_sha256": hashlib.sha256(canonical_profile_bytes(body)).hexdigest(),
    }


def validate_native_thomas_export_profile(payload: dict[str, Any]) -> str:
    expected = {
        "schema",
        "source_bindings",
        "amplitude",
        "fields",
        "gauge",
        "domain_order",
        "claim_level",
        "profile_sha256",
    }
    if set(payload) != expected or payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError("native Thomas export profile fields drift")
    if payload.get("claim_level") != "generic-physical-inspired":
        raise ValueError("native Thomas export claim level drift")
    if payload.get("domain_order") != [
        "relative-layer-log-exposure",
        "developed-density",
        "film-transmittance",
        "display-linear-neutral-gauge",
        "srgb-oetf",
        "uint16-quantization",
        "png-with-fixed-srgb-icc",
    ]:
        raise ValueError("native Thomas export domain order drift")
    bindings = payload["source_bindings"]
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("native Thomas export bindings missing")
    for name, value in bindings.items():
        if not isinstance(name, str) or not name:
            raise ValueError("native Thomas export binding name invalid")
        _sha256_hex(value, field=f"binding {name}")
    amplitude = payload["amplitude"]
    if not isinstance(amplitude, dict) or set(amplitude) != {
        "source_profile_sha256",
        "channels",
        "shared_amplitude",
        "measurement_energy",
    }:
        raise ValueError("native Thomas amplitude fields drift")
    _sha256_hex(amplitude["source_profile_sha256"], field="amplitude source")
    channels = amplitude["channels"]
    if not isinstance(channels, list) or len(channels) != 3:
        raise ValueError("native Thomas amplitude channel count drift")
    for row in channels:
        if not isinstance(row, dict) or set(row) != {
            "log_exposure_knots",
            "density_knots",
            "floor_variance",
        }:
            raise ValueError("native Thomas amplitude channel fields drift")
        x = row["log_exposure_knots"]
        y = row["density_knots"]
        if (
            not isinstance(x, list)
            or not isinstance(y, list)
            or len(x) != len(y)
            or len(x) < 3
            or len(x) > MAX_KNOTS
        ):
            raise ValueError("native Thomas amplitude knots invalid")
        x_values = [_finite(value, field="exposure knot") for value in x]
        y_values = [_finite(value, field="density knot") for value in y]
        if any(right <= left for left, right in pairwise(x_values)):
            raise ValueError("native Thomas exposure knots are not increasing")
        if any(right < left for left, right in pairwise(y_values)):
            raise ValueError("native Thomas density knots are decreasing")
        if _finite(row["floor_variance"], field="floor variance") <= 0.0:
            raise ValueError("native Thomas floor variance is not positive")
    if _finite(amplitude["shared_amplitude"], field="shared amplitude") <= 0.0:
        raise ValueError("native Thomas shared amplitude is not positive")
    measurement_energy = _finite(
        amplitude["measurement_energy"], field="measurement energy"
    )
    if measurement_energy <= 0.0 or measurement_energy > 1.0:
        raise ValueError("native Thomas measurement energy is not positive")
    fields = payload["fields"]
    if not isinstance(fields, list) or len(fields) != 3:
        raise ValueError("native Thomas field count drift")
    for row in fields:
        if not isinstance(row, dict) or set(row) != {
            "particle_sigma_pixels",
            "cluster_sigma_pixels",
            "mean_offspring",
            "truncate",
            "component_seeds",
            "realization_seed",
        }:
            raise ValueError("native Thomas field fields drift")
        for name in (
            "particle_sigma_pixels",
            "cluster_sigma_pixels",
            "mean_offspring",
            "truncate",
        ):
            if _finite(row[name], field=name) <= 0.0:
                raise ValueError(f"native Thomas {name} is not positive")
        seeds = row["component_seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in seeds
            )
            or any(value < 0 or value >= 2**64 for value in seeds)
            or isinstance(row["realization_seed"], bool)
            or not isinstance(row["realization_seed"], int)
            or row["realization_seed"] < 0
            or row["realization_seed"] >= 2**64
        ):
            raise ValueError("native Thomas seeds invalid")
    validate_native_gauge_payload(payload["gauge"])
    profile_sha = _sha256_hex(payload["profile_sha256"], field="profile")
    body = {key: value for key, value in payload.items() if key != "profile_sha256"}
    if hashlib.sha256(canonical_profile_bytes(body)).hexdigest() != profile_sha:
        raise ValueError("native Thomas export profile identity drift")
    return profile_sha


def reconstruct_native_thomas_export_profile(
    payload: dict[str, Any],
) -> tuple[
    NativeGranularityAmplitudeProfileV1,
    tuple[
        NativeThomasFieldProfileV1,
        NativeThomasFieldProfileV1,
        NativeThomasFieldProfileV1,
    ],
    NativeGaugeProfileF32V1,
]:
    validate_native_thomas_export_profile(payload)
    amplitude_payload = payload["amplitude"]
    amplitude = NativeGranularityAmplitudeProfileV1()
    amplitude.struct_size = ctypes.sizeof(NativeGranularityAmplitudeProfileV1)
    amplitude.abi_version = 1
    amplitude.source_profile_sha256 = amplitude_payload["source_profile_sha256"].encode(
        "ascii"
    )
    for channel, row in enumerate(amplitude_payload["channels"]):
        count = len(row["log_exposure_knots"])
        amplitude.knot_count[channel] = count
        for index in range(count):
            amplitude.log_exposure_knots[channel][index] = row["log_exposure_knots"][
                index
            ]
            amplitude.density_knots[channel][index] = row["density_knots"][index]
        amplitude.channel_floor_variance[channel] = row["floor_variance"]
    amplitude.shared_amplitude = amplitude_payload["shared_amplitude"]
    amplitude.measurement_energy = amplitude_payload["measurement_energy"]
    fields = tuple(
        NativeThomasFieldProfile(
            particle_sigma_pixels=row["particle_sigma_pixels"],
            cluster_sigma_pixels=row["cluster_sigma_pixels"],
            mean_offspring=row["mean_offspring"],
            truncate=row["truncate"],
            component_seeds=tuple(row["component_seeds"]),
            realization_seed=row["realization_seed"],
        ).as_abi()
        for row in payload["fields"]
    )
    return amplitude, fields, native_gauge_profile_struct(payload["gauge"])


__all__ = [
    "PROFILE_SCHEMA",
    "canonical_profile_bytes",
    "compile_native_thomas_export_profile",
    "reconstruct_native_thomas_export_profile",
    "validate_native_thomas_export_profile",
]
