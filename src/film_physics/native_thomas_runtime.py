"""Opt-in canonical-profile native Thomas RGB16 PNG runtime."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from typing import Any

from .atomic_native_output import (
    NativeByteSink,
    publish_native_thomas_profile_rgb16_png,
)
from .contracts import PhysicalDomainArray
from .native_gauge_profile import NativeGaugeProfileF32V1
from .native_granularity_amplitude import NativeGranularityAmplitudeProfileV1
from .native_thomas_field import NativeThomasFieldProfileV1
from .native_thomas_input import RelativeLayerLogExposure
from .native_thomas_package import (
    ResolvedNativeThomasPackage,
    native_thomas_package_sha256,
    validate_native_thomas_package,
)
from .native_thomas_spatial_chain import (
    NativeThomasSpatialChain,
    apply_native_thomas_spatial_chain,
)


class NativeThomasRuntimeError(RuntimeError):
    """Raised when the native Thomas export runtime fails closed."""


def _load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_thomas_rgb16_png_f32_abi_version_v1.argtypes = []
    library.nf_thomas_rgb16_png_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1.restype = (
        ctypes.c_int
    )
    library.nf_thomas_rgb16_png_cached_parallel_apply_v1.argtypes = [
        ctypes.POINTER(NativeGranularityAmplitudeProfileV1),
        ctypes.POINTER(NativeThomasFieldProfileV1),
        ctypes.POINTER(NativeGaugeProfileF32V1),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        NativeByteSink,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.nf_thomas_rgb16_png_cached_parallel_apply_v1.restype = ctypes.c_int
    if library.nf_thomas_rgb16_png_f32_abi_version_v1() != 1:
        raise NativeThomasRuntimeError("native Thomas export ABI mismatch")
    return library


class NativeThomasExportRuntime:
    """Resolved package runtime for one canonical physical-inspired profile."""

    def __init__(
        self,
        *,
        package: dict[str, Any],
        resolved: ResolvedNativeThomasPackage,
    ) -> None:
        validate_native_thomas_package(package)
        if native_thomas_package_sha256(package) != resolved.package_sha256:
            raise ValueError("resolved native Thomas package drift")
        self.package_sha256 = resolved.package_sha256
        self.profile = resolved.profile
        self.profile_sha256 = package["profile_asset"]["profile_sha256"]
        self.policy = dict(package["execution_policy"])
        self._library = _load_library(resolved.library_path)

    def publish(
        self, exposure: RelativeLayerLogExposure, *, destination: Path
    ) -> dict[str, Any]:
        """Publish one explicitly typed log-exposure render and return a receipt."""
        if not isinstance(exposure, RelativeLayerLogExposure):
            raise TypeError("native Thomas runtime requires RelativeLayerLogExposure")
        input_descriptor = exposure.descriptor()
        source = exposure.values_chw
        native_input_sha256 = hashlib.sha256(source.tobytes()).hexdigest()
        published = publish_native_thomas_profile_rgb16_png(
            self._library,
            self.profile,
            source,
            expected_profile_sha256=self.profile_sha256,
            row_partition=int(self.policy["row_partition"]),
            destination=destination,
            maximum_output_bytes=int(self.policy["maximum_output_bytes"]),
        )
        if hashlib.sha256(source.tobytes()).hexdigest() != native_input_sha256:
            raise NativeThomasRuntimeError("native Thomas runtime mutated input")
        if exposure.descriptor() != input_descriptor:
            raise NativeThomasRuntimeError("native Thomas runtime mutated typed input")
        return {
            "schema": "neuro_film.native_thomas_export_receipt.v1",
            "package_sha256": self.package_sha256,
            "profile_sha256": self.profile_sha256,
            "input_sha256": native_input_sha256,
            "input_contract": input_descriptor,
            "output": published,
            "production_default_changed": False,
            "claim_ceiling": (
                "opt-in exact-package generic physical-inspired RGB16 PNG; "
                "not calibrated stock response, arbitrary media or delivery"
            ),
        }

    def publish_layer_exposure(
        self, exposure: PhysicalDomainArray, *, destination: Path
    ) -> dict[str, Any]:
        """Publish explicit linear film-layer exposure through the package."""
        if not isinstance(exposure, PhysicalDomainArray):
            raise TypeError("native Thomas runtime requires PhysicalDomainArray")
        return self.publish(
            RelativeLayerLogExposure.from_layer_exposure(exposure),
            destination=destination,
        )

    def publish_spatial_layer_exposure(
        self,
        exposure: PhysicalDomainArray,
        spatial_chain: NativeThomasSpatialChain,
        *,
        destination: Path,
    ) -> dict[str, Any]:
        """Apply retained exposure-domain spatial physics before publication."""
        spatial = apply_native_thomas_spatial_chain(exposure, spatial_chain)
        return self.publish_layer_exposure(spatial, destination=destination)


__all__ = ["NativeThomasExportRuntime", "NativeThomasRuntimeError"]
