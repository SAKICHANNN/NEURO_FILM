"""Opt-in native Standard renderer consuming a resolved P8AZ package."""

from __future__ import annotations

import ctypes
import hashlib
from typing import Any, Callable

import numpy as np

from src.color_engine.srgb_transfer import linear_srgb_to_encoded

from .bounded_ordered_pipeline import BoundedOrderedPipeline
from .native_abi_layouts import (
    NativeAdjacencyProfileV1,
    NativeGaussianProfileV1,
    NativePrintProfileV1,
    native_adjacency_profile_struct,
    native_gaussian_profile_struct,
    native_print_profile_struct,
)
from .native_adjacency_profile import (
    compile_native_adjacency_profile_payload,
)
from .native_ao6_base_profile import (
    NativeAo6BaseContextF32V1,
    NativeAo6BaseProfileF32V1,
    native_ao6_base_profile_struct,
)
from .native_ao6_context_profile import NativeAo6ContextStateF32V1
from .native_ao6_residual_profile import (
    NativeAo6ResidualProfileF32V1,
    native_ao6_residual_profile_struct,
)
from .native_gauge_profile import (
    NativeGaugeProfileF32V1,
    native_gauge_profile_struct,
)
from .native_profile import compile_native_domains_profile_payload
from .native_spatial_profile import (
    compile_native_gaussian_profile_payload,
)
from .native_standard_package import (
    ResolvedNativeStandardLibraries,
    native_standard_package_sha256,
    validate_native_standard_package,
    validate_package_profile_artifact,
)


OutputSink = Callable[[int, int, np.ndarray], None]


class NativeStandardRuntimeError(RuntimeError):
    """Raised when the opt-in native Standard runtime fails closed."""


class NativeStandardRuntime:
    """Loaded, hash-bound native Standard runtime with ordered row output."""

    def __init__(
        self,
        *,
        package: dict[str, Any],
        artifact: dict[str, Any],
        resolved: ResolvedNativeStandardLibraries,
    ) -> None:
        validate_native_standard_package(package)
        validate_package_profile_artifact(package, artifact)
        package_sha = native_standard_package_sha256(package)
        if resolved.package_sha256 != package_sha:
            raise ValueError("resolved native Standard package drift")
        self.package_sha256 = package_sha
        self.artifact_sha256 = _canonical_artifact_sha256(artifact)
        self.bundle_sha256 = artifact["bundle_sha256"]
        self.policy = dict(package["execution_policy"])

        self._domains_payload = compile_native_domains_profile_payload(
            artifact
        )
        self._spatial_payload = compile_native_gaussian_profile_payload(
            artifact
        )
        self._adjacency_payload = (
            compile_native_adjacency_profile_payload(artifact)
        )
        self._domains_profile = native_print_profile_struct(
            self._domains_payload
        )
        self._adjacency_profile = native_adjacency_profile_struct(
            self._adjacency_payload
        )
        component_payloads = artifact["component_payloads"]
        self._gauge_profile = native_gauge_profile_struct(
            component_payloads["neutral-axis-gauge"]
        )
        display_payload = component_payloads[
            "ao6-source-context-display-look"
        ]
        self._base_profile = native_ao6_base_profile_struct(
            display_payload
        )
        self._residual_profile = native_ao6_residual_profile_struct(
            display_payload
        )
        self._spatial_stages = {
            row["stage"]: row
            for row in self._spatial_payload["stages"]
        }
        self._halo = sum(
            int(row["maximum_radius"])
            for row in self._spatial_payload["stages"]
        )
        paths = resolved.paths
        self._domains = _load_domains(paths["domains"])
        self._gaussian = _load_gaussian(paths["gaussian"])
        self._adjacency = _load_adjacency(paths["adjacency"])
        self._gauge = _load_gauge(paths["gauge"])
        self._context = _load_context(paths["context"])
        self._display = _load_display(paths["display"])
        _validate_loaded_abi(
            self._domains,
            self._gaussian,
            self._adjacency,
            self._gauge,
        )

    def render_to_sink(
        self,
        scene_linear: np.ndarray,
        *,
        output_sink: OutputSink,
    ) -> dict[str, Any]:
        """Render float32 encoded RGB rows; the sink must stage atomically."""

        source = np.asarray(scene_linear)
        if (
            source.dtype != np.float32
            or source.ndim != 3
            or source.shape[-1] != 3
            or source.size == 0
            or not source.flags.c_contiguous
            or not np.all(np.isfinite(source))
            or np.any(source < 0.0)
            or np.any(source > 1.0)
        ):
            raise ValueError(
                "native Standard source must be C-contiguous finite "
                "float32 HxWx3 in [0,1]"
            )
        if not callable(output_sink):
            raise ValueError("native Standard output_sink must be callable")
        height, width, _ = source.shape
        input_sha = hashlib.sha256(source.tobytes()).hexdigest()
        previous_writeable = source.flags.writeable
        source.flags.writeable = False
        try:
            context = self._build_source_context(source)
            receipt = self._render_physical_and_display(
                source,
                context=context,
                output_sink=output_sink,
            )
            if hashlib.sha256(source.tobytes()).hexdigest() != input_sha:
                raise NativeStandardRuntimeError(
                    "native Standard source mutated during render"
                )
        finally:
            source.flags.writeable = previous_writeable
        core = {
            "schema": "neuro_film.native_standard_stream_receipt.v1",
            "package_sha256": self.package_sha256,
            "artifact_sha256": self.artifact_sha256,
            "bundle_sha256": self.bundle_sha256,
            "input": {
                "array_sha256": input_sha,
                "dtype": "float32",
                "shape": [height, width, 3],
                "domain": "scene-linear-relative-exposure",
                "working_space": "linear-srgb",
            },
            "output": {
                "array_sha256": receipt["output_sha256"],
                "dtype": "float32",
                "shape": [height, width, 3],
                "domain": "display-encoded-rgb",
                "quantized": False,
            },
            "execution": {
                **self.policy,
                "submitted_tiles": receipt["submitted_tiles"],
                "consumed_tiles": receipt["consumed_tiles"],
                "maximum_observed_in_flight": receipt[
                    "maximum_observed_in_flight"
                ],
            },
            "production_default_changed": False,
            "claim_ceiling": (
                "opt-in exact-package Windows x64 native Standard "
                "float32 row stream; sink staging only, no decoder, "
                "encoder, durable commit, mobile, calibration or "
                "production promotion"
            ),
        }
        return {
            **core,
            "receipt_sha256": hashlib.sha256(
                _canonical_bytes(core)
            ).hexdigest(),
        }

    def _build_source_context(
        self, source: np.ndarray
    ) -> NativeAo6BaseContextF32V1:
        state = NativeAo6ContextStateF32V1()
        if self._context.nf_ao6_context_f32_init_v1(
            ctypes.byref(state)
        ) != 0:
            raise NativeStandardRuntimeError("AO6 context init failed")
        tile_rows = int(self.policy["tile_rows"])
        for y0 in range(0, source.shape[0], tile_rows):
            y1 = min(source.shape[0], y0 + tile_rows)
            encoded = np.ascontiguousarray(
                linear_srgb_to_encoded(
                    source[y0:y1].astype(np.float64)
                ),
                dtype=np.float32,
            )
            scratch = np.empty_like(encoded)
            status = self._context.nf_ao6_context_f32_update_v2(
                ctypes.byref(self._base_profile),
                ctypes.byref(state),
                _pointer(encoded),
                encoded.shape[0] * encoded.shape[1],
                _pointer(scratch),
            )
            if status != 0:
                raise NativeStandardRuntimeError(
                    f"AO6 context update failed: {status}"
                )
        context = NativeAo6BaseContextF32V1()
        if self._context.nf_ao6_context_f32_finalize_v1(
            ctypes.byref(state), ctypes.byref(context)
        ) != 0:
            raise NativeStandardRuntimeError("AO6 context finalize failed")
        return context

    def _render_physical_and_display(
        self,
        source: np.ndarray,
        *,
        context: NativeAo6BaseContextF32V1,
        output_sink: OutputSink,
    ) -> dict[str, Any]:
        height, width, _ = source.shape
        digest = hashlib.sha256()
        consumed_rows = 0

        def transform(
            item: tuple[int, int, np.ndarray],
        ) -> tuple[int, int, np.ndarray]:
            y0, y1, rows = item
            gauged = self._apply_gauge(rows)
            encoded = np.ascontiguousarray(
                linear_srgb_to_encoded(gauged.astype(np.float64)),
                dtype=np.float32,
            )
            return y0, y1, self._apply_display(
                encoded, context=context
            )

        def consume(item: tuple[int, int, np.ndarray]) -> None:
            nonlocal consumed_rows
            y0, y1, rows = item
            if (
                y0 != consumed_rows
                or rows.shape != (y1 - y0, width, 3)
            ):
                raise NativeStandardRuntimeError(
                    "native Standard ordered output drift"
                )
            rows.flags.writeable = False
            digest.update(rows.tobytes())
            output_sink(y0, y1, rows)
            consumed_rows = y1

        pipeline = BoundedOrderedPipeline[
            tuple[int, int, np.ndarray],
            tuple[int, int, np.ndarray],
        ](
            worker=transform,
            consumer=consume,
            max_workers=int(self.policy["pipeline_workers"]),
            max_in_flight=int(self.policy["max_in_flight"]),
        )
        tile_rows = int(self.policy["tile_rows"])
        try:
            for y0 in range(0, height, tile_rows):
                y1 = min(height, y0 + tile_rows)
                source_y0 = max(0, y0 - self._halo)
                source_y1 = min(height, y1 + self._halo)
                rendered = self._render_physical_tile(
                    source[source_y0:source_y1]
                )
                core = np.ascontiguousarray(
                    rendered[
                        y0 - source_y0 : y1 - source_y0
                    ]
                )
                pipeline.submit(
                    (
                        y0,
                        y1,
                        np.array(
                            core,
                            dtype=np.float32,
                            order="C",
                            copy=True,
                        ),
                    )
                )
            stats = pipeline.finish()
        except BaseException:
            pipeline.abort()
            raise
        if consumed_rows != height:
            raise NativeStandardRuntimeError(
                "native Standard output incomplete"
            )
        return {
            "output_sha256": digest.hexdigest(),
            "submitted_tiles": stats.submitted,
            "consumed_tiles": stats.consumed,
            "maximum_observed_in_flight": stats.maximum_in_flight,
        }

    def _render_physical_tile(
        self, source: np.ndarray
    ) -> np.ndarray:
        height, width, _ = source.shape
        count = height * width

        def blur(values: np.ndarray, stage: str) -> np.ndarray:
            profile = native_gaussian_profile_struct(
                self._spatial_payload,
                self._spatial_stages[stage],
            )
            workspace = np.empty_like(values)
            output = np.empty_like(values)
            status = self._gaussian.nf_gaussian_f32_apply_v1(
                ctypes.byref(profile),
                _pointer(values),
                height,
                width,
                _pointer(workspace),
                workspace.size,
                _pointer(output),
            )
            if status != 0:
                raise NativeStandardRuntimeError(
                    f"native Gaussian {stage} failed: {status}"
                )
            return output

        forward = blur(source, "forward_scatter")
        density = np.empty_like(source)
        if self._domains.nf_physical_sensitometry_f32_apply_v1(
            ctypes.byref(self._domains_profile),
            _pointer(forward),
            count,
            _pointer(density),
        ) != 0:
            raise NativeStandardRuntimeError(
                "native sensitometry failed"
            )
        blurred_density = blur(density, "development_adjacency")
        adjacent = np.empty_like(source)
        if self._adjacency.nf_bounded_adjacency_f32_apply_v1(
            ctypes.byref(self._adjacency_profile),
            _pointer(density),
            _pointer(blurred_density),
            count,
            _pointer(adjacent),
        ) != 0:
            raise NativeStandardRuntimeError(
                "native adjacency failed"
            )
        diffused = blur(adjacent, "dye_diffusion")
        interpreted = np.empty_like(source)
        if self._domains.nf_physical_interpretation_f32_apply_v1(
            ctypes.byref(self._domains_profile),
            _pointer(diffused),
            count,
            _pointer(interpreted),
        ) != 0:
            raise NativeStandardRuntimeError(
                "native interpretation failed"
            )
        return blur(interpreted, "scanner_mtf")

    def _apply_gauge(self, values: np.ndarray) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        output = np.empty_like(source)
        status = self._gauge.nf_neutral_gauge_f32_apply_v1(
            ctypes.byref(self._gauge_profile),
            _pointer(source),
            source.shape[0] * source.shape[1],
            _pointer(output),
        )
        if status != 0:
            raise NativeStandardRuntimeError(
                f"native gauge failed: {status}"
            )
        return output

    def _apply_display(
        self,
        encoded: np.ndarray,
        *,
        context: NativeAo6BaseContextF32V1,
    ) -> np.ndarray:
        source = np.ascontiguousarray(encoded.reshape(-1, 3))
        scratch = np.empty_like(source)
        output = np.empty_like(source)
        status = self._display.nf_ao6_display_f32_apply_v4(
            ctypes.byref(self._base_profile),
            ctypes.byref(context),
            ctypes.byref(self._residual_profile),
            _pointer(source),
            source.shape[0],
            _pointer(scratch),
            _pointer(output),
        )
        if status != 0:
            raise NativeStandardRuntimeError(
                f"native AO6 display failed: {status}"
            )
        return output.reshape(encoded.shape)


def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _load_domains(path: Any) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    arguments = [
        ctypes.POINTER(NativePrintProfileV1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_physical_domains_f32_abi_version_v1.argtypes = []
    library.nf_physical_domains_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_physical_sensitometry_f32_apply_v1.argtypes = arguments
    library.nf_physical_sensitometry_f32_apply_v1.restype = ctypes.c_int
    library.nf_physical_interpretation_f32_apply_v1.argtypes = arguments
    library.nf_physical_interpretation_f32_apply_v1.restype = ctypes.c_int
    return library


def _load_gaussian(path: Any) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_gaussian_f32_abi_version_v1.argtypes = []
    library.nf_gaussian_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_gaussian_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeGaussianProfileV1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_gaussian_f32_apply_v1.restype = ctypes.c_int
    return library


def _load_adjacency(path: Any) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_bounded_adjacency_f32_abi_version_v1.argtypes = []
    library.nf_bounded_adjacency_f32_abi_version_v1.restype = (
        ctypes.c_uint32
    )
    library.nf_bounded_adjacency_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeAdjacencyProfileV1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_bounded_adjacency_f32_apply_v1.restype = ctypes.c_int
    return library


def _load_gauge(path: Any) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_neutral_gauge_f32_abi_version_v1.argtypes = []
    library.nf_neutral_gauge_f32_abi_version_v1.restype = ctypes.c_uint32
    library.nf_neutral_gauge_f32_apply_v1.argtypes = [
        ctypes.POINTER(NativeGaugeProfileF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_neutral_gauge_f32_apply_v1.restype = ctypes.c_int
    return library


def _load_context(path: Any) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_context_f32_init_v1.argtypes = [
        ctypes.POINTER(NativeAo6ContextStateF32V1)
    ]
    library.nf_ao6_context_f32_init_v1.restype = ctypes.c_int
    library.nf_ao6_context_f32_update_v2.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6ContextStateF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_context_f32_update_v2.restype = ctypes.c_int
    library.nf_ao6_context_f32_finalize_v1.argtypes = [
        ctypes.POINTER(NativeAo6ContextStateF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
    ]
    library.nf_ao6_context_f32_finalize_v1.restype = ctypes.c_int
    return library


def _load_display(path: Any) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.nf_ao6_display_f32_apply_v4.argtypes = [
        ctypes.POINTER(NativeAo6BaseProfileF32V1),
        ctypes.POINTER(NativeAo6BaseContextF32V1),
        ctypes.POINTER(NativeAo6ResidualProfileF32V1),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.nf_ao6_display_f32_apply_v4.restype = ctypes.c_int
    return library


def _validate_loaded_abi(
    domains: ctypes.CDLL,
    gaussian: ctypes.CDLL,
    adjacency: ctypes.CDLL,
    gauge: ctypes.CDLL,
) -> None:
    observed = (
        domains.nf_physical_domains_f32_abi_version_v1(),
        gaussian.nf_gaussian_f32_abi_version_v1(),
        adjacency.nf_bounded_adjacency_f32_abi_version_v1(),
        gauge.nf_neutral_gauge_f32_abi_version_v1(),
    )
    if observed != (1, 1, 1, 1):
        raise NativeStandardRuntimeError(
            f"native Standard ABI drift: {observed}"
        )


def _canonical_bytes(value: Any) -> bytes:
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_artifact_sha256(artifact: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(artifact)).hexdigest()


__all__ = [
    "NativeStandardRuntime",
    "NativeStandardRuntimeError",
    "OutputSink",
]
