"""Static field-DAG and resource planning for staged halation candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from .global_resample import (
    CHUNK_INVARIANT_GLOBAL_RESAMPLE_VERSION,
    build_chunk_invariant_global_stage_from_rows,
    plan_shape_stable_global_resample,
)
from .gradient_window import (
    GRADIENT_WINDOW_VERSION,
    coordinate_gradient_window,
)


HALATION_DAG_VERSION = "halation-field-dag-v1"
PHYSICAL_COLOUR_FAMILY = "physical-colour-v1"
DENSITY_FAMILY = "density-v1"
REQUIRED_INTEGRATION_CAPABILITIES = frozenset(
    {"row_chunked_global_stage_builder", "coordinate_exact_gradient_window"}
)
_NODE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_OPERATORS = frozenset(
    {
        "external",
        "pointwise_stream",
        "gradient",
        "exact_percentile",
        "finite_halo_blur",
        "global_grid_blur",
        "combine",
        "output",
    }
)
_STORAGE = frozenset({"external", "stream", "scalar", "context", "output"})
_OPERATOR_STORAGE = {
    "external": "external",
    "pointwise_stream": "stream",
    "gradient": "stream",
    "exact_percentile": "scalar",
    "finite_halo_blur": "stream",
    "global_grid_blur": "context",
    "combine": "stream",
    "output": "output",
}
_MAX_DIMENSION = 1_000_000
_FLOAT_BYTES = np.dtype(np.float32).itemsize
_PERCENTILE_HISTOGRAM_BYTES = 3 * 65536 * np.dtype(np.uint64).itemsize


@dataclass(frozen=True)
class HalationFieldNode:
    node_id: str
    operator: str
    dependencies: tuple[str, ...]
    storage: str
    channels: int = 1
    sigma: float | None = None
    percentile: float | None = None
    halo: int = 0
    input_halo: int = 0
    coarse_shape: tuple[int, int] | None = None
    allocation_bytes: int = 0
    workspace_bytes: int = 0


@dataclass(frozen=True)
class FieldLifetime:
    node_id: str
    storage: str
    allocation_bytes: int
    create_step: int
    last_consumer_step: int
    release_step: int


@dataclass(frozen=True)
class HalationResourcePlan:
    version: str
    family: str
    source_shape: tuple[int, int]
    tile_size: int
    local_diffusion: float
    source_linear_provided: bool
    nodes: tuple[HalationFieldNode, ...]
    lifetimes: tuple[FieldLifetime, ...]
    fingerprint: str
    blur_count: int
    percentile_count: int
    finite_halo_blurs: tuple[str, ...]
    global_grid_blurs: tuple[str, ...]
    external_bytes: int
    required_output_bytes: int
    peak_context_bytes: int
    peak_scalar_bytes: int
    peak_workspace_bytes: int
    peak_planner_ram_bytes: int
    scratch_disk_bytes: int
    maximum_output_input_halo: int
    unresolved_workspace_nodes: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    integration_ready: bool


@dataclass(frozen=True)
class HalationCapabilityBinding:
    capability: str
    implementation_version: str
    module: str
    symbol: str


_CAPABILITY_PROVIDERS = {
    "coordinate_exact_gradient_window": (
        GRADIENT_WINDOW_VERSION,
        "src.filmfx.gradient_window",
        "coordinate_gradient_window",
        coordinate_gradient_window,
    ),
    "row_chunked_global_stage_builder": (
        CHUNK_INVARIANT_GLOBAL_RESAMPLE_VERSION,
        "src.filmfx.global_resample",
        "build_chunk_invariant_global_stage_from_rows",
        build_chunk_invariant_global_stage_from_rows,
    ),
}


def validate_halation_capability_bindings(
    bindings: tuple[HalationCapabilityBinding, ...],
) -> tuple[HalationCapabilityBinding, ...]:
    """Verify exact provider identity for every required G3 capability."""

    if not isinstance(bindings, tuple) or any(
        not isinstance(binding, HalationCapabilityBinding) for binding in bindings
    ):
        raise ValueError("bindings must be a tuple of HalationCapabilityBinding records")
    names = tuple(binding.capability for binding in bindings)
    if len(set(names)) != len(names):
        raise ValueError("bindings contain duplicate capabilities")
    if frozenset(names) != REQUIRED_INTEGRATION_CAPABILITIES:
        raise ValueError("bindings must cover exactly the required integration capabilities")
    for binding in bindings:
        expected_version, expected_module, expected_symbol, provider = _CAPABILITY_PROVIDERS[
            binding.capability
        ]
        if binding != HalationCapabilityBinding(
            capability=binding.capability,
            implementation_version=expected_version,
            module=expected_module,
            symbol=expected_symbol,
        ):
            raise ValueError(f"binding metadata mismatch: {binding.capability}")
        if not callable(provider):
            raise ValueError(f"binding provider is not callable: {binding.capability}")
        if provider.__module__ != expected_module or provider.__name__ != expected_symbol:
            raise ValueError(f"binding provider identity mismatch: {binding.capability}")
    return bindings


def resolve_halation_integration_capability_bindings(
) -> tuple[HalationCapabilityBinding, ...]:
    """Resolve the passed G4A/G4C providers into G3 capability records."""

    bindings = tuple(
        HalationCapabilityBinding(
            capability=capability,
            implementation_version=provider[0],
            module=provider[1],
            symbol=provider[2],
        )
        for capability, provider in sorted(_CAPABILITY_PROVIDERS.items())
    )
    return validate_halation_capability_bindings(bindings)


def available_halation_integration_capabilities() -> frozenset[str]:
    """Return capability names only after their concrete providers validate."""

    return frozenset(
        binding.capability
        for binding in resolve_halation_integration_capability_bindings()
    )


def _shape2(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("source_shape must be an (H, W) tuple")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError("source_shape dimensions must be positive integers")
    if any(item > _MAX_DIMENSION for item in value):
        raise ValueError(f"source_shape dimensions must not exceed {_MAX_DIMENSION}")
    pixels = value[0] * value[1]
    if pixels > (2**63 - 1) // (4 * _FLOAT_BYTES):
        raise ValueError("source_shape byte arithmetic exceeds signed 64-bit range")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _diffusion(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("local_diffusion must be a finite number in [0, 8]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 8.0:
        raise ValueError("local_diffusion must be a finite number in [0, 8]")
    return 0.0 if result == 0.0 else result


def validate_halation_nodes(nodes: tuple[HalationFieldNode, ...]) -> tuple[HalationFieldNode, ...]:
    """Validate vocabulary, dependency identity, cycles and producer order."""

    if not isinstance(nodes, tuple) or not nodes:
        raise ValueError("nodes must be a non-empty tuple")
    by_id: dict[str, HalationFieldNode] = {}
    positions: dict[str, int] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, HalationFieldNode):
            raise ValueError("every graph node must be HalationFieldNode")
        if not _NODE_ID.fullmatch(node.node_id):
            raise ValueError(f"invalid node_id: {node.node_id!r}")
        if node.node_id in by_id:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        if node.operator not in _OPERATORS:
            raise ValueError(f"unknown operator for {node.node_id}: {node.operator}")
        if node.storage not in _STORAGE:
            raise ValueError(f"unknown storage for {node.node_id}: {node.storage}")
        if isinstance(node.channels, bool) or not isinstance(node.channels, int) or not 1 <= node.channels <= 3:
            raise ValueError(f"invalid channels for {node.node_id}")
        if not isinstance(node.dependencies, tuple) or any(
            not isinstance(dependency, str) for dependency in node.dependencies
        ):
            raise ValueError(f"dependencies for {node.node_id} must be strings")
        if len(set(node.dependencies)) != len(node.dependencies):
            raise ValueError(f"duplicate dependency for {node.node_id}")
        required_storage = _OPERATOR_STORAGE[node.operator]
        if node.storage != required_storage:
            raise ValueError(
                f"{node.operator} node {node.node_id} must use {required_storage} storage"
            )
        if node.operator == "exact_percentile" and node.percentile is None:
            raise ValueError(f"percentile node {node.node_id} must declare percentile")
        if node.operator == "global_grid_blur" and (
            node.sigma is None
            or not isinstance(node.coarse_shape, tuple)
            or len(node.coarse_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in node.coarse_shape
            )
        ):
            raise ValueError(f"global blur {node.node_id} must declare sigma and coarse_shape")
        if node.operator == "finite_halo_blur" and (node.sigma is None or node.halo <= 0):
            raise ValueError(f"finite blur {node.node_id} must declare sigma and positive halo")
        if node.sigma is not None and (not math.isfinite(node.sigma) or node.sigma < 0.0):
            raise ValueError(f"invalid sigma for {node.node_id}")
        if node.percentile is not None and (
            not math.isfinite(node.percentile) or not 0.0 <= node.percentile <= 100.0
        ):
            raise ValueError(f"invalid percentile for {node.node_id}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (node.halo, node.input_halo, node.allocation_bytes, node.workspace_bytes)
        ):
            raise ValueError(f"invalid resource field for {node.node_id}")
        by_id[node.node_id] = node
        positions[node.node_id] = index

    for node in nodes:
        for dependency in node.dependencies:
            if dependency not in by_id:
                raise ValueError(f"unknown dependency {dependency} for {node.node_id}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("graph contains a dependency cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in by_id[node_id].dependencies:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node in nodes:
        visit(node.node_id)
    for node in nodes:
        for dependency in node.dependencies:
            if positions[dependency] >= positions[node.node_id]:
                raise ValueError(f"dependency {dependency} is used before production by {node.node_id}")
    return nodes


def _raw_node(
    node_id: str,
    operator: str,
    dependencies: tuple[str, ...] = (),
    *,
    channels: int = 1,
    sigma: float | None = None,
    percentile: float | None = None,
) -> HalationFieldNode:
    storage = {
        "external": "external",
        "exact_percentile": "scalar",
        "output": "output",
    }.get(operator, "stream")
    return HalationFieldNode(
        node_id=node_id,
        operator=operator,
        dependencies=dependencies,
        storage=storage,
        channels=channels,
        sigma=sigma,
        percentile=percentile,
    )


def _physical_nodes(source_linear_provided: bool, local_diffusion: float) -> tuple[HalationFieldNode, ...]:
    diffusion = max(0.15, local_diffusion)
    background_sigma = max(4.0, 24.0 * local_diffusion)
    linear = _raw_node(
        "source_linear",
        "external" if source_linear_provided else "pointwise_stream",
        () if source_linear_provided else ("base_rgb",),
        channels=3,
    )
    return (
        _raw_node("base_rgb", "external", channels=3),
        linear,
        _raw_node("y", "pointwise_stream", ("source_linear",)),
        _raw_node("log_e", "pointwise_stream", ("y",)),
        _raw_node("source_raw", "pointwise_stream", ("log_e",)),
        _raw_node("source_percentile", "exact_percentile", ("source_raw",), percentile=99.7),
        _raw_node("source_normalized", "pointwise_stream", ("source_raw", "source_percentile")),
        _raw_node("maxc", "pointwise_stream", ("base_rgb",)),
        _raw_node("minc", "pointwise_stream", ("base_rgb",)),
        _raw_node("chroma", "pointwise_stream", ("maxc", "minc")),
        _raw_node("white_hot", "pointwise_stream", ("maxc", "chroma")),
        _raw_node("color_hot", "pointwise_stream", ("maxc", "chroma")),
        _raw_node("specular_confidence", "combine", ("white_hot", "color_hot")),
        _raw_node("edge", "gradient", ("y",)),
        _raw_node("edge_confidence", "pointwise_stream", ("edge",)),
        _raw_node(
            "source_weighted",
            "combine",
            ("source_normalized", "specular_confidence", "edge_confidence"),
        ),
        _raw_node("background_probe", "pointwise_stream", ("y",)),
        _raw_node("local_mean", "adaptive_blur", ("background_probe",), sigma=background_sigma),
        _raw_node("background_deviation", "combine", ("background_probe", "local_mean")),
        _raw_node(
            "local_abs",
            "adaptive_blur",
            ("background_deviation",),
            sigma=max(2.0, background_sigma * 0.35),
        ),
        _raw_node("dark_visibility", "pointwise_stream", ("local_mean",)),
        _raw_node("contrast_visibility", "combine", ("local_abs", "edge")),
        _raw_node("skin", "combine", ("base_rgb", "chroma")),
        _raw_node("visibility", "combine", ("dark_visibility", "contrast_visibility", "skin")),
        _raw_node("red_near", "adaptive_blur", ("source_weighted",), sigma=2.0 * diffusion),
        _raw_node("red_mid", "adaptive_blur", ("source_weighted",), sigma=8.0 * diffusion),
        _raw_node("red_tail", "adaptive_blur", ("source_weighted",), sigma=18.0 * diffusion),
        _raw_node(
            "red_glare",
            "adaptive_blur",
            ("source_weighted",),
            sigma=max(24.0, 52.0 * diffusion),
        ),
        _raw_node("red_exposure", "combine", ("red_near", "red_mid", "red_tail", "red_glare")),
        _raw_node("source_high_raw", "pointwise_stream", ("log_e",)),
        _raw_node(
            "source_high_percentile",
            "exact_percentile",
            ("source_high_raw",),
            percentile=99.8,
        ),
        _raw_node(
            "source_high_normalized",
            "pointwise_stream",
            ("source_high_raw", "source_high_percentile"),
        ),
        _raw_node(
            "source_high_weighted",
            "combine",
            ("source_high_normalized", "specular_confidence", "edge_confidence"),
        ),
        _raw_node("green_near", "adaptive_blur", ("source_high_weighted",), sigma=0.75 * diffusion),
        _raw_node("green_mid", "adaptive_blur", ("source_high_weighted",), sigma=3.0 * diffusion),
        _raw_node("green_exposure", "combine", ("green_near", "green_mid")),
        _raw_node("h_r", "combine", ("visibility", "red_exposure")),
        _raw_node("h_g", "combine", ("visibility", "green_exposure")),
        _raw_node("energy", "combine", ("h_r", "h_g")),
        _raw_node("color", "output", ("h_r", "h_g", "energy"), channels=3),
        _raw_node("alpha", "output", ("energy",)),
    )


def _density_nodes(source_linear_provided: bool, local_diffusion: float) -> tuple[HalationFieldNode, ...]:
    diffusion = max(0.15, local_diffusion)
    background_sigma = max(4.0, 28.0 * diffusion)
    linear = _raw_node(
        "source_linear",
        "external" if source_linear_provided else "pointwise_stream",
        () if source_linear_provided else ("base_rgb",),
        channels=3,
    )
    return (
        _raw_node("base_rgb", "external", channels=3),
        linear,
        _raw_node("y", "pointwise_stream", ("source_linear",)),
        _raw_node("log_e", "pointwise_stream", ("y",)),
        _raw_node("source_raw", "pointwise_stream", ("log_e",)),
        _raw_node("source_percentile", "exact_percentile", ("source_raw",), percentile=99.7),
        _raw_node("source_normalized", "pointwise_stream", ("source_raw", "source_percentile")),
        _raw_node("maxc", "pointwise_stream", ("base_rgb",)),
        _raw_node("minc", "pointwise_stream", ("base_rgb",)),
        _raw_node("chroma", "pointwise_stream", ("maxc", "minc")),
        _raw_node("white_hot", "pointwise_stream", ("maxc", "chroma")),
        _raw_node("color_hot", "pointwise_stream", ("maxc", "chroma")),
        _raw_node("specular_confidence", "combine", ("white_hot", "color_hot")),
        _raw_node("edge", "gradient", ("y",)),
        _raw_node("edge_confidence", "pointwise_stream", ("edge",)),
        _raw_node(
            "source_weighted",
            "combine",
            ("source_normalized", "specular_confidence", "edge_confidence"),
        ),
        _raw_node("background_probe", "pointwise_stream", ("y",)),
        _raw_node("local_mean", "adaptive_blur", ("background_probe",), sigma=background_sigma),
        _raw_node("background_deviation", "combine", ("background_probe", "local_mean")),
        _raw_node(
            "local_abs",
            "adaptive_blur",
            ("background_deviation",),
            sigma=max(2.0, background_sigma * 0.35),
        ),
        _raw_node("dark_visibility", "pointwise_stream", ("local_mean",)),
        _raw_node("contrast_visibility", "combine", ("local_abs", "edge")),
        _raw_node("skin", "combine", ("base_rgb", "chroma")),
        _raw_node("visibility", "combine", ("dark_visibility", "contrast_visibility", "skin")),
        _raw_node("near", "adaptive_blur", ("source_weighted",), sigma=2.4 * diffusion),
        _raw_node("mid", "adaptive_blur", ("source_weighted",), sigma=10.0 * diffusion),
        _raw_node("tail", "adaptive_blur", ("source_weighted",), sigma=26.0 * diffusion),
        _raw_node(
            "glare",
            "adaptive_blur",
            ("source_weighted",),
            sigma=max(32.0, 70.0 * diffusion),
        ),
        _raw_node("density_exposure", "combine", ("visibility", "near", "mid", "tail", "glare")),
        _raw_node("color", "output", ("base_rgb",), channels=3),
        _raw_node("alpha", "output", ("density_exposure",)),
    )


def _finite_blur_workspace_bytes(
    shape: tuple[int, int], tile_size: int, radius: int, input_halo: int
) -> int:
    support = radius + input_halo
    height = min(shape[0], tile_size + 2 * support)
    width = min(shape[1], tile_size + 2 * support)
    field = height * width * _FLOAT_BYTES
    padded_y = (height + 2 * radius) * width * _FLOAT_BYTES
    padded_x = height * (width + 2 * radius) * _FLOAT_BYTES
    return max(field + padded_y + field, field + padded_x + field)


def _materialize_nodes(
    raw_nodes: tuple[HalationFieldNode, ...],
    shape: tuple[int, int],
    tile_size: int,
) -> tuple[HalationFieldNode, ...]:
    pixels = shape[0] * shape[1]
    stream_halo: dict[str, int] = {}
    materialized: list[HalationFieldNode] = []
    for raw in raw_nodes:
        dependency_halo = max((stream_halo[dependency] for dependency in raw.dependencies), default=0)
        node = raw
        if raw.operator == "adaptive_blur":
            if raw.sigma is None:
                raise ValueError(f"adaptive blur {raw.node_id} has no sigma")
            grid = plan_shape_stable_global_resample(shape, raw.sigma)
            radius = 0 if raw.sigma <= 1e-6 else max(1, int(round(grid.truncate * raw.sigma)))
            if grid.coarse_shape == shape:
                node = replace(
                    raw,
                    operator="finite_halo_blur",
                    storage="stream",
                    halo=radius,
                    input_halo=dependency_halo,
                    workspace_bytes=_finite_blur_workspace_bytes(
                        shape, tile_size, radius, dependency_halo
                    ),
                )
            else:
                coarse_bytes = grid.coarse_shape[0] * grid.coarse_shape[1] * _FLOAT_BYTES
                node = replace(
                    raw,
                    operator="global_grid_blur",
                    storage="context",
                    halo=0,
                    input_halo=dependency_halo,
                    coarse_shape=grid.coarse_shape,
                    allocation_bytes=coarse_bytes,
                )
            stream_halo[raw.node_id] = 0
        elif raw.operator == "gradient":
            stream_halo[raw.node_id] = dependency_halo + 1
        elif raw.operator in {"external", "exact_percentile"}:
            stream_halo[raw.node_id] = 0
        elif raw.operator == "output":
            allocation = pixels * raw.channels * _FLOAT_BYTES
            node = replace(raw, allocation_bytes=allocation, input_halo=dependency_halo)
            stream_halo[raw.node_id] = dependency_halo
        else:
            stream_halo[raw.node_id] = dependency_halo

        if node.operator == "external":
            node = replace(node, allocation_bytes=pixels * node.channels * _FLOAT_BYTES)
        elif node.operator == "exact_percentile":
            node = replace(
                node,
                allocation_bytes=np.dtype(np.float64).itemsize,
                workspace_bytes=_PERCENTILE_HISTOGRAM_BYTES,
            )
        materialized.append(node)
    return validate_halation_nodes(tuple(materialized))


def _lifetimes(nodes: tuple[HalationFieldNode, ...]) -> tuple[FieldLifetime, ...]:
    positions = {node.node_id: index for index, node in enumerate(nodes)}
    by_id = {node.node_id: node for node in nodes}
    consumers: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    for index, node in enumerate(nodes):
        for dependency in node.dependencies:
            consumers[dependency].append(node.node_id)

    last_use_cache: dict[str, int] = {}

    def last_required_step(node_id: str) -> int:
        """Propagate through lazy streams until the next materialized consumer."""

        if node_id in last_use_cache:
            return last_use_cache[node_id]
        required = positions[node_id]
        for consumer_id in consumers[node_id]:
            consumer = by_id[consumer_id]
            if consumer.storage == "stream":
                required = max(required, last_required_step(consumer_id))
            else:
                required = max(required, positions[consumer_id])
        last_use_cache[node_id] = required
        return required

    terminal = len(nodes)
    result = []
    for node in nodes:
        last_consumer = last_required_step(node.node_id)
        if node.storage in {"external", "output"}:
            release = terminal
        else:
            release = last_consumer + 1
        result.append(
            FieldLifetime(
                node_id=node.node_id,
                storage=node.storage,
                allocation_bytes=node.allocation_bytes,
                create_step=index,
                last_consumer_step=last_consumer,
                release_step=release,
            )
        )
    return tuple(result)


def _resource_peaks(
    nodes: tuple[HalationFieldNode, ...], lifetimes: tuple[FieldLifetime, ...]
) -> tuple[int, int, int, int]:
    peak_context = 0
    peak_scalar = 0
    peak_workspace = 0
    peak_total = 0
    for step, node in enumerate(nodes):
        context = sum(
            lifetime.allocation_bytes
            for lifetime in lifetimes
            if lifetime.storage == "context"
            and lifetime.create_step <= step < lifetime.release_step
        )
        scalar = sum(
            lifetime.allocation_bytes
            for lifetime in lifetimes
            if lifetime.storage == "scalar"
            and lifetime.create_step <= step < lifetime.release_step
        )
        workspace = node.workspace_bytes
        peak_context = max(peak_context, context)
        peak_scalar = max(peak_scalar, scalar)
        peak_workspace = max(peak_workspace, workspace)
        peak_total = max(peak_total, context + scalar + workspace)
    return peak_context, peak_scalar, peak_workspace, peak_total


def _fingerprint_payload(
    family: str,
    shape: tuple[int, int],
    tile_size: int,
    local_diffusion: float,
    source_linear_provided: bool,
    nodes: tuple[HalationFieldNode, ...],
    missing: tuple[str, ...],
) -> str:
    payload = {
        "version": HALATION_DAG_VERSION,
        "family": family,
        "source_shape": shape,
        "tile_size": tile_size,
        "local_diffusion": local_diffusion,
        "source_linear_provided": source_linear_provided,
        "missing_capabilities": missing,
        "nodes": [asdict(node) for node in nodes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_halation_resource_plan(
    family: str,
    source_shape: tuple[int, int],
    *,
    tile_size: int,
    local_diffusion: float = 1.0,
    source_linear_provided: bool = False,
    available_capabilities: Iterable[str] = (),
) -> HalationResourcePlan:
    """Build one faithful static plan without executing an effect."""

    if family not in {PHYSICAL_COLOUR_FAMILY, DENSITY_FAMILY}:
        raise ValueError(f"unsupported halation family: {family}")
    shape = _shape2(source_shape)
    resolved_tile = _positive_integer(tile_size, "tile_size")
    resolved_diffusion = _diffusion(local_diffusion)
    if not isinstance(source_linear_provided, bool):
        raise ValueError("source_linear_provided must be boolean")
    if isinstance(available_capabilities, (str, bytes)):
        raise ValueError("available_capabilities must be an iterable of strings")
    try:
        capabilities = frozenset(available_capabilities)
    except TypeError as error:
        raise ValueError("available_capabilities must be an iterable of strings") from error
    if any(not isinstance(capability, str) or not capability for capability in capabilities):
        raise ValueError("available_capabilities must contain non-empty strings")
    missing = tuple(sorted(REQUIRED_INTEGRATION_CAPABILITIES - capabilities))

    if family == PHYSICAL_COLOUR_FAMILY:
        raw = _physical_nodes(source_linear_provided, resolved_diffusion)
    else:
        raw = _density_nodes(source_linear_provided, resolved_diffusion)
    nodes = _materialize_nodes(raw, shape, resolved_tile)
    lifetimes = _lifetimes(nodes)
    peak_context, peak_scalar, peak_workspace, peak_total = _resource_peaks(nodes, lifetimes)
    blur_nodes = tuple(
        node for node in nodes if node.operator in {"finite_halo_blur", "global_grid_blur"}
    )
    percentiles = tuple(node for node in nodes if node.operator == "exact_percentile")
    finite = tuple(node.node_id for node in blur_nodes if node.operator == "finite_halo_blur")
    global_grid = tuple(node.node_id for node in blur_nodes if node.operator == "global_grid_blur")
    external_bytes = sum(node.allocation_bytes for node in nodes if node.storage == "external")
    output_bytes = sum(node.allocation_bytes for node in nodes if node.storage == "output")
    unresolved = global_grid if "row_chunked_global_stage_builder" in missing else ()
    maximum_output_halo = max(
        (node.input_halo for node in nodes if node.storage == "output"), default=0
    )
    fingerprint = _fingerprint_payload(
        family,
        shape,
        resolved_tile,
        resolved_diffusion,
        source_linear_provided,
        nodes,
        missing,
    )
    return HalationResourcePlan(
        version=HALATION_DAG_VERSION,
        family=family,
        source_shape=shape,
        tile_size=resolved_tile,
        local_diffusion=resolved_diffusion,
        source_linear_provided=source_linear_provided,
        nodes=nodes,
        lifetimes=lifetimes,
        fingerprint=fingerprint,
        blur_count=len(blur_nodes),
        percentile_count=len(percentiles),
        finite_halo_blurs=finite,
        global_grid_blurs=global_grid,
        external_bytes=external_bytes,
        required_output_bytes=output_bytes,
        peak_context_bytes=peak_context,
        peak_scalar_bytes=peak_scalar,
        peak_workspace_bytes=peak_workspace,
        peak_planner_ram_bytes=peak_total,
        scratch_disk_bytes=0,
        maximum_output_input_halo=maximum_output_halo,
        unresolved_workspace_nodes=unresolved,
        missing_capabilities=missing,
        integration_ready=not missing,
    )
