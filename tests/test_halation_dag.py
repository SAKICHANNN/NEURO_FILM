from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from src.filmfx import (
    DENSITY_FAMILY,
    PHYSICAL_COLOUR_FAMILY,
    REQUIRED_INTEGRATION_CAPABILITIES,
    HalationFieldNode,
    build_halation_resource_plan,
    validate_halation_nodes,
)


def _plan(family: str, shape=(1024, 1536), **kwargs):
    return build_halation_resource_plan(
        family,
        shape,
        tile_size=kwargs.pop("tile_size", 256),
        **kwargs,
    )


def _current_effect_call_counts(function_name: str) -> tuple[int, int]:
    source = Path("src/filmfx/effects.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    blur_count = 0
    percentile_count = 0
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "gaussian_filter_safe":
            blur_count += 1
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "percentile"
        ):
            percentile_count += 1
    return blur_count, percentile_count


def test_default_physical_inventory_matches_current_effect() -> None:
    plan = _plan(PHYSICAL_COLOUR_FAMILY)
    assert (plan.blur_count, plan.percentile_count) == _current_effect_call_counts(
        "physical_halation_layer"
    )
    assert plan.blur_count == 8
    assert plan.percentile_count == 2
    assert plan.finite_halo_blurs == (
        "local_abs",
        "red_near",
        "red_mid",
        "green_near",
        "green_mid",
    )
    assert plan.global_grid_blurs == ("local_mean", "red_tail", "red_glare")
    assert plan.missing_capabilities == tuple(sorted(REQUIRED_INTEGRATION_CAPABILITIES))
    assert plan.integration_ready is False
    assert plan.unresolved_workspace_nodes == plan.global_grid_blurs
    assert plan.maximum_output_input_halo == 1


def test_default_density_inventory_matches_current_effect() -> None:
    plan = _plan(DENSITY_FAMILY)
    assert (plan.blur_count, plan.percentile_count) == _current_effect_call_counts(
        "density_halation_layer"
    )
    assert plan.blur_count == 6
    assert plan.percentile_count == 1
    assert plan.finite_halo_blurs == ("local_abs", "near", "mid")
    assert plan.global_grid_blurs == ("local_mean", "tail", "glare")
    assert plan.integration_ready is False
    assert plan.maximum_output_input_halo == 1


def test_capabilities_are_explicit_and_do_not_change_graph_pixels_contract() -> None:
    missing = _plan(PHYSICAL_COLOUR_FAMILY)
    ready = _plan(
        PHYSICAL_COLOUR_FAMILY,
        available_capabilities=REQUIRED_INTEGRATION_CAPABILITIES,
    )
    assert ready.integration_ready is True
    assert ready.missing_capabilities == ()
    assert ready.unresolved_workspace_nodes == ()
    assert ready.finite_halo_blurs == missing.finite_halo_blurs
    assert ready.global_grid_blurs == missing.global_grid_blurs
    assert ready.fingerprint != missing.fingerprint


def test_gradient_capability_alone_keeps_row_stage_gate_closed() -> None:
    plan = _plan(
        PHYSICAL_COLOUR_FAMILY,
        available_capabilities=("coordinate_exact_gradient_window",),
    )
    assert plan.missing_capabilities == ("row_chunked_global_stage_builder",)
    assert plan.integration_ready is False
    assert plan.unresolved_workspace_nodes == plan.global_grid_blurs


def test_diffusion_and_geometry_reclassify_deterministically() -> None:
    default = _plan(PHYSICAL_COLOUR_FAMILY)
    repeated = _plan(PHYSICAL_COLOUR_FAMILY)
    weak = _plan(PHYSICAL_COLOUR_FAMILY, local_diffusion=0.15)
    tiny = _plan(PHYSICAL_COLOUR_FAMILY, shape=(8, 9), tile_size=4)
    assert repeated == default
    assert weak.fingerprint != default.fingerprint
    assert weak.global_grid_blurs != default.global_grid_blurs
    assert next(node for node in weak.nodes if node.node_id == "green_near").halo == 1
    assert tiny.global_grid_blurs == ()
    assert tiny.fingerprint != default.fingerprint


@pytest.mark.parametrize("shape", [(4000, 6000), (10000, 10000)])
@pytest.mark.parametrize("family", [PHYSICAL_COLOUR_FAMILY, DENSITY_FAMILY])
def test_large_arithmetic_plans_keep_resource_categories_separate(shape, family) -> None:
    plan = _plan(family, shape=shape, tile_size=512)
    pixels = shape[0] * shape[1]
    assert plan.external_bytes == pixels * 3 * 4
    assert plan.required_output_bytes == pixels * 4 * 4
    assert plan.peak_context_bytes > 0
    assert plan.peak_scalar_bytes in {8, 16}
    assert plan.peak_workspace_bytes > 0
    assert plan.peak_planner_ram_bytes >= plan.peak_context_bytes
    assert plan.scratch_disk_bytes == 0
    assert plan.integration_ready is False


def test_source_linear_external_is_accounted_without_graph_ambiguity() -> None:
    derived = _plan(DENSITY_FAMILY, source_linear_provided=False)
    supplied = _plan(DENSITY_FAMILY, source_linear_provided=True)
    pixels = derived.source_shape[0] * derived.source_shape[1]
    assert supplied.external_bytes - derived.external_bytes == pixels * 3 * 4
    assert supplied.fingerprint != derived.fingerprint


def test_lifetimes_release_only_after_last_consumer_and_peaks_recompute() -> None:
    plan = _plan(PHYSICAL_COLOUR_FAMILY)
    by_id = {node.node_id: node for node in plan.nodes}
    positions = {node.node_id: index for index, node in enumerate(plan.nodes)}
    consumers = {node.node_id: [] for node in plan.nodes}
    for node in plan.nodes:
        for dependency in node.dependencies:
            consumers[dependency].append(node.node_id)

    cache = {}

    def expected_last_use(node_id):
        if node_id in cache:
            return cache[node_id]
        result = positions[node_id]
        for consumer_id in consumers[node_id]:
            consumer = by_id[consumer_id]
            result = max(
                result,
                expected_last_use(consumer_id)
                if consumer.storage == "stream"
                else positions[consumer_id],
            )
        cache[node_id] = result
        return result

    for lifetime in plan.lifetimes:
        assert lifetime.last_consumer_step == expected_last_use(lifetime.node_id)
        assert lifetime.release_step > lifetime.last_consumer_step or lifetime.storage in {
            "external",
            "output",
        }
        assert lifetime.allocation_bytes == by_id[lifetime.node_id].allocation_bytes

    recomputed_context = 0
    recomputed_scalar = 0
    recomputed_total = 0
    for step, node in enumerate(plan.nodes):
        context = sum(
            item.allocation_bytes
            for item in plan.lifetimes
            if item.storage == "context" and item.create_step <= step < item.release_step
        )
        scalar = sum(
            item.allocation_bytes
            for item in plan.lifetimes
            if item.storage == "scalar" and item.create_step <= step < item.release_step
        )
        recomputed_context = max(recomputed_context, context)
        recomputed_scalar = max(recomputed_scalar, scalar)
        recomputed_total = max(recomputed_total, context + scalar + node.workspace_bytes)
    assert plan.peak_context_bytes == recomputed_context
    assert plan.peak_scalar_bytes == recomputed_scalar
    assert plan.peak_planner_ram_bytes == recomputed_total

    lifetime_by_id = {item.node_id: item for item in plan.lifetimes}
    assert lifetime_by_id["local_mean"].last_consumer_step == positions["alpha"]
    assert lifetime_by_id["source_percentile"].last_consumer_step == positions["alpha"]
    assert lifetime_by_id["source_high_percentile"].last_consumer_step == positions["alpha"]


def test_graph_families_remain_distinct() -> None:
    physical = _plan(PHYSICAL_COLOUR_FAMILY)
    density = _plan(DENSITY_FAMILY)
    assert physical.fingerprint != density.fingerprint
    assert tuple(node.node_id for node in physical.nodes) != tuple(
        node.node_id for node in density.nodes
    )


def test_graph_validator_rejects_duplicate_unknown_cycle_and_use_before_produce() -> None:
    external = HalationFieldNode("source", "external", (), "external")
    stream = HalationFieldNode("derived", "pointwise_stream", ("source",), "stream")
    validate_halation_nodes((external, stream))
    with pytest.raises(ValueError, match="duplicate"):
        validate_halation_nodes((external, external))
    with pytest.raises(ValueError, match="unknown dependency"):
        validate_halation_nodes((replace(stream, dependencies=("missing",)),))
    cycle_a = HalationFieldNode("cycle_a", "pointwise_stream", ("cycle_b",), "stream")
    cycle_b = HalationFieldNode("cycle_b", "pointwise_stream", ("cycle_a",), "stream")
    with pytest.raises(ValueError, match="cycle"):
        validate_halation_nodes((cycle_a, cycle_b))
    with pytest.raises(ValueError, match="before production"):
        validate_halation_nodes((stream, external))


@pytest.mark.parametrize(
    "node",
    [
        HalationFieldNode("bad", "mystery", (), "stream"),
        HalationFieldNode("bad", "pointwise_stream", (), "mystery"),
        HalationFieldNode("bad", "external", (), "stream"),
        HalationFieldNode("bad", "output", (), "stream"),
        HalationFieldNode("bad", "exact_percentile", (), "stream", percentile=50.0),
        HalationFieldNode("bad", "pointwise_stream", (), "context"),
    ],
)
def test_graph_validator_rejects_invalid_vocabulary_and_storage(node) -> None:
    with pytest.raises(ValueError):
        validate_halation_nodes((node,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_shape": (0, 9)},
        {"source_shape": (1_000_001, 1)},
        {"tile_size": 0},
        {"local_diffusion": -0.1},
        {"local_diffusion": float("nan")},
        {"source_linear_provided": 1},
        {"available_capabilities": ("",)},
        {"available_capabilities": "row_chunked_global_stage_builder"},
    ],
)
def test_plan_rejects_invalid_contracts(kwargs) -> None:
    parameters = dict(
        family=PHYSICAL_COLOUR_FAMILY,
        source_shape=(17, 23),
        tile_size=8,
    )
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        build_halation_resource_plan(**parameters)


def test_plan_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_halation_resource_plan("creative", (17, 23), tile_size=8)
