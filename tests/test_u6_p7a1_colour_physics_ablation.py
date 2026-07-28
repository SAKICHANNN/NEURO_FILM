from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_joint_ablation import (
    apply_physical_display,
    load_contracts,
    render_arms,
)
from src.film_physics import (
    apply_interpretation_bounded_development_adjacency,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p7a1_interpretation_bounded_ablation_v1.json"
)


def test_interpretation_bound_is_smooth_and_inside_references() -> None:
    correction = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _, runtime = load_contracts(ROOT, correction)
    black = runtime.print_operator.interpretation.black_reference_density
    white = runtime.print_operator.interpretation.white_reference_density
    x = np.linspace(0.0, 1.0, 97)
    density = black + x[:, None] * (white - black)
    density = np.broadcast_to(density[None, :, :], (31, 97, 3)).copy()
    output = apply_interpretation_bounded_development_adjacency(
        density,
        runtime.profile,
        maximum_absolute_transmittance_delta=0.008,
        maximum_absolute_density_delta=0.08,
        black_reference_density=black,
        white_reference_density=white,
    )
    assert np.all(output >= black - 1e-12)
    assert np.all(output <= white + 1e-12)
    assert np.array_equal(output[:, (0, -1)], density[:, (0, -1)])


def test_small_joint_ablation_is_bounded_and_deterministic() -> None:
    correction = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract, runtime = load_contracts(ROOT, correction)
    assert set(contract["arms"]) == {
        "colour_only",
        "physics_only",
        "combined",
        "cheap",
        "wrong_order",
    }
    y = np.linspace(0.02, 0.98, 43)[:, None]
    x = np.linspace(0.02, 0.98, 61)[None, :]
    source = np.empty((43, 61, 3), dtype=np.float64)
    source[..., 0] = 0.7 * x + 0.3 * y
    source[..., 1] = 0.2 * x + 0.8 * y
    source[..., 2] = 0.5 * x + 0.5 * y
    first = render_arms(source, runtime)
    second = render_arms(source, runtime)
    for name in contract["arms"]:
        assert np.array_equal(first[name], second[name])
        assert np.all(first[name] >= 0.0)
        assert np.all(first[name] <= 1.0)
    assert not np.array_equal(first["combined"], first["colour_only"])
    assert not np.array_equal(first["combined"], first["wrong_order"])


def test_endpoint_roundoff_does_not_block_full_white() -> None:
    correction = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _, runtime = load_contracts(ROOT, correction)
    output = apply_physical_display(
        np.ones((17, 19, 3), dtype=np.float64),
        runtime,
        spatial=True,
    )
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)


def test_physical_display_rejects_out_of_domain() -> None:
    correction = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _, runtime = load_contracts(ROOT, correction)
    values = np.full((9, 11, 3), 0.5, dtype=np.float64)
    values[0, 0, 0] = 1.01
    try:
        apply_physical_display(values, runtime, spatial=True)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-domain physical input was accepted")
