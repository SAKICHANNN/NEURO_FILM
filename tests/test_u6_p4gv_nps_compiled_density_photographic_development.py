import json
from pathlib import Path

import numpy as np

from src.film_physics.density_lod_residual import apply_density_lod_residual

ROOT = Path(__file__).resolve().parents[1]


def test_p4gv_contract_binds_exact_nps_gains():
    c = json.loads(
        (
            ROOT
            / "configs/u6_p4gv_nps_compiled_density_photographic_development_v1.json"
        ).read_text()
    )
    assert (
        c["candidate"]["residual_projection"]
        == "nps-compiled-compound-poisson-density-multiplicative"
    )
    assert c["parents"]["nps_compiler"]["compiled_channel_gain"] == [
        0.7874638824806821,
        0.7008792161934756,
        0.6589818299954651,
    ]


def test_p4gv_gain_reduces_density_energy_and_preserves_ratios():
    base = np.full((7, 9, 3), (0.2, 0.4, 0.8), np.float32)
    ref = np.full_like(base, 0.5)
    physical = ref.copy()
    physical[3, 4] = (0.1, 0.2, 0.3)
    plain, _ = apply_density_lod_residual(base, physical, ref, finite_tail_density=0.12)
    gained, d = apply_density_lod_residual(
        base,
        physical,
        ref,
        finite_tail_density=0.09,
        channel_density_gain=(
            0.7874638824806821,
            0.7008792161934756,
            0.6589818299954651,
        ),
    )
    assert np.max(np.abs(gained - base)) < np.max(np.abs(plain - base))
    assert np.allclose(gained[..., 0] / gained[..., 1], 0.5)
    assert d["minimum_channel_density_gain"] == 0.6589818299954651
