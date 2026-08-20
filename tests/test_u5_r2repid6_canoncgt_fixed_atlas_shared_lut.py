from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.canoncgt_fixed_atlas_shared_lut import (
    aggregate_atlas_luts,
    make_lattice_atlas,
)


def test_lattice_atlas_is_deterministic_and_covers_all_nodes() -> None:
    first = make_lattice_atlas(
        atlas_id="rgb", size=224, bins=17, multiplier=1723
    )
    second = make_lattice_atlas(
        atlas_id="rgb", size=224, bins=17, multiplier=1723
    )
    assert first.shape == (224, 224, 3)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert len(np.unique(first.reshape(-1, 3), axis=0)) == 17**3


def test_channel_permutations_preserve_lattice_population() -> None:
    rgb = make_lattice_atlas(
        atlas_id="rgb", size=224, bins=17, multiplier=1723
    )
    gbr = make_lattice_atlas(
        atlas_id="gbr", size=224, bins=17, multiplier=1723
    )
    assert np.array_equal(rgb[..., [1, 2, 0]], gbr)


def test_aggregation_is_key_order_invariant_and_float32() -> None:
    base = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    forward, forward_metrics = aggregate_atlas_luts(
        {"rgb": base, "gbr": base + 1, "brg": base - 1}
    )
    reverse, reverse_metrics = aggregate_atlas_luts(
        {"brg": base - 1, "gbr": base + 1, "rgb": base}
    )
    assert forward.dtype == np.float32
    assert np.array_equal(forward, reverse)
    assert forward_metrics == reverse_metrics
    assert np.array_equal(forward, base)


def test_runner_requires_explicit_output_root_and_has_three_phases() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_u5_r2repid6_canoncgt_fixed_atlas_shared_lut.py"
    ).read_text(encoding="utf-8")
    assert 'choices=("build", "apply", "evaluate")' in source
    assert 'parser.add_argument("--output-root", type=Path, required=True)' in source
    assert "D:\\" not in source
    assert "P:\\" not in source


def test_build_path_does_not_call_gold_source_loader() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/eval/canoncgt_fixed_atlas_shared_lut.py"
    ).read_text(encoding="utf-8")
    build = source[source.index("def build_lut_bank") : source.index("def apply_lut_bank")]
    assert "_load_gold_samples" not in build
    assert '"application_source_file_reads": 0' in build
