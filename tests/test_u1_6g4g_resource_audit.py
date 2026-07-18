from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.audit_u1_6g4g_24mp_resources import (
    _sha_array,
    analytic_highlight_field,
    run_parent,
    screen_composite_rows,
)
from src.filmfx import FilmLayer, composite_layers


def test_analytic_source_is_chunk_invariant() -> None:
    first = analytic_highlight_field((73, 101, 3), row_chunk=7)
    second = analytic_highlight_field((73, 101, 3), row_chunk=19)
    assert first.dtype == np.float32
    assert first.tobytes() == second.tobytes()
    assert _sha_array(first) == _sha_array(second)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0


def test_row_screen_composite_matches_current_compositor_bytes() -> None:
    base = analytic_highlight_field((73, 101, 3), row_chunk=11)
    rng = np.random.default_rng(44)
    rgb = rng.uniform(0.0, 1.0, size=base.shape).astype(np.float32)
    alpha = rng.uniform(0.0, 0.26, size=(*base.shape[:2], 1)).astype(np.float32)
    layer = FilmLayer("test", "screen", rgb=rgb, alpha=alpha)
    expected = composite_layers(base, [layer])
    for row_chunk in (7, 19, 73):
        actual = screen_composite_rows(base, layer, row_chunk=row_chunk)
        assert actual.tobytes() == expected.tobytes()


def test_analytic_source_rejects_invalid_contract() -> None:
    for shape, chunk in (((1, 10, 3), 4), ((10, 10, 2), 4), ((10, 10, 3), 0)):
        try:
            analytic_highlight_field(shape, row_chunk=chunk)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid source contract must fail")


def test_small_parent_harness_repeats_and_cleans_injected_failure(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/u1_6g4g_24mp_resource_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        {
            "shape": [73, 101, 3],
            "input_generation_row_chunk": 7,
            "source_row_chunk": 11,
            "coarse_row_chunk": 3,
            "tile_size": 7,
            "composite_row_chunk": 9,
            "rss_sample_interval_seconds": 0.005,
            "worker_timeout_seconds": 30,
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report = run_parent(config_path, tmp_path / "workers")
    assert report["failure_probe"]["passed"]
    assert report["repeat_hashes_pass"]
    assert report["gate_result"]["automatic_pass"]
    assert not (tmp_path / "workers/failure_probe.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))
