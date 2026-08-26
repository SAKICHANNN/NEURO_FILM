from __future__ import annotations

import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from src.inference.render_contract import load_render_profile
from src.inference.three_stock_look import iter_three_stock_look_rgb_shared_context

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6d_three_stock_yield_lifetime_v1.json"


def test_contract_freezes_generator_lifetime_and_resource_gates() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["node_id"] == "U7.6D"
    assert payload["execution"]["run_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert payload["gates"]["minimum_peak_process_tree_rss_reduction_bytes"] == 134217728
    assert payload["gates"]["candidate_peak_process_tree_rss_ratio_max"] == 0.95


def test_prior_yield_is_collectable_before_next_render_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_render_profile(
        ROOT / "configs/render_profiles/safe_rich_v1.json", root=ROOT
    )
    source = np.full((3, 5, 3), 0.25, dtype=np.float32)
    output_refs: list[weakref.ReferenceType[np.ndarray]] = []

    def fake_render(*args, **kwargs) -> np.ndarray:
        if output_refs:
            assert output_refs[-1]() is None
        output = np.full(source.shape, len(output_refs) / 10.0, dtype=np.float32)
        output_refs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(
        "src.inference.three_stock_look.build_safe_lab_source_context",
        lambda value: object(),
    )
    monkeypatch.setattr(
        "src.inference.three_stock_look.render_resolved_safe_lab_rgb", fake_render
    )
    inputs = {style: {} for style in ("velvia_50", "portra_400", "ektar_100")}
    iterator = iter_three_stock_look_rgb_shared_context(
        source,
        profile=profile,
        look_amount=1.0,
        style_statistics=inputs,
        guardrails=inputs,
        seed=31,
        tile_size=2,
    )
    for expected_style in ("velvia_50", "portra_400", "ektar_100"):
        catalog_row, output = next(iterator)
        assert catalog_row["style_id"] == expected_style
        del output
    with pytest.raises(StopIteration):
        next(iterator)
    assert output_refs[-1]() is None
