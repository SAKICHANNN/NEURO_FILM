from __future__ import annotations

import importlib.util
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from src.inference.render_contract import load_render_profile
from src.inference.three_stock_look import iter_three_stock_look_rgb_shared_context

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6d_three_stock_yield_lifetime_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_6d_three_stock_yield_lifetime.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("u7_6d_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_evaluator_requires_exact_outputs_and_memory_reduction(tmp_path: Path) -> None:
    module = _audit_module()
    module.SCRATCH = tmp_path / "scratch"
    module.SCRATCH.mkdir()
    stock_ids = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    output_rows = [
        {
            "style_id": style,
            "output_sha256": f"output-{style}",
            "decoded_rgb16_sha256": f"decoded-{style}",
            "normalized_recipe_sha256": f"recipe-{style}",
        }
        for style in ("velvia_50", "portra_400", "ektar_100")
    ]
    rows = [
        {
            "mode": mode,
            "wall_seconds": 10.0 if mode == "baseline" else 9.9,
            "peak_process_tree_rss_bytes": 1_000_000_000
            if mode == "baseline"
            else 850_000_000,
            "rows": output_rows,
            "manifest_stock_ids": stock_ids,
        }
        for mode in ("baseline", "candidate", "candidate", "baseline")
    ]
    gates = json.loads(CONFIG.read_text(encoding="utf-8"))["gates"]
    result = module.evaluate_rows(rows, gates)
    assert result["decision"] == "PASS"
    assert result["peak_rss_reduction_bytes"] == 150_000_000

    rows[2]["rows"] = [{**output_rows[0], "output_sha256": "drift"}, *output_rows[1:]]
    assert module.evaluate_rows(rows, gates)["decision"] == "FAIL_CLOSED"
