from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.color_engine.safe_lab_rgb_context as context_module
from src.color_engine.safe_lab_rgb_context import (
    build_safe_lab_source_context,
    build_safe_lab_source_context_bounded,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_6g_three_stock_source_context_memory_v1.json"


def _rgb(height: int, width: int) -> np.ndarray:
    generator = np.random.default_rng(7601)
    return generator.random((height, width, 3), dtype=np.float32)


@pytest.mark.parametrize("row_chunk", [1, 7, 128, 1024])
def test_bounded_context_is_exact_for_uneven_rows(
    tmp_path: Path, row_chunk: int
) -> None:
    source = _rgb(131, 47)
    expected = build_safe_lab_source_context(source)
    actual = build_safe_lab_source_context_bounded(
        source,
        scratch_directory=tmp_path,
        row_chunk=row_chunk,
    )
    assert actual == expected
    assert list(tmp_path.iterdir()) == []


def test_bounded_context_rejects_bad_chunk_without_scratch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row_chunk"):
        build_safe_lab_source_context_bounded(
            _rgb(4, 5),
            scratch_directory=tmp_path,
            row_chunk=0,
        )
    assert list(tmp_path.iterdir()) == []


def test_bounded_context_removes_scratch_after_conversion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = context_module.rgb2lab
    calls = 0

    def fail_second(value: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected conversion failure")
        return original(value)

    monkeypatch.setattr(context_module, "rgb2lab", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        build_safe_lab_source_context_bounded(
            _rgb(9, 11),
            scratch_directory=tmp_path,
            row_chunk=4,
        )
    assert list(tmp_path.iterdir()) == []


def test_contract_freezes_full_product_resource_gates() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert contract["node_id"] == "U7.6G"
    assert contract["execution"]["lab_row_chunk"] == 128
    assert contract["execution"]["run_order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert contract["gates"]["source_context_cross_execution_exact"] is True
    assert (
        contract["gates"]["minimum_peak_process_tree_rss_reduction_bytes"]
        == 192 * 1024 * 1024
    )
    assert contract["gates"]["candidate_peak_process_tree_rss_ratio_max"] == 0.90
