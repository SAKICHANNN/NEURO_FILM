from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_u4_5b_jpeg_scaled_preview_decode import U45BError, evaluate_runs

ROOT = Path(__file__).resolve().parents[1]


def _run(wall: float, peak: int) -> dict:
    rows = [
        {
            "style_id": style,
            "output_sha256": token,
            "rgb_rmse": 0.02,
            "rgb_absolute_error_p95": 0.04,
            "new_boundary_fraction": 0.0,
        }
        for style, token in (
            ("velvia_50", "a"),
            ("portra_400", "b"),
            ("ektar_100", "c"),
        )
    ]
    return {
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": peak,
        "source_width": 4032,
        "source_height": 6048,
        "decoded_width": 1008,
        "decoded_height": 1512,
        "preview_width": 816,
        "preview_height": 1224,
        "preview_pixels": 998784,
        "jpeg_scaled_decode": True,
        "preview_basis": "libjpeg scaled decode",
        "rows": rows,
        "output_hashes": {row["style_id"]: row["output_sha256"] for row in rows},
    }


def test_u4_5b_contract_keeps_existing_fidelity_and_one_gib_gates() -> None:
    config = json.loads(
        (ROOT / "configs/u4_5b_jpeg_scaled_preview_decode_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["render"]["runs"] == 2
    assert config["render"]["jpeg_scaled_decode"] is True
    assert config["gates"]["maximum_rgb_rmse_vs_downsampled_full_output"] == 0.03
    assert config["gates"]["maximum_process_tree_rss_bytes"] == 1_073_741_824


def test_u4_5b_evaluation_requires_all_quality_and_resource_gates() -> None:
    config = json.loads(
        (ROOT / "configs/u4_5b_jpeg_scaled_preview_decode_v1.json").read_text(
            encoding="utf-8"
        )
    )
    passing = evaluate_runs(config, [_run(4.0, 500_000_000), _run(4.2, 510_000_000)])
    assert passing["pass"] is True
    failing_runs = [_run(4.0, 500_000_000), _run(4.2, 1_500_000_000)]
    failing = evaluate_runs(config, failing_runs)
    assert failing["pass"] is False
    assert failing["gates"]["process_tree_rss"] is False


def test_u4_5b_evaluation_requires_two_runs() -> None:
    config = json.loads(
        (ROOT / "configs/u4_5b_jpeg_scaled_preview_decode_v1.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(U45BError, match="exactly two"):
        evaluate_runs(config, [_run(4.0, 500_000_000)])
