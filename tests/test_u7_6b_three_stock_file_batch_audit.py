from __future__ import annotations

import json

from scripts.audit_u7_6b_three_stock_file_batch import (
    _normalized_recipe_sha256,
    evaluate_rows,
)


def _row(mode: str, wall: float, peak: int) -> dict:
    return {
        "mode": mode,
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": peak,
        "stderr": "",
        "rows": [
            {
                "style_id": style,
                "output_sha256": style,
                "decoded_rgb16_sha256": style,
                "normalized_recipe_sha256": style,
            }
            for style in ("velvia_50", "portra_400", "ektar_100")
        ],
    }


def test_file_batch_gate_evaluation_passes_exact_faster_candidate(tmp_path, monkeypatch) -> None:
    from scripts import audit_u7_6b_three_stock_file_batch as audit

    monkeypatch.setattr(audit, "SCRATCH", tmp_path / "scratch")
    rows = [
        _row("baseline", 100.0, 500),
        _row("candidate", 90.0, 510),
        _row("candidate", 91.0, 505),
        _row("baseline", 101.0, 500),
    ]
    result = evaluate_rows(
        rows,
        {
            "median_wall_ratio_max": 0.95,
            "candidate_peak_process_tree_rss_ratio_max": 1.05,
            "candidate_residue_count": 0,
        },
    )
    assert result["decision"] == "PASS"
    assert all(result["gate_results"].values())


def test_recipe_comparison_excludes_only_path_and_concurrent_commit(tmp_path) -> None:
    first = {
        "output": {"path": "A", "sha256": "pixels"},
        "software": {"commit": "a" * 40},
        "render": {"style": "velvia_50"},
    }
    second = json.loads(json.dumps(first))
    second["output"]["path"] = "B"
    second["software"]["commit"] = "b" * 40
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    assert _normalized_recipe_sha256(first_path) == _normalized_recipe_sha256(second_path)
    second["render"]["style"] = "portra_400"
    second_path.write_text(json.dumps(second), encoding="utf-8")
    assert _normalized_recipe_sha256(first_path) != _normalized_recipe_sha256(second_path)
