from __future__ import annotations

from scripts.audit_u7_6b_three_stock_file_batch import evaluate_rows


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
