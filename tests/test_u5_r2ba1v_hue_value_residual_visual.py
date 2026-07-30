from __future__ import annotations

import json
from pathlib import Path

from src.eval.factorization_visual import build_visual_evidence


ROOT = Path(__file__).resolve().parents[1]


def test_visual_builder_is_deterministic_and_hides_mapping(
    tmp_path: Path,
) -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2ba1v_hue_value_residual_visual_v1.json"
        ).read_text(encoding="utf-8")
    )
    first = build_visual_evidence(
        root=ROOT, config=config, output_dir=tmp_path / "first"
    )
    second = build_visual_evidence(
        root=ROOT, config=config, output_dir=tmp_path / "second"
    )
    assert first["gold_sample_count"] == 9
    assert first["private_mapping_sha256"] == second["private_mapping_sha256"]
    assert [row["sha256"] for row in first["blind_sheets"]] == [
        row["sha256"] for row in second["blind_sheets"]
    ]
    assert "candidate" not in (
        tmp_path / "first" / "build_report.json"
    ).read_text(encoding="utf-8")
