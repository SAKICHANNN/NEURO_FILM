from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.three_stock_structural_visual_review import (
    ThreeStockStructuralVisualReviewError,
    build_review_material,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_3b_three_stock_structural_visual_adjudication_v1.json"


def test_u4_3b_material_is_order_independent_and_preselected(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forward = build_review_material(config, ROOT, tmp_path / "forward")
    reverse = build_review_material(config, ROOT, tmp_path / "reverse", reverse=True)
    assert forward == reverse
    assert forward["status"] == "REVIEW_MATERIAL_READY"
    payload = forward["scientific_payload"]
    assert len(payload["sheets"]) == 11
    assert payload["render_calls"] == 0
    assert payload["network_reads"] == 0
    assert payload["adjudications_present"] is False
    for row in payload["sheets"]:
        assert len(row["review_boxes"]) == 3
        assert (tmp_path / "forward" / row["sheet"]["relative_path"]).is_file()


def test_u4_3b_rejects_report_identity_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["input_report"]["sha256"] = "0" * 64
    with pytest.raises(
        ThreeStockStructuralVisualReviewError, match="input hash mismatch"
    ):
        build_review_material(config, ROOT, tmp_path / "invalid")


def test_u4_3b_direct_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_u4_3b_three_stock_structural_visual_review.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
