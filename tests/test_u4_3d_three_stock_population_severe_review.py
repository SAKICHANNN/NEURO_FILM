from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.three_stock_structural_visual_review import (
    ThreeStockStructuralVisualReviewError,
    build_population_review_material,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_3d_three_stock_population_severe_review_v1.json"


def test_u4_3d_material_is_order_independent_and_complete(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forward = build_population_review_material(config, ROOT, tmp_path / "forward")
    reverse = build_population_review_material(
        config, ROOT, tmp_path / "reverse", reverse=True
    )
    assert forward == reverse
    payload = forward["scientific_payload"]
    assert payload["source_count"] == 16
    assert payload["output_count"] == 48
    assert payload["render_calls"] == 0
    assert payload["network_reads"] == 0
    assert payload["adjudications_present"] is False
    assert len(payload["sheets"]) == 16
    assert all(len(row["rows"]) == 3 for row in payload["sheets"])
    assert all(
        (tmp_path / "forward" / row["sheet"]["relative_path"]).is_file()
        for row in payload["sheets"]
    )


def test_u4_3d_rejects_report_identity_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["input_report"]["sha256"] = "0" * 64
    with pytest.raises(
        ThreeStockStructuralVisualReviewError, match="input hash mismatch"
    ):
        build_population_review_material(config, ROOT, tmp_path / "invalid")


def test_u4_3d_contract_excludes_ao6_and_product_claims() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["selection"]["arm_ids"]) == 3
    assert all("ao6" not in arm for arm in config["selection"]["arm_ids"])
    assert config["forbidden"]["operator_or_threshold_rescue"] is True
    assert "not target-film closeness" in config["claim_ceiling"]


def test_u4_3d_direct_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_u4_3d_three_stock_population_severe_review.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
