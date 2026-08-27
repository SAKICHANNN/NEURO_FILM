from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.three_stock_face_severe_review import canonical_bytes, run_review

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u4_3c_three_stock_face_severe_review_v1.json"


def test_u4_3c_real_face_forward_reverse_are_exact(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forward = run_review(config, ROOT, tmp_path / "forward")
    reverse = run_review(config, ROOT, tmp_path / "reverse", reverse=True)
    assert canonical_bytes(forward) == canonical_bytes(reverse)
    assert (
        forward["scientific_payload"]["automatic_status"] == "PASS_OPEN_VISUAL_REVIEW"
    )
    assert [
        row["film_stock_id"] for row in forward["scientific_payload"]["rows"]
    ] == config["execution"]["film_stock_ids"]
    assert all(forward["scientific_payload"]["automatic_gates"].values())
    assert forward["scientific_payload"]["source"]["original_file_reopened"] is False
    for key, fact in forward["artifacts"].items():
        assert fact == reverse["artifacts"][key]
        assert (
            hashlib.sha256(
                (tmp_path / "forward" / fact["path"]).read_bytes()
            ).hexdigest()
            == fact["sha256"]
        )


def test_u4_3c_contract_keeps_claims_and_ao6_out() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["execution"]["film_stock_ids"] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert "AO6 is excluded" in (
        ROOT / "docs/planning/U4_3C_THREE_STOCK_FACE_SEVERE_REVIEW_CONTRACT.md"
    ).read_text(encoding="utf-8")
    assert "not original file ingress" in config["claim_ceiling"]
