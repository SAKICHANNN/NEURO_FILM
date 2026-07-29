from __future__ import annotations

import json
from pathlib import Path

from src.eval.normalized_reversal_photographic_stress import SCHEMA, evaluate


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs/u6_p2j_normalized_reversal_photographic_stress_v1.json"
)


def test_contract_excludes_noise_fitting_and_clipping() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == SCHEMA
    assert "noise" not in contract["pipeline"]["scanner_stages"]
    assert contract["automatic_gates"][
        "maximum_endpoint_domain_escape_fraction"
    ] == 0.0
    assert any("clip endpoint" in item for item in contract["forbidden"])


def test_full_evaluator_smoke_contract_has_fixed_population() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p2g = json.loads(
        (ROOT / contract["parents"]["p2g_contract_path"]).read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (ROOT / p2g["input"]["manifest"]).read_text(encoding="utf-8")
    )
    assert len(manifest) == contract["input"]["expected_rows"]
    assert len({row["make"] for row in manifest}) == contract["input"][
        "expected_camera_makes"
    ]
