from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.inference import list_product_looks

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2E_PRODUCT_LOOK_CATALOG_RESULT.json"


def test_u7_2e_evidence_is_bound_to_current_product_catalog() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_UNIFIED_EVIDENCE_BOUNDED_PRODUCT_LOOK_CATALOG"
    for binding in evidence["bindings"].values():
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    assert evidence["catalog"]["ordered_look_ids"] == [
        row["look_id"] for row in list_product_looks()
    ]
    assert evidence["catalog"]["generic_bw_film_stock_id"] is None
    assert evidence["catalog"]["named_legacy_bw_product_ids_exposed"] is False
    assert evidence["formal_probe"]["all_full_tiled_exact"] is True
    assert evidence["multi_stock_completion"] is False
