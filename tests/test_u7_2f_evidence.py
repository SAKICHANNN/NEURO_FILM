from __future__ import annotations

import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2F_PRODUCT_LOOK_CLI_RESULT.json"


def test_u7_2f_evidence_is_bound_to_current_cli_and_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_OPT_IN_GENERIC_BW_CLI_AND_EXACT_RECIPE_REPLAY"
    for binding in evidence["bindings"].values():
        assert_historical_evidence_binding(ROOT, binding)
    assert evidence["formal_probe"]["forward_reverse_exact"] is True
    assert evidence["formal_probe"]["rows"][0][
        "decoded_source_identity_exact"
    ] is True
    assert evidence["formal_probe"]["temporary_media_residue_count"] == 0
    assert evidence["compatibility"]["legacy_profile_changed"] is False
    assert evidence["compatibility"]["legacy_recipe_schema_changed"] is False
    assert evidence["multi_stock_completion"] is False
