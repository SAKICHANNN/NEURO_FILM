from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/RF3_D13_DARKROOM_THREE_STOCK_BASELINE_SOURCE_RESULT.json"


def test_rf3_d13_evidence_is_exact_and_fail_closed() -> None:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "65e1f666ca7472c92eb9a673a1c9ec1f612809fa7dea3267e1189251f09a29bd"
    )
    value = json.loads(payload)
    scientific = value["scientific"]
    assert scientific["status"] == (
        "FAIL_CLOSED_MISSING_ROOT_LICENSE_AND_PER_STOCK_PROVENANCE"
    )
    assert scientific["decision"] == (
        "CLOSE_SOURCE_WITHOUT_PIXEL_EXECUTION_OR_RIGHTS_RESCUE"
    )
    assert scientific["source"]["commit"] == (
        "eb8f3d86f49a57e54ff426b3f517f8747248ec99"
    )
    assert scientific["source"]["root_license_candidates"] == []
    assert scientific["gates"]["three_exact_machine_readable_stock_operators"] is True
    assert scientific["gates"]["materially_distinct_from_closed_spectral_film_lut_route"] is True
    assert scientific["gates"]["root_license_covers_code_and_operator_data"] is False
    assert scientific["gates"]["per_stock_primary_source_and_extraction_provenance"] is False
    assert scientific["gates"]["no_unresolved_third_party_data_rights"] is False
    assert {row["stock_id"] for row in scientific["stock_facts"]} == {
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    }
    assert all(row["provenance_keys_present"] == [] for row in scientific["stock_facts"])
    assert scientific["observations"]["image_or_film_scan_reads"] == 0
    assert scientific["observations"]["operator_pixel_executions"] == 0
