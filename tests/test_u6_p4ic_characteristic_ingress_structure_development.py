import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4ic_contract_is_typed_development_only() -> None:
    contract = json.loads(
        (
            ROOT
            / "configs/u6_p4ic_characteristic_ingress_structure_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert "Development-only typed characteristic-ingress" in contract["claim_ceiling"]
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["decision_if_fail"].startswith("close_exact_p4ic")
