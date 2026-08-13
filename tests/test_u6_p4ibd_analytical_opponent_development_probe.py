import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4ibd_is_explicitly_development_only() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u6_p4ibd_analytical_opponent_development_probe_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert "Development-only" in contract["claim_ceiling"]
    assert "wholly_fresh" in contract["decision_if_pass"]
    assert contract["candidate"]["hard_clipping_allowed"] is False
