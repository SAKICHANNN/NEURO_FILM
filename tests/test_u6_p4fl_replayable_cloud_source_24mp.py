import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4fl_contract() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4fl_replayable_cloud_source_24mp_v1.json").read_text()
    )
    assert contract["scenario"]["height"] * contract["scenario"]["width"] == 24_000_000
    assert contract["scenario"]["runs"] == 2
