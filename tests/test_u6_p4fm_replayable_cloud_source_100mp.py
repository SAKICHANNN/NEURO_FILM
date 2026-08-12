import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4fm_contract() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4fm_replayable_cloud_source_100mp_v1.json").read_text()
    )
    assert contract["scenario"]["height"] * contract["scenario"]["width"] == 100_000_000
    assert contract["scenario"]["runs"] == 2
    assert "expected_output_sha256" not in contract["scenario"]
