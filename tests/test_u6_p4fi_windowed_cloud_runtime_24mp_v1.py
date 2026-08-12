import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_p4fi_contract()->None:
    contract=json.loads((ROOT/"configs/u6_p4fi_windowed_cloud_runtime_24mp_v1.json").read_text())
    assert contract["scenario"]["run_order"]==["windowed","windowed"]
    assert contract["scenario"]["height"]*contract["scenario"]["width"]==24_000_000
