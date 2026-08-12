import json
from pathlib import Path

from scripts.benchmark_u6_p4fg_windowed_cloud_runtime_resources import benchmark

ROOT=Path(__file__).resolve().parents[1]


def test_p4fg_contract_parent()->None:
    contract=json.loads((ROOT/"configs/u6_p4fg_windowed_cloud_runtime_resources_v1.json").read_text())
    assert contract["scenario"]["run_order"]==["full","windowed","windowed","full"]


def test_p4fg_small_smoke(tmp_path:Path)->None:
    contract=json.loads((ROOT/"configs/u6_p4fg_windowed_cloud_runtime_resources_v1.json").read_text());contract["scenario"].update({"height":73,"width":47})
    path=tmp_path/"contract.json";path.write_text(json.dumps(contract));result=benchmark(path,tmp_path/"run")
    assert result["stable"]["gates"]["output"]
