from __future__ import annotations

import json
from pathlib import Path

from src.eval.rawpixls_confirmation_preflight import validate_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4hy_opponent_diffusion_fresh_source_preflight_v1.json"


def test_p4hy_source_contract_is_valid_and_fresh_against_p7h() -> None:
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, contract)
    assert len(contract["candidates"]) == 12
    assert len({row["make"] for row in contract["candidates"]}) == 12
    p7h = json.loads(
        (
            ROOT
            / "outputs/u6_p7h0_native_structure_value_source_preflight_v1/run_d/manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert not (
        {row["repository_id"] for row in contract["candidates"]}
        & {int(row["source_url"].split("getfile.php/")[1].split("/")[0]) for row in p7h}
    )
    assert not (
        {row["sha256"] for row in contract["candidates"]}
        & {row["raw_sha256"] for row in p7h}
    )
