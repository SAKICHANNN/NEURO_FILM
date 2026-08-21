from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p90_contract_is_frozen_and_bound() -> None:
    config = json.loads(
        (ROOT / "configs/p90_dng_aces2_hdr_pq_png_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "FROZEN_BEFORE_P90_FILE_PUBLICATION"
    assert config["target"] == "hdr_rec2020_pq"
    assert config["required_rows"] == 4
    assert len(config["bindings"]) == 4
