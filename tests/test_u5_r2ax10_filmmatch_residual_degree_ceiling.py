import json
from pathlib import Path


def test_ax10_freezes_capacity_stop_after_failure() -> None:
    config = json.loads(
        Path(
            "configs/u5_r2ax10_filmmatch_residual_degree_ceiling_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["gates"]["capacity_expansion_after_failure_allowed"] is False
    assert max(
        int(row["residual_degree"])
        for row in config["candidate"]["variants"]
    ) == 5
