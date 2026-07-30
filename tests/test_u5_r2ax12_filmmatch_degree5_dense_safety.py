import json
from pathlib import Path


def test_ax12_preserves_full_dense_grid() -> None:
    config = json.loads(
        Path(
            "configs/u5_r2ax12_filmmatch_degree5_dense_safety_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["execution"]["cube_grid_size"] == 65
    assert config["automatic_gate"]["maximum_ramp_code_jump"] == 128
