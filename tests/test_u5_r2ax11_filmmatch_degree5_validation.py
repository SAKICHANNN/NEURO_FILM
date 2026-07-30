import json
from pathlib import Path


def test_ax11_binds_exact_degree5_champion() -> None:
    config = json.loads(
        Path(
            "configs/u5_r2ax11_filmmatch_degree5_validation_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["expected_development_champion"] == "degree5_ridge10"
    assert config["validation"]["fit_forbidden"] is True
