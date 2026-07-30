import json
from pathlib import Path


def test_ax15_keeps_validation_target_fit_and_metrics_forbidden() -> None:
    config = json.loads(
        Path(
            "configs/u5_r2ax15_filmmatch_cap070_validation_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["validation"]["fit_forbidden"] is True
    assert config["validation"]["pixel_metric_forbidden"] is True
    assert config["gates"]["validation_target_fit_allowed"] is False
    assert config["gates"]["target_pixel_metric_allowed"] is False
