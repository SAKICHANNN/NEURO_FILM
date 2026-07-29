from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_u5_r2aq3t_colorreference_calibration_suite_topology_v2 import (
    CONFIG_SHA256,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq3t_colorreference_calibration_suite_topology_v2.json"
)


def test_v2_repairs_only_source_design() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["fit_allowed"] is False
    assert config["target_fields_allowed"] is False
    assert config["aq2_reopened"] is False
    assert config["operator_promotion_allowed"] is False
    assert config["source_colour_fold_contract"][
        "expected_fold_row_counts"
    ] == [295, 302, 274, 275, 294]
    assert config["source_colour_fold_contract"][
        "maximum_outside_development_convex_hull_fraction"
    ] == 0.05


def test_v2_binds_failed_predecessor() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["supersedes_source_design_only"]["report_sha256"] == (
        "f07c9bb6f09d6b5357e86cb1c28de04488c925866d48f6127c5f7d00bad993fa"
    )


def test_v2_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["fit_allowed"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
