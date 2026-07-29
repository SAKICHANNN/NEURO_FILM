from __future__ import annotations

import json
from pathlib import Path

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw
from scripts.benchmark_u6_p8ax_native_ordered_pipeline_grid import (
    _worker as pipeline_worker,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8aw_native_display_v4_resources_v1.json"


def test_p8ax_four_worker_pipeline_is_exact_to_serial_chain(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    p8aw._patch_runtime()
    builds = p8aq._build_components(config, tmp_path / "binaries")
    common = {
        "profile_config": ROOT / config["profile_compiler_config"],
        "domains_dll": Path(builds["domains"]["dll_path"]),
        "gaussian_dll": Path(builds["gaussian"]["dll_path"]),
        "adjacency_dll": Path(builds["adjacency"]["dll_path"]),
        "gauge_dll": Path(builds["gauge"]["dll_path"]),
        "context_dll": Path(builds["context"]["dll_path"]),
        "display_dll": Path(builds["display"]["dll_path"]),
        "height": 64,
        "width": 96,
        "tile_rows": 8,
    }
    serial_path = tmp_path / "serial.json"
    pipeline_path = tmp_path / "pipeline.json"
    p8aq._worker(output=serial_path, **common)
    pipeline_worker(
        output=pipeline_path,
        pipeline_workers=4,
        max_in_flight=4,
        **common,
    )
    serial = json.loads(serial_path.read_text())
    pipeline = json.loads(pipeline_path.read_text())

    assert pipeline["output_sha256"] == serial["output_sha256"]
    assert pipeline["submitted_tiles"] == 8
    assert pipeline["consumed_tiles"] == 8
    assert pipeline["maximum_observed_in_flight"] == 4
