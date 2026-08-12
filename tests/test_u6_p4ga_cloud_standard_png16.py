from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure,
    render_physical_partition,
)
from src.film_physics.native_cloud_scan_runtime_v2 import (
    WindowedNativeCloudScanRuntime,
)
from src.film_physics.native_cloud_standard_staging import (
    stage_cloud_scan_with_standard_display,
)
from src.film_physics.native_standard_output import (
    commit_verified_native_standard_png16,
    verify_native_standard_png16,
)
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile
from tests.test_u6_p4fn_native_standard_replayable_rows import _runtime

ROOT = Path(__file__).resolve().parents[1]
P4FB = ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json"


def test_cloud_standard_display_reaches_restart_verified_png16(tmp_path: Path) -> None:
    height, width = 64, 96
    source = np.ascontiguousarray(
        p8aq._source_rows(y0=0, y1=height, height=height, width=width),
        dtype=np.float32,
    )
    source_sha = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
    standard = _runtime(tmp_path)
    contract = json.loads(P4FB.read_text())
    contract["fixture"]["full_height"] = height
    contract["fixture"]["width"] = width
    dll = _build(ROOT, tmp_path / "native", None)
    lib = _configure(dll)
    component_sha = hashlib.sha256(
        (ROOT / "native/film_physics/nf_cloud_post_spatial_f32_v1.c").read_bytes()
    ).hexdigest()

    def provider(forward_rows, y0: int, count: int) -> np.ndarray:
        return render_physical_partition(lib, contract, forward_rows, y0, count)

    cloud = WindowedNativeCloudScanRuntime(
        gaussian_library=dll,
        forward_scatter_profile=_profile(),
        physical_rows=provider,
        physical_component_sha256=component_sha,
        tile_rows=8,
    )
    staged = stage_cloud_scan_with_standard_display(
        standard,
        cloud,
        height=height,
        width=width,
        source_rows=lambda start, count: np.ascontiguousarray(source[start : start + count]),
        expected_input_sha256=source_sha,
        output_path=tmp_path / "cloud.f32",
        report_path=tmp_path / "cloud.raw.json",
    )
    committed = commit_verified_native_standard_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=tmp_path / "cloud.png",
        report_path=tmp_path / "cloud.png.json",
    )
    verified = verify_native_standard_png16(
        report_path=Path(committed["report_path"]),
        expected_report_sha256=committed["report_sha256"],
        expected_delivery_id=committed["delivery_id"],
    )
    assert verified["output_sha256"] == committed["output_sha256"]
    assert (tmp_path / "cloud.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    staging_payload = json.loads((tmp_path / "cloud.raw.json").read_text())
    assert staging_payload["working_image_receipt"]["source_passes"] == 4
    assert staging_payload["working_image_receipt"]["physical_order"][-3:] == [
        "neutral-gauge",
        "srgb-oetf",
        "ao6-display",
    ]
