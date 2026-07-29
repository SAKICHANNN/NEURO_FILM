from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.film_physics.native_standard_file import (
    render_native_standard_file_to_png16,
)
from tests.test_u6_p8bf_native_standard_staging import _fixture


ROOT = Path(__file__).resolve().parents[1]
RAW_FIXTURE = (
    ROOT
    / "data/external/rawpixls_ao7_v1/raw/fujifilm_finepix_s5000.raf"
)
DECISION = (
    ROOT
    / "configs/u6_p8bh_native_standard_file_ingress_decision_v1.json"
)


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "raw_output_path": tmp_path / "render.f32",
        "raw_report_path": tmp_path / "render.raw.json",
        "png_output_path": tmp_path / "render.png",
        "png_report_path": tmp_path / "render.png.json",
    }


def test_p8bh_display_referred_raster_fails_before_output(
    tmp_path: Path,
) -> None:
    runtime, _ = _fixture(tmp_path)
    raster = tmp_path / "input.png"
    Image.fromarray(
        np.full((8, 9, 3), 127, dtype=np.uint8), mode="RGB"
    ).save(raster)
    outputs = _paths(tmp_path)
    with pytest.raises(ValueError, match="scene-linear"):
        render_native_standard_file_to_png16(
            runtime,
            input_path=raster,
            **outputs,
        )
    assert not any(path.exists() for path in outputs.values())


@pytest.mark.skipif(
    not RAW_FIXTURE.is_file(), reason="bounded RAW fixture absent"
)
def test_p8bh_real_raw_file_reaches_exact_png_transaction(
    tmp_path: Path,
) -> None:
    runtime, _ = _fixture(tmp_path)
    result = render_native_standard_file_to_png16(
        runtime,
        input_path=RAW_FIXTURE,
        **_paths(tmp_path),
    )
    assert result["transfer_state"] == "scene_linear"
    assert result["orientation_applied"]
    assert len(result["png_output_sha256"]) == 64
    assert not result["production_default_changed"]


def test_p8bh_decision_binds_ingress_sources_and_raw_fixture() -> None:
    decision = json.loads(DECISION.read_text())
    for row in decision["implementation"].values():
        assert hashlib.sha256(
            (ROOT / row["path"]).read_bytes()
        ).hexdigest() == row["sha256"]
    assert hashlib.sha256(RAW_FIXTURE.read_bytes()).hexdigest() == (
        decision["real_raw_smoke"]["sha256"]
    )
    assert decision["result"]["status"] == "pass-file-ingress-smoke"
    assert decision["next_leaf"].startswith("U6.P8BI")
