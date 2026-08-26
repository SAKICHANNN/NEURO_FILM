from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_p100_dng_forward_aces2_output as p100
from src.preprocess.ocio_aces2_output import apply_working_image_aces2_output
from src.preprocess.types import SourceProfile, WorkingImage


def _working() -> WorkingImage:
    pixels = np.asarray(
        [
            [[-0.01, 0.0, 0.01], [0.18, 0.2, 0.25]],
            [[1.0, 1.2, 2.0], [4.0, 0.5, 0.125]],
        ],
        dtype=np.float32,
    )
    return WorkingImage(
        pixels=pixels,
        working_space="linear_rec2020",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "test"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("fixture.dng"),
        warnings=[],
    )


@pytest.mark.parametrize("target", ["sdr_rec709", "hdr_rec2020_pq"])
def test_direct_official_path_matches_adapter_bytes(target: str) -> None:
    working = _working()
    before = working.pixels.copy()
    adapter = apply_working_image_aces2_output(working, target)  # type: ignore[arg-type]
    direct = p100._direct_official_output(working, target)
    assert np.array_equal(adapter, direct)
    assert np.array_equal(working.pixels, before)
    assert np.isfinite(direct).all()


def test_config_is_frozen_before_execution() -> None:
    config = json.loads(
        (p100.ROOT / "configs/p100_dng_forward_aces2_output_v1.json").read_text()
    )
    assert config["status"] == "frozen_before_implementation_or_formal_execution"
    assert config["gates"]["required_rows"] == 5
    assert config["targets"] == ["sdr_rec709", "hdr_rec2020_pq"]


def test_binding_drift_rejects_before_source_decode(tmp_path: Path) -> None:
    config = json.loads(
        (p100.ROOT / "configs/p100_dng_forward_aces2_output_v1.json").read_text()
    )
    config["bindings"]["contract_sha256"] = "0" * 64
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch: contract"):
        p100.run(path, reverse=False)
