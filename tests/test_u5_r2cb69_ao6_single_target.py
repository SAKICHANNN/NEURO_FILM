from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.analytic_y_chromaticity_throughput_candidate import (
    render_analytic_y_chromaticity_throughput_candidate,
)
from src.eval.fixed_ao6_single_target import render_fixed_ao6_single_target
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.inference.analytic_y_chromaticity_profile import (
    load_analytic_y_chromaticity_profile,
    render_analytic_y_chromaticity_profile,
)
from src.preprocess.types import SourceProfile, WorkingImage

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/analytic_y_chromaticity_cb66_v3.json"


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("row_chunk", [7, 32])
def test_cb69_single_target_is_byte_exact(workers: int, row_chunk: int) -> None:
    rng = np.random.default_rng(6969)
    source = rng.uniform(0.001, 0.999, size=(71, 93, 3)).astype(np.float32)
    runtime = load_analytic_y_chromaticity_profile(PROFILE, root=ROOT)
    expected = render_fixed_pair(
        source, runtime.artifact, runtime.ao6_config["component"]
    )[runtime.ao6_config["arm_id"]]
    actual = render_fixed_ao6_single_target(
        source,
        runtime.artifact,
        runtime.ao6_config["component"],
        row_chunk=row_chunk,
        workers=workers,
    )
    assert actual.tobytes() == expected.tobytes()


def test_cb69_complete_throughput_candidate_is_byte_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(6970)
    pixels = rng.uniform(0.002, 0.94, size=(61, 89, 3)).astype(np.float32)
    working = WorkingImage(
        pixels=pixels,
        working_space="linear_srgb",
        transfer_state="display_linear",
        source_transfer_state="display_referred",
        source_profile=SourceProfile(kind="assumed_srgb", description="test"),
        hdr_metadata={},
        bit_depth_in=8,
        orientation_applied=True,
        alpha_policy="absent",
        source_path=Path("synthetic.png"),
    )
    runtime = load_analytic_y_chromaticity_profile(PROFILE, root=ROOT)
    expected, expected_facts = render_analytic_y_chromaticity_profile(
        working, runtime, scratch_root=tmp_path
    )
    actual, actual_facts = render_analytic_y_chromaticity_throughput_candidate(
        working, runtime, scratch_root=tmp_path
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_facts == expected_facts
    assert list(tmp_path.iterdir()) == []
