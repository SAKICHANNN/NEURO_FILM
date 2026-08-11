from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.analytic_y_chromaticity_compact_parallel_candidate import (
    render_analytic_y_chromaticity_compact_parallel_candidate,
)
from src.eval.analytic_y_chromaticity_compact_selector import (
    select_analytic_y_chromaticity_candidate_compact,
)
from src.eval.analytic_y_chromaticity_parallel_target_candidate import (
    render_analytic_y_chromaticity_parallel_target_candidate,
)
from src.eval.analytic_y_chromaticity_shared_parallel_candidate import (
    render_analytic_y_chromaticity_shared_parallel_candidate,
)
from src.eval.analytic_y_chromaticity_streaming import (
    select_analytic_y_chromaticity_candidate_streamed,
)
from src.eval.analytic_y_chromaticity_throughput_candidate import (
    render_analytic_y_chromaticity_throughput_candidate,
)
from src.eval.nonexpansive_fraction_transport_external_sort import (
    nonexpansive_fraction_transport_target_external_sorted,
)
from src.eval.nonexpansive_fraction_transport_parallel import (
    nonexpansive_fraction_transport_target_parallel,
)
from src.eval.nonexpansive_fraction_transport_shared_parallel import (
    nonexpansive_fraction_transport_target_shared_parallel,
)
from src.inference.analytic_y_chromaticity_profile import (
    load_analytic_y_chromaticity_profile,
)
from src.preprocess.types import SourceProfile, WorkingImage

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/analytic_y_chromaticity_cb66_v3.json"


@pytest.mark.parametrize("row_chunk", [1, 7, 64])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_cb70_parallel_target_is_byte_exact(
    tmp_path: Path, row_chunk: int, workers: int
) -> None:
    rng = np.random.default_rng(7070)
    base = rng.uniform(0.002, 0.94, size=(67, 83, 3)).astype(np.float32)
    target = np.clip(
        base + rng.normal(0.0, 0.04, size=base.shape), 0.002, 0.998
    ).astype(np.float32)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    expected = nonexpansive_fraction_transport_target_external_sorted(
        base,
        target,
        weights=weights,
        row_chunk=row_chunk,
        scratch_root=tmp_path,
    )
    actual = nonexpansive_fraction_transport_target_parallel(
        base,
        target,
        weights=weights,
        row_chunk=row_chunk,
        scratch_root=tmp_path,
        workers=workers,
    )
    assert actual.tobytes() == expected.tobytes()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("row_chunk", [1, 7, 64])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_cb72_shared_parallel_target_is_byte_exact(
    tmp_path: Path, row_chunk: int, workers: int
) -> None:
    rng = np.random.default_rng(7272)
    base = rng.uniform(0.002, 0.94, size=(67, 83, 3)).astype(np.float32)
    target = np.clip(
        base + rng.normal(0.0, 0.04, size=base.shape), 0.002, 0.998
    ).astype(np.float32)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    expected = nonexpansive_fraction_transport_target_external_sorted(
        base,
        target,
        weights=weights,
        row_chunk=row_chunk,
        scratch_root=tmp_path,
    )
    actual = nonexpansive_fraction_transport_target_shared_parallel(
        base,
        target,
        weights=weights,
        row_chunk=row_chunk,
        scratch_root=tmp_path,
        workers=workers,
    )
    assert actual.tobytes() == expected.tobytes()
    assert list(tmp_path.iterdir()) == []


def test_cb70_complete_candidate_is_byte_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(7071)
    pixels = rng.uniform(0.002, 0.94, size=(73, 97, 3)).astype(np.float32)
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
    expected, expected_facts = render_analytic_y_chromaticity_throughput_candidate(
        working, runtime, scratch_root=tmp_path
    )
    actual, actual_facts = render_analytic_y_chromaticity_parallel_target_candidate(
        working, runtime, scratch_root=tmp_path
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_facts == expected_facts
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("row_chunk", [1, 7, 64])
def test_cb71_compact_selector_is_exact(tmp_path: Path, row_chunk: int) -> None:
    rng = np.random.default_rng(7171)
    source = rng.uniform(0.002, 0.94, size=(67, 83, 3)).astype(np.float32)
    target = np.clip(
        source + rng.normal(0.0, 0.025, size=source.shape), 0.002, 0.998
    ).astype(np.float32)
    runtime = load_analytic_y_chromaticity_profile(PROFILE, root=ROOT)
    operator = runtime.cb11["operator"]
    config = runtime.cb52
    kwargs = {
        "curve": runtime.curve,
        "strength": float(operator["nominal_strength"]),
        "boundary_epsilon": float(operator["boundary_epsilon"]),
        "dose_grid": config["operator"]["dose_grid"],
        "maximum_gradient_ratio": float(
            config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
        ),
        "maximum_lstar_inversion_fraction": float(
            config["automatic_gates"][
                "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
            ]
        ),
        "lstar_order_epsilon": float(config["operator"]["lstar_order_epsilon"]),
        "row_chunk": row_chunk,
        "scratch_root": tmp_path,
    }
    expected, _, _, expected_facts = (
        select_analytic_y_chromaticity_candidate_streamed(source, target, **kwargs)
    )
    actual, actual_facts = select_analytic_y_chromaticity_candidate_compact(
        source, target, **kwargs
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_facts == expected_facts
    assert list(tmp_path.iterdir()) == []


def test_cb71_complete_compact_parallel_candidate_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(7172)
    pixels = rng.uniform(0.002, 0.94, size=(73, 97, 3)).astype(np.float32)
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
    expected, expected_facts = render_analytic_y_chromaticity_throughput_candidate(
        working, runtime, scratch_root=tmp_path
    )
    actual, actual_facts = (
        render_analytic_y_chromaticity_compact_parallel_candidate(
            working, runtime, scratch_root=tmp_path
        )
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_facts == expected_facts
    assert list(tmp_path.iterdir()) == []


def test_cb72_complete_shared_parallel_candidate_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(7273)
    pixels = rng.uniform(0.002, 0.94, size=(73, 97, 3)).astype(np.float32)
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
    expected, expected_facts = render_analytic_y_chromaticity_throughput_candidate(
        working, runtime, scratch_root=tmp_path
    )
    actual, actual_facts = render_analytic_y_chromaticity_shared_parallel_candidate(
        working, runtime, scratch_root=tmp_path
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_facts == expected_facts
    assert list(tmp_path.iterdir()) == []
