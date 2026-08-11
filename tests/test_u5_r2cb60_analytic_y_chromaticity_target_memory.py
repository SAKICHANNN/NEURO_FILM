from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.analytic_y_chromaticity_memory_optimized import (
    apply_characteristic_luma_chroma_output_row_materialized,
    render_analytic_y_chromaticity_memory_candidate,
)
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.nonexpansive_fraction_transport import (
    NonexpansiveFractionTransportError,
    nonexpansive_fraction_transport_target,
)
from src.eval.nonexpansive_fraction_transport_external_sort import (
    nonexpansive_fraction_transport_target_external_sorted,
)
from src.eval.nonexpansive_fraction_transport_statistics_streaming import (
    nonexpansive_fraction_transport_target_statistics_streamed,
)
from src.eval.nonexpansive_fraction_transport_streaming import (
    nonexpansive_fraction_transport_target_row_materialized,
)
from src.inference.analytic_y_chromaticity_profile import (
    load_analytic_y_chromaticity_profile,
    render_analytic_y_chromaticity_profile,
)
from src.preprocess.types import SourceProfile, WorkingImage

WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/analytic_y_chromaticity_cb56_v1.json"


@pytest.mark.parametrize("row_chunk", [1, 7, 64])
def test_cb60_row_materialized_target_is_byte_exact(row_chunk: int) -> None:
    rng = np.random.default_rng(6059)
    base = rng.uniform(0.002, 0.94, size=(79, 113, 3)).astype(np.float32)
    ao6 = np.empty_like(base)
    ao6[..., 0] = np.float32(0.88) * base[..., 0] + np.float32(0.12) * base[..., 1]
    ao6[..., 1] = np.float32(0.88) * base[..., 1] + np.float32(0.12) * base[..., 2]
    ao6[..., 2] = np.float32(0.88) * base[..., 2] + np.float32(0.12) * base[..., 0]
    expected = nonexpansive_fraction_transport_target(base, ao6, weights=WEIGHTS)
    actual = nonexpansive_fraction_transport_target_row_materialized(
        base, ao6, weights=WEIGHTS, row_chunk=row_chunk
    )
    assert actual.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


def test_cb60_rejects_invalid_row_chunk_without_output() -> None:
    source = np.full((3, 4, 3), 0.25, dtype=np.float32)
    with pytest.raises(NonexpansiveFractionTransportError, match="CB60 input drift"):
        nonexpansive_fraction_transport_target_row_materialized(
            source, source, weights=WEIGHTS, row_chunk=0
        )


@pytest.mark.parametrize("row_chunk", [1, 7, 64])
def test_cb63_statistics_streamed_target_is_byte_exact(row_chunk: int) -> None:
    rng = np.random.default_rng(6359)
    base = rng.uniform(0.002, 0.94, size=(79, 113, 3)).astype(np.float32)
    ao6 = np.empty_like(base)
    ao6[..., 0] = np.float32(0.88) * base[..., 0] + np.float32(0.12) * base[..., 1]
    ao6[..., 1] = np.float32(0.88) * base[..., 1] + np.float32(0.12) * base[..., 2]
    ao6[..., 2] = np.float32(0.88) * base[..., 2] + np.float32(0.12) * base[..., 0]
    expected = nonexpansive_fraction_transport_target(base, ao6, weights=WEIGHTS)
    actual = nonexpansive_fraction_transport_target_statistics_streamed(
        base, ao6, weights=WEIGHTS, row_chunk=row_chunk
    )
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("row_chunk", [1, 7, 64])
def test_cb64_external_sorted_target_is_byte_exact(
    row_chunk: int, tmp_path: Path
) -> None:
    rng = np.random.default_rng(6464)
    base = rng.uniform(0.002, 0.94, size=(79, 113, 3)).astype(np.float32)
    ao6 = np.empty_like(base)
    ao6[..., 0] = np.float32(0.88) * base[..., 0] + np.float32(0.12) * base[..., 1]
    ao6[..., 1] = np.float32(0.88) * base[..., 1] + np.float32(0.12) * base[..., 2]
    ao6[..., 2] = np.float32(0.88) * base[..., 2] + np.float32(0.12) * base[..., 0]
    expected = nonexpansive_fraction_transport_target(base, ao6, weights=WEIGHTS)
    actual = nonexpansive_fraction_transport_target_external_sorted(
        base,
        ao6,
        weights=WEIGHTS,
        row_chunk=row_chunk,
        scratch_root=tmp_path,
    )
    assert actual.tobytes() == expected.tobytes()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "target_builder",
    [
        nonexpansive_fraction_transport_target_row_materialized,
        nonexpansive_fraction_transport_target_external_sorted,
    ],
)
def test_cb60_complete_profile_output_is_exact(target_builder, tmp_path: Path) -> None:
    rng = np.random.default_rng(6060)
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
    expected, expected_facts = render_analytic_y_chromaticity_profile(working, runtime)
    actual, actual_facts = render_analytic_y_chromaticity_profile(
        working,
        runtime,
        scratch_root=tmp_path,
        target_builder=target_builder,
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_facts == expected_facts


@pytest.mark.parametrize("row_chunk", [1, 7, 64])
def test_cb66_row_materialized_safe_base_is_byte_exact(row_chunk: int) -> None:
    rng = np.random.default_rng(6661)
    pixels = rng.uniform(0.002, 0.94, size=(37, 53, 3)).astype(np.float32)
    runtime = load_analytic_y_chromaticity_profile(PROFILE, root=ROOT)
    operator = runtime.cb11["operator"]
    weights = np.asarray(operator["luminance_weights"], dtype=np.float64)
    expected, _, _ = apply_characteristic_luma_chroma(
        pixels,
        runtime.curve,
        weights=weights,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=float(operator["boundary_epsilon"]),
    )
    actual = apply_characteristic_luma_chroma_output_row_materialized(
        pixels,
        runtime.curve,
        weights=weights,
        strength=float(operator["nominal_strength"]),
        boundary_epsilon=float(operator["boundary_epsilon"]),
        row_chunk=row_chunk,
    )
    assert actual.tobytes() == expected.tobytes()


def test_cb66_complete_memory_candidate_is_byte_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(6662)
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
    expected, expected_facts = render_analytic_y_chromaticity_profile(working, runtime)
    actual, actual_facts = render_analytic_y_chromaticity_memory_candidate(
        working, runtime, scratch_root=tmp_path, row_chunk=7
    )
    assert actual.tobytes() == expected.tobytes()
    assert actual_facts == expected_facts
    assert list(tmp_path.iterdir()) == []
