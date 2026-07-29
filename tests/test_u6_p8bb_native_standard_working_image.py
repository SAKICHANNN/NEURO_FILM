from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq
from src.film_physics.native_standard_consumer import (
    render_native_standard_working_image_to_sink,
)
from src.preprocess.types import SourceProfile, WorkingImage
from tests.test_u6_p8ba_native_standard_runtime import _runtime_fixture


def _working(
    pixels: np.ndarray,
    *,
    working_space: str = "linear_srgb_d65",
    transfer_state: str = "scene_linear",
    orientation_applied: bool = True,
    alpha_policy: str = "absent",
) -> WorkingImage:
    return WorkingImage(
        pixels=pixels,
        working_space=working_space,
        transfer_state=transfer_state,
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "synthetic"),
        hdr_metadata={},
        orientation_applied=orientation_applied,
        alpha_policy=alpha_policy,
        bit_depth_in=16,
        source_path=Path("synthetic.raw"),
    )


def test_p8bb_working_image_adapter_is_exact_and_receipted(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime_fixture(tmp_path)
    pixels = np.ascontiguousarray(
        p8aq._source_rows(
            y0=0,
            y1=64,
            height=64,
            width=96,
        ),
        dtype=np.float32,
    )
    working = _working(pixels)
    digest = hashlib.sha256()
    row_starts: list[int] = []

    def sink(y0: int, y1: int, rows: np.ndarray) -> None:
        row_starts.append(y0)
        digest.update(rows.tobytes())

    receipt = render_native_standard_working_image_to_sink(
        runtime, working, output_sink=sink
    )

    assert row_starts == [0, 32]
    assert digest.hexdigest() == receipt["output"]["array_sha256"]
    assert receipt["input"]["array_sha256"] == hashlib.sha256(
        pixels.tobytes()
    ).hexdigest()
    assert receipt["runtime_receipt_sha256"] == receipt[
        "runtime_receipt"
    ]["receipt_sha256"]
    assert receipt["production_default_changed"] is False


def test_p8bb_working_image_adapter_fails_closed_before_sink(
    tmp_path: Path,
) -> None:
    runtime, _ = _runtime_fixture(tmp_path)
    for overrides in (
        {"working_space": "linear_rec2020"},
        {"transfer_state": "display_linear"},
        {"orientation_applied": False},
        {"alpha_policy": "preserved"},
    ):
        working = _working(
            np.zeros((8, 8, 3), dtype=np.float32),
            **overrides,
        )
        calls = 0

        def sink(y0: int, y1: int, rows: np.ndarray) -> None:
            nonlocal calls
            calls += 1

        try:
            render_native_standard_working_image_to_sink(
                runtime,
                working,
                output_sink=sink,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid WorkingImage was accepted")
        assert calls == 0
