from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.film_physics.native_standard_output import (
    commit_verified_native_standard_png16,
)
from src.film_physics.native_standard_staging import (
    stage_native_standard_working_image,
)
from src.film_physics.native_standard_strength_output import (
    commit_verified_native_standard_strength_png16,
    verify_native_standard_strength_png16,
)
from src.film_physics.native_standard_strength_staging import (
    stage_native_standard_working_image_with_strength,
    verify_native_standard_strength_staging,
)
from tests.test_u6_p8bf_native_standard_staging import _fixture


def test_p8bm_full_strength_preserves_v1_float_and_png_bytes(
    tmp_path: Path,
) -> None:
    runtime, working = _fixture(tmp_path)
    legacy = stage_native_standard_working_image(
        runtime,
        working,
        output_path=tmp_path / "legacy.f32",
        report_path=tmp_path / "legacy.json",
    )
    strength = stage_native_standard_working_image_with_strength(
        runtime,
        working,
        strength=1.0,
        output_path=tmp_path / "strength.f32",
        report_path=tmp_path / "strength.json",
    )
    assert legacy["output_sha256"] == strength["output_sha256"]
    assert (
        (tmp_path / "legacy.f32").read_bytes()
        == (tmp_path / "strength.f32").read_bytes()
    )

    legacy_png = commit_verified_native_standard_png16(
        staging_report_path=Path(legacy["report_path"]),
        expected_staging_report_sha256=legacy["report_sha256"],
        expected_staging_run_id=legacy["run_id"],
        output_path=tmp_path / "legacy.png",
        report_path=tmp_path / "legacy-png.json",
    )
    strength_png = commit_verified_native_standard_strength_png16(
        staging_report_path=Path(strength["report_path"]),
        expected_staging_report_sha256=strength["report_sha256"],
        expected_staging_run_id=strength["run_id"],
        output_path=tmp_path / "strength.png",
        report_path=tmp_path / "strength-png.json",
    )
    assert legacy_png["output_sha256"] == strength_png["output_sha256"]
    assert (
        (tmp_path / "legacy.png").read_bytes()
        == (tmp_path / "strength.png").read_bytes()
    )


def test_p8bm_zero_strength_is_exact_encoded_source(
    tmp_path: Path,
) -> None:
    runtime, working = _fixture(tmp_path)
    staged = stage_native_standard_working_image_with_strength(
        runtime,
        working,
        strength=0.0,
        output_path=tmp_path / "zero.f32",
        report_path=tmp_path / "zero.json",
    )
    actual = np.fromfile(
        staged["output_path"], dtype="<f4"
    ).reshape(working.pixels.shape)
    expected = np.ascontiguousarray(
        linear_srgb_to_encoded(
            working.pixels.astype(np.float64)
        ),
        dtype=np.float32,
    )
    assert np.array_equal(actual, expected)


def test_p8bm_strength_transaction_restart_verifies_and_tamper_fails(
    tmp_path: Path,
) -> None:
    runtime, working = _fixture(tmp_path)
    staged = stage_native_standard_working_image_with_strength(
        runtime,
        working,
        strength=0.8,
        output_path=tmp_path / "candidate.f32",
        report_path=tmp_path / "candidate.json",
    )
    first_stage = verify_native_standard_strength_staging(
        report_path=Path(staged["report_path"]),
        expected_report_sha256=staged["report_sha256"],
        expected_run_id=staged["run_id"],
    )
    second_stage = verify_native_standard_strength_staging(
        report_path=Path(staged["report_path"]),
        expected_report_sha256=staged["report_sha256"],
        expected_run_id=staged["run_id"],
    )
    assert first_stage == second_stage
    assert first_stage["strength"] == 0.8

    committed = commit_verified_native_standard_strength_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=tmp_path / "candidate.png",
        report_path=tmp_path / "candidate-png.json",
    )
    first_png = verify_native_standard_strength_png16(
        report_path=Path(committed["report_path"]),
        expected_report_sha256=committed["report_sha256"],
        expected_delivery_id=committed["delivery_id"],
    )
    second_png = verify_native_standard_strength_png16(
        report_path=Path(committed["report_path"]),
        expected_report_sha256=committed["report_sha256"],
        expected_delivery_id=committed["delivery_id"],
    )
    assert first_png == second_png
    assert first_png["strength"] == 0.8
    decoded = cv2.imread(
        committed["output_path"], cv2.IMREAD_UNCHANGED
    )[..., ::-1]
    raw = np.fromfile(
        staged["output_path"], dtype="<f4"
    ).reshape(working.pixels.shape)
    assert np.array_equal(
        decoded,
        np.rint(raw * 65535.0).astype(np.uint16),
    )

    output = Path(committed["output_path"])
    original = output.read_bytes()
    output.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="output drift"):
        verify_native_standard_strength_png16(
            report_path=Path(committed["report_path"]),
            expected_report_sha256=committed["report_sha256"],
            expected_delivery_id=committed["delivery_id"],
        )


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), True])
def test_p8bm_rejects_invalid_strength_before_artifacts(
    tmp_path: Path,
    strength: float,
) -> None:
    runtime, working = _fixture(tmp_path)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        stage_native_standard_working_image_with_strength(
            runtime,
            working,
            strength=strength,
            output_path=tmp_path / "bad.f32",
            report_path=tmp_path / "bad.json",
        )
    assert not list(tmp_path.glob("bad.*"))
    assert not list(tmp_path.glob(".*.stage"))
