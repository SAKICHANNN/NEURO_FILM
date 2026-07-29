from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from src.film_physics.native_standard_output import (
    commit_verified_native_standard_png16,
    verify_native_standard_png16,
)
from src.film_physics.native_standard_staging import (
    stage_native_standard_working_image,
)
from tests.test_u6_p8bf_native_standard_staging import _fixture


def test_p8bg_png16_commit_and_restart_verification(
    tmp_path: Path,
) -> None:
    runtime, working = _fixture(tmp_path)
    staged = stage_native_standard_working_image(
        runtime,
        working,
        output_path=tmp_path / "source.f32",
        report_path=tmp_path / "source.json",
    )
    committed = commit_verified_native_standard_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=tmp_path / "output.png",
        report_path=tmp_path / "output.json",
    )
    first = verify_native_standard_png16(
        report_path=Path(committed["report_path"]),
        expected_report_sha256=committed["report_sha256"],
        expected_delivery_id=committed["delivery_id"],
    )
    second = verify_native_standard_png16(
        report_path=Path(committed["report_path"]),
        expected_report_sha256=committed["report_sha256"],
        expected_delivery_id=committed["delivery_id"],
    )
    assert first == second
    with Image.open(committed["output_path"]) as image:
        assert image.size == (35, 33)
        assert image.info["icc_profile"]
    decoded = cv2.imread(
        str(committed["output_path"]), cv2.IMREAD_UNCHANGED
    )[..., ::-1]
    staged_values = np.fromfile(
        staged["output_path"], dtype="<f4"
    ).reshape(33, 35, 3)
    expected = np.rint(staged_values * 65535.0).astype(np.uint16)
    assert np.array_equal(decoded, expected)

    output = Path(committed["output_path"])
    raw = output.read_bytes()
    output.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="output drift"):
        verify_native_standard_png16(
            report_path=Path(committed["report_path"]),
            expected_report_sha256=committed["report_sha256"],
            expected_delivery_id=committed["delivery_id"],
        )
