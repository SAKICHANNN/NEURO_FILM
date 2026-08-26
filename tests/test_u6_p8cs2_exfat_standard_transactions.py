from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.film_physics import create_only_file
from src.film_physics.native_standard_output import (
    commit_verified_native_standard_png16,
    verify_native_standard_png16,
)
from src.film_physics.native_standard_staging import (
    stage_native_standard_working_image,
    verify_native_standard_staging,
)
from src.film_physics.native_standard_strength_output import (
    commit_verified_native_standard_strength_png16,
    verify_native_standard_strength_png16,
)
from src.film_physics.native_standard_strength_staging import (
    stage_native_standard_working_image_with_strength,
    verify_native_standard_strength_staging,
)
from src.preprocess.types import SourceProfile, WorkingImage


def _working() -> WorkingImage:
    return WorkingImage(
        pixels=np.linspace(0.05, 0.85, 36, dtype=np.float32).reshape(3, 4, 3),
        working_space="linear_srgb_d65",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "synthetic"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("synthetic.raw"),
    )


def _fake_renderer(
    _runtime: object,
    working: WorkingImage,
    *,
    output_sink: Any,
) -> dict[str, Any]:
    styled = np.ascontiguousarray(
        0.2 + working.pixels * np.float32(0.6),
        dtype=np.float32,
    )
    output_sink(0, styled.shape[0], styled)
    return {
        "output": {
            "shape": list(styled.shape),
            "array_sha256": hashlib.sha256(styled.tobytes()).hexdigest(),
        }
    }


def _forbid_hardlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == "nt":
        monkeypatch.setattr(
            create_only_file.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("hardlinks unavailable")
            ),
        )


def test_standard_stage_and_png_publish_without_hardlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.film_physics.native_standard_staging as staging_module

    _forbid_hardlinks(monkeypatch)
    monkeypatch.setattr(
        staging_module,
        "render_native_standard_working_image_to_sink",
        _fake_renderer,
    )
    staged = stage_native_standard_working_image(
        object(),
        _working(),
        output_path=tmp_path / "standard.f32",
        report_path=tmp_path / "standard.json",
    )
    verified = verify_native_standard_staging(
        report_path=Path(staged["report_path"]),
        expected_report_sha256=staged["report_sha256"],
        expected_run_id=staged["run_id"],
    )
    published = commit_verified_native_standard_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=tmp_path / "standard.png",
        report_path=tmp_path / "standard-png.json",
    )

    assert verified["output_sha256"] == staged["output_sha256"]
    assert verify_native_standard_png16(
        report_path=Path(published["report_path"]),
        expected_report_sha256=published["report_sha256"],
        expected_delivery_id=published["delivery_id"],
    )["output_sha256"] == published["output_sha256"]


def test_strength_stage_and_png_publish_without_hardlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.film_physics.native_standard_strength_staging as staging_module

    _forbid_hardlinks(monkeypatch)
    monkeypatch.setattr(
        staging_module,
        "render_native_standard_working_image_to_sink",
        _fake_renderer,
    )
    staged = stage_native_standard_working_image_with_strength(
        object(),
        _working(),
        strength=0.75,
        output_path=tmp_path / "strength.f32",
        report_path=tmp_path / "strength.json",
    )
    verified = verify_native_standard_strength_staging(
        report_path=Path(staged["report_path"]),
        expected_report_sha256=staged["report_sha256"],
        expected_run_id=staged["run_id"],
    )
    published = commit_verified_native_standard_strength_png16(
        staging_report_path=Path(staged["report_path"]),
        expected_staging_report_sha256=staged["report_sha256"],
        expected_staging_run_id=staged["run_id"],
        output_path=tmp_path / "strength.png",
        report_path=tmp_path / "strength-png.json",
    )

    assert verified["strength"] == 0.75
    assert verify_native_standard_strength_png16(
        report_path=Path(published["report_path"]),
        expected_report_sha256=published["report_sha256"],
        expected_delivery_id=published["delivery_id"],
    )["output_sha256"] == published["output_sha256"]
