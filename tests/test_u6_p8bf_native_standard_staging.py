from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pytest

from src.film_physics.native_standard_factory import (
    create_opt_in_native_standard_runtime,
)
from src.film_physics.native_standard_package import (
    resolve_native_standard_libraries,
)
from src.film_physics.native_standard_staging import (
    stage_native_standard_working_image,
    verify_native_standard_staging,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)
from src.preprocess.types import SourceProfile, WorkingImage
from tests.test_u6_p8ba_native_standard_runtime import (
    PACKAGE,
    PROFILE_CONFIG,
    ROOT,
    _runtime_fixture,
)

DECISION = (
    ROOT / "configs/u6_p8bf_native_standard_staging_decision_v1.json"
)


def _fixture(tmp_path: Path) -> tuple[object, WorkingImage]:
    _, paths = _runtime_fixture(tmp_path)
    package = json.loads(PACKAGE.read_text())
    artifact = compile_standalone_profile_artifact(
        root=ROOT,
        config=json.loads(PROFILE_CONFIG.read_text()),
    )
    runtime, _ = create_opt_in_native_standard_runtime(
        package=package,
        artifact=artifact,
        library_paths=paths,
    )
    working = WorkingImage(
        pixels=np.random.default_rng(2026072929).random(
            (33, 35, 3), dtype=np.float32
        ),
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
    return runtime, working


def test_p8bf_stage_and_restart_verify_are_exact(
    tmp_path: Path,
) -> None:
    runtime, working = _fixture(tmp_path)
    output = tmp_path / "candidate.f32"
    report = tmp_path / "candidate.json"
    committed = stage_native_standard_working_image(
        runtime,
        working,
        output_path=output,
        report_path=report,
    )
    first = verify_native_standard_staging(
        report_path=report,
        expected_report_sha256=committed["report_sha256"],
        expected_run_id=committed["run_id"],
    )
    second = verify_native_standard_staging(
        report_path=report,
        expected_report_sha256=committed["report_sha256"],
        expected_run_id=committed["run_id"],
    )
    assert first == second
    assert first["state"] == "verified-native-standard-staging"
    assert output.stat().st_size == 33 * 35 * 3 * 4

    original = output.read_bytes()
    output.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="output drift"):
        verify_native_standard_staging(
            report_path=report,
            expected_report_sha256=committed["report_sha256"],
            expected_run_id=committed["run_id"],
        )


def test_p8bf_create_only_and_failure_leave_no_artifacts(
    tmp_path: Path,
) -> None:
    runtime, working = _fixture(tmp_path)
    output = tmp_path / "candidate.f32"
    report = tmp_path / "candidate.json"
    output.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        stage_native_standard_working_image(
            runtime,
            working,
            output_path=output,
            report_path=report,
        )
    assert output.read_bytes() == b"owned"
    assert not report.exists()

    output.unlink()
    original_render = runtime.render_to_sink

    def fail_after_one(
        self: object, source: np.ndarray, *, output_sink: object
    ) -> dict[str, object]:
        output_sink(0, 1, np.zeros((1, 35, 3), dtype=np.float32))
        raise RuntimeError("injected runtime failure")

    runtime.render_to_sink = types.MethodType(fail_after_one, runtime)
    with pytest.raises(RuntimeError, match="injected runtime failure"):
        stage_native_standard_working_image(
            runtime,
            working,
            output_path=output,
            report_path=report,
        )
    runtime.render_to_sink = original_render
    assert not output.exists()
    assert not report.exists()
    assert not list(tmp_path.glob(".*.stage"))


def test_p8bf_decision_binds_staging_source_and_claim_ceiling() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["result"]["status"] == (
        "pass-local-staging-boundary"
    )
    assert decision["result"]["report_is_commit_marker"]
    assert not decision["result"]["final_quantization_performed"]
    assert not decision["result"]["delivery_defined"]
    assert hashlib.sha256(
        (ROOT / decision["implementation"]).read_bytes()
    ).hexdigest() == decision["implementation_sha256"]
    assert decision["next_leaf"].startswith("U6.P8BG")
