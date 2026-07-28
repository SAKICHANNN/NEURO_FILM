from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_native_spatial_conformance import (
    build_msvc_native_gaussian_dll,
    run_loaded_spatial_conformance,
)
from src.film_physics.native_spatial_profile import (
    build_native_gaussian_oracle,
    compile_native_gaussian_profile_payload,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio"
        r"\Installer\vswhere.exe"
    ).is_file(),
    reason="MSVC Build Tools are unavailable",
)
def test_msvc_native_spatial_abi_matches_scipy_oracle(
    tmp_path: Path,
) -> None:
    config = json.loads(
        (
            ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
        ).read_text()
    )
    artifact = compile_standalone_profile_artifact(
        root=ROOT, config=config
    )
    payload = compile_native_gaussian_profile_payload(artifact)
    oracle = build_native_gaussian_oracle(payload)
    build = build_msvc_native_gaussian_dll(
        root=ROOT, output_dir=tmp_path
    )
    result = run_loaded_spatial_conformance(
        dll_path=Path(build["dll_path"]),
        payload=payload,
        oracle=oracle,
    )
    assert result["status"] == "pass"
    assert result["invalid_input_buffers_unchanged"]
    assert result["overlap_rejected_before_write"]
    assert all(
        row["maximum_absolute_error"] <= row["tolerance"]
        for row in result["stages"]
    )
