from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.physical_native_ordered_chain_conformance import (
    build_msvc_native_adjacency_dll,
    build_msvc_native_domains_dll,
    build_msvc_native_gaussian_dll,
    run_loaded_ordered_chain,
)
from src.film_physics.native_adjacency_profile import (
    build_native_ordered_chain_oracle,
    compile_native_adjacency_profile_payload,
    validate_native_adjacency_profile_payload,
)
from src.film_physics.native_profile import (
    compile_native_domains_profile_payload,
)
from src.film_physics.native_spatial_profile import (
    compile_native_gaussian_profile_payload,
)
from src.film_physics.profile_consumer import (
    compile_standalone_profile_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _artifact() -> dict:
    config = json.loads(
        (
            ROOT / "configs/u6_p8b_artifact_only_cpu_consumer_v1.json"
        ).read_text()
    )
    return compile_standalone_profile_artifact(root=ROOT, config=config)


def test_native_adjacency_payload_is_exact_and_fail_closed() -> None:
    artifact = _artifact()
    payload = compile_native_adjacency_profile_payload(artifact)
    assert payload == compile_native_adjacency_profile_payload(artifact)
    forged = copy.deepcopy(payload)
    forged["maximum_absolute_density_delta"] = 0.0
    with pytest.raises(ValueError, match="parameter"):
        validate_native_adjacency_profile_payload(forged)
    forged = copy.deepcopy(payload)
    forged["source_component"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate_native_adjacency_profile_payload(
            forged, artifact=artifact
        )


@pytest.mark.skipif(
    not Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio"
        r"\Installer\vswhere.exe"
    ).is_file(),
    reason="MSVC Build Tools are unavailable",
)
def test_msvc_ordered_chain_matches_all_python_stages(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    domains_payload = compile_native_domains_profile_payload(artifact)
    spatial_payload = compile_native_gaussian_profile_payload(artifact)
    adjacency_payload = compile_native_adjacency_profile_payload(artifact)
    oracle = build_native_ordered_chain_oracle(
        artifact, adjacency_payload
    )
    domains = build_msvc_native_domains_dll(
        root=ROOT, output_dir=tmp_path / "domains"
    )
    gaussian = build_msvc_native_gaussian_dll(
        root=ROOT, output_dir=tmp_path / "gaussian"
    )
    adjacency = build_msvc_native_adjacency_dll(
        root=ROOT, output_dir=tmp_path / "adjacency"
    )
    result = run_loaded_ordered_chain(
        domains_dll=Path(domains["dll_path"]),
        gaussian_dll=Path(gaussian["dll_path"]),
        adjacency_dll=Path(adjacency["dll_path"]),
        domains_payload=domains_payload,
        spatial_payload=spatial_payload,
        adjacency_payload=adjacency_payload,
        oracle=oracle,
    )
    assert result["status"] == "pass"
    assert [row["stage"] for row in result["stages"]] == [
        "forward_scatter",
        "developed_density",
        "bounded_adjacency",
        "dye_diffusion",
        "interpretation",
        "scanner_mtf",
    ]
    assert all(
        row["maximum_absolute_error"] <= row["tolerance"]
        for row in result["stages"]
    )
