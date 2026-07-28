from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_joint_ablation import load_contracts
from src.eval.physical_virtual_scan_sampling import (
    _render_candidate,
    compile_virtual_scan_profile,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p7d_virtual_scan_sampling_audit_v1.json"


def _config() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_p7d_contract_and_sampling_formula_are_exact() -> None:
    runtime_parent = validate_contract(ROOT, _config())
    assert runtime_parent["node"] == "U6.P7A3"
    _, runtime = load_contracts(ROOT, runtime_parent)
    for row in [
        *_config()["sampling_candidates"],
        _config()["negative_control"],
    ]:
        compiled = compile_virtual_scan_profile(
            runtime.profile, sampling_dpi=row["sampling_dpi"]
        )
        assert compiled.pixel_pitch_um == pytest.approx(
            row["pixel_pitch_um"], abs=1e-12
        )
        assert (
            compiled.forward_scatter_sigma_um_rgb
            == runtime.profile.forward_scatter_sigma_um_rgb
        )


def test_25400_dpi_compiles_to_the_exact_parent_profile() -> None:
    runtime_parent = validate_contract(ROOT, _config())
    _, runtime = load_contracts(ROOT, runtime_parent)
    assert (
        compile_virtual_scan_profile(runtime.profile, sampling_dpi=25_400)
        == runtime.profile
    )


def test_virtual_scan_candidate_is_repeat_exact_and_bounded() -> None:
    runtime_parent = validate_contract(ROOT, _config())
    _, runtime = load_contracts(ROOT, runtime_parent)
    source = np.random.default_rng(20260728).random((37, 41, 3))
    first = _render_candidate(source, runtime, sampling_dpi=4000)
    second = _render_candidate(source, runtime, sampling_dpi=4000)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert np.all((first[name] >= 0.0) & (first[name] <= 1.0))


@pytest.mark.parametrize("invalid", [True, 0, -1, 4000.0])
def test_sampling_compiler_rejects_invalid_dpi(invalid: object) -> None:
    runtime_parent = validate_contract(ROOT, _config())
    _, runtime = load_contracts(ROOT, runtime_parent)
    with pytest.raises(ValueError, match="positive integer"):
        compile_virtual_scan_profile(runtime.profile, sampling_dpi=invalid)
