from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import dng_three_stock_proxy as target
from src.preprocess.types import SourceProfile, WorkingImage

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "rf3_d0r_dng_three_stock_proxy_v1.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _working(pixels: np.ndarray) -> WorkingImage:
    return WorkingImage(
        pixels=pixels,
        working_space="linear_rec2020",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("fixture", "fixture"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("fixture.dng"),
    )


def test_real_contract_keeps_ao6_velvia_only_and_claims_proxy_mechanics() -> None:
    contract = target.load_contract(CONTRACT, ROOT)
    assert contract["arms"][-1] == {
        "arm_id": "ao6_velvia50_display_proxy_t15_c35",
        "stock_id": "fujifilm_velvia_50",
        "role": "velvia_display_proxy_baseline_only",
    }
    assert "No photographic quality" in contract["claim_ceiling"]
    assert "stock distinguishability" in contract["claim_ceiling"]


@pytest.mark.parametrize("gate_value", [None, float("nan"), float("inf")])
def test_contract_rejects_missing_or_nonfinite_gate(tmp_path: Path, gate_value: float | None) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    key = "minimum_pairwise_population_median_delta_e76"
    if gate_value is None:
        del contract["gates"][key]
    else:
        contract["gates"][key] = gate_value
    path = tmp_path / "contract.json"
    _write_json(path, contract)
    with pytest.raises(target.DngThreeStockProxyError, match="gate missing or non-finite"):
        target.load_contract(path, ROOT)


def _install_synthetic_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *, nonfinite_style: str | None = None,
) -> tuple[Path, dict[str, object]]:
    p98 = tmp_path / "p98.json"
    _write_json(
        p98,
        {
            "rows": [
                {
                    "source_id": "a",
                    "camera_make": "A",
                    "logical_path": "a.dng",
                    "source_bytes": 1,
                    "source_sha256": "a" * 64,
                },
                {
                    "source_id": "b",
                    "camera_make": "B",
                    "logical_path": "b.dng",
                    "source_bytes": 1,
                    "source_sha256": "b" * 64,
                },
            ]
        },
    )
    stats = tmp_path / "stats.json"
    _write_json(stats, {"styles": {name: {} for name in ("velvia_50", "portra_400", "ektar_100")}})
    compiler = tmp_path / "compiler.json"
    _write_json(compiler, {})
    placeholder = tmp_path / "placeholder.json"
    _write_json(placeholder, {})

    bindings = {
        "p98_contract": {"path": "p98.json"},
        "legacy_stats": {"path": "stats.json"},
        "legacy_profiles": {"path": "placeholder.json"},
        "legacy_guardrails": {"path": "placeholder.json"},
        "ao6_profile_compiler": {"path": "compiler.json"},
    }
    contract: dict[str, object] = {
        "experiment_id": "fixture",
        "bindings": bindings,
        "input": {
            "working_space": "linear_rec2020",
            "transfer_state": "scene_linear",
            "display_target": "sdr_rec709",
            "expected_rows": 2,
            "expected_camera_makes": 2,
        },
        "arms": [
            {"arm_id": "velvia", "style": "velvia_50"},
            {"arm_id": "portra", "style": "portra_400"},
            {"arm_id": "ektar", "style": "ektar_100"},
            {"arm_id": "ao6"},
        ],
        "gates": {
            "minimum_pairwise_population_median_delta_e76": 0.0,
            "minimum_ao6_vs_legacy_velvia_population_median_delta_e76": 0.0,
            "maximum_output_code_boundary_fraction": 1.0,
            "maximum_new_output_code_boundary_fraction": 1.0,
        },
        "decision_if_pass": "pass",
        "decision_if_fail": "fail",
        "claim_ceiling": "fixture",
    }
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)

    lookup = {binding["path"]: tmp_path / str(binding["path"]) for binding in bindings.values()}
    monkeypatch.setattr(target, "load_contract", lambda *_args: copy.deepcopy(contract))
    monkeypatch.setattr(target, "_bound", lambda _root, binding: lookup[binding["path"]])
    base = np.asarray(
        [[[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]], [[0.15, 0.25, 0.35], [0.45, 0.55, 0.65]]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        target,
        "load_dng_forward_working_image",
        lambda path, **_kwargs: _working(base + (0.01 if path.name == "b.dng" else 0.0)),
    )
    monkeypatch.setattr(target, "apply_working_image_aces2_output", lambda working, _target: working.pixels.copy())
    monkeypatch.setattr(target, "load_profile_values", lambda *_args: {})
    monkeypatch.setattr(target, "load_guardrail_config", lambda *_args: {})
    offsets = {"velvia_50": 0.01, "portra_400": 0.03, "ektar_100": 0.05}

    def transfer(rgb: np.ndarray, _stats: object, style: str, **_kwargs: object) -> np.ndarray:
        if style == nonfinite_style:
            return np.full_like(rgb, np.nan)
        return np.clip(rgb + offsets[style], 0.0, 1.0).astype(np.float32)

    monkeypatch.setattr(target, "style_transfer_rgb", transfer)
    monkeypatch.setattr(target, "compile_standalone_profile_artifact", lambda **_kwargs: {"component_payloads": {"ao6-source-context-display-look": {}}})
    monkeypatch.setattr(
        target,
        "build_source_context_display_look_stages",
        lambda _payload, _source: (lambda value: value, lambda value: np.clip(value + 0.08, 0.0, 1.0)),
    )
    return contract_path, contract


def _inventory(path: Path) -> list[tuple[str, str]]:
    return sorted(
        (item.relative_to(path).as_posix(), hashlib.sha256(item.read_bytes()).hexdigest())
        for item in (path / "renders").rglob("*.png")
    )


def test_synthetic_evaluation_is_order_and_rgb16_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract_path, _ = _install_synthetic_runtime(monkeypatch, tmp_path)
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    canonical = target.evaluate(contract_path, tmp_path, run_a, "canonical")
    reverse = target.evaluate(contract_path, tmp_path, run_b, "reverse")
    left = dict(canonical)
    right = dict(reverse)
    left.pop("execution_order")
    right.pop("execution_order")
    assert left == right
    assert canonical["stable_evidence_id"] == reverse["stable_evidence_id"]
    assert canonical["decision"] == "pass"
    assert len(_inventory(run_a)) == 8
    assert _inventory(run_a) == _inventory(run_b)


def test_nonfinite_proxy_fails_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract_path, _ = _install_synthetic_runtime(monkeypatch, tmp_path, nonfinite_style="portra_400")
    output = tmp_path / "run"
    with pytest.raises(target.DngThreeStockProxyError, match="non-finite or unbounded"):
        target.evaluate(contract_path, tmp_path, output, "canonical")
    assert not (output / "report.json").exists()
