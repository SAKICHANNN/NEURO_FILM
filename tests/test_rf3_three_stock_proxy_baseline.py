from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.pipeline_color_baseline import load_guardrail_config, load_profile_values
from src.eval import three_stock_proxy_baseline as target

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "rf3_three_stock_proxy_baseline_d0_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_contract_preserves_claim_boundary() -> None:
    contract = target.load_contract(CONTRACT)
    assert contract["direction"]["ao6_role"] == "fujifilm_velvia_50_display_proxy_look_approximation_baseline_only"
    assert all(row["target_closeness_evaluable"] is False for row in contract["stock_evidence_matrix"])
    assert contract["deferred_baselines"]["latent_modes_or_k_greater_than_1"].startswith("not_admitted")


def test_ao6_cannot_be_relabelled_as_portra_or_ektar() -> None:
    contract = target.load_contract(CONTRACT)
    for stock in ("kodak_portra_400", "kodak_ektar_100"):
        changed = copy.deepcopy(contract)
        changed["ao6_velvia_baseline"]["allowed_stock_id"] = stock
        with pytest.raises(target.ThreeStockProxyError, match="AO6"):
            target._validate_contract_payload(changed)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_synthetic_evaluation_is_order_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = np.asarray([[[32, 64, 96], [128, 160, 192]], [[16, 80, 144], [48, 112, 176]]], dtype=np.uint8)
    ao6 = np.clip(source.astype(np.int16) + 18, 0, 255).astype(np.uint8)
    source_path = tmp_path / "source.png"
    ao6_path = tmp_path / "ao6.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(ao6).save(ao6_path)

    evidence = tmp_path / "evidence.json"
    _write_json(evidence, {"decision": "closed"})
    stats = tmp_path / "stats.json"
    _write_json(
        stats,
        {"styles": {name: {"mean": [50, 0, 0], "std": [10, 10, 10]} for name in ("velvia_50", "portra_400", "ektar_100")}},
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text("fixture", encoding="utf-8")
    guardrails = tmp_path / "guardrails.json"
    _write_json(guardrails, {})
    source_manifest = tmp_path / "sources.json"
    _write_json(
        source_manifest,
        [
            {
                "id": "sample",
                "make": "Fixture",
                "raw_sha256": "1" * 64,
                "decoded_path": "source.png",
                "decoded_sha256": _sha(source_path),
                "width": 2,
                "height": 2,
                "allowed_use": "test",
                "rights_scope": "test",
                "decoded_color_state": "test",
            }
        ],
    )
    ao6_manifest = tmp_path / "ao6_manifest.json"
    _write_json(
        ao6_manifest,
        {
            "records": [
                {
                    "candidate_id": "b0_plus_film_t15_c35",
                    "sample_id": "sample",
                    "decoded_source_sha256": _sha(source_path),
                    "output": "ao6.png",
                    "output_sha256": _sha(ao6_path),
                }
            ]
        },
    )

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["evidence_bindings"] = {"fixture": {"path": "evidence.json", "sha256": _sha(evidence), "required_decision": "closed"}}
    contract["source"].update(
        {
            "manifest": "sources.json",
            "manifest_sha256": _sha(source_manifest),
            "included_ids": ["sample"],
            "expected_rows": 1,
            "required_allowed_use": "test",
            "required_rights_scope": "test",
            "required_color_state": "test",
        }
    )
    contract["legacy_k1_operator"]["stats"] = {"path": "stats.json", "sha256": _sha(stats)}
    contract["legacy_k1_operator"]["profile"] = {"path": "profile.yaml", "sha256": _sha(profile), "preset": "safe-rich"}
    contract["legacy_k1_operator"]["guardrails"] = {"path": "guardrails.json", "sha256": _sha(guardrails)}
    contract["ao6_velvia_baseline"].update({"run_manifest": "ao6_manifest.json", "run_manifest_sha256": _sha(ao6_manifest)})
    contract["comparison"].update(
        {
            "minimum_pairwise_population_median_delta_e76": 0.0,
            "minimum_ao6_vs_legacy_velvia_population_median_delta_e76": 0.0,
            "maximum_new_output_boundary_fraction": 1.0,
            "maximum_output_boundary_fraction": 1.0,
        }
    )
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)

    offsets = {"velvia_50": 0.04, "portra_400": 0.08, "ektar_100": 0.12}
    monkeypatch.setattr(target, "load_profile_values", lambda *_args: {})
    monkeypatch.setattr(target, "load_guardrail_config", lambda *_args: {})
    monkeypatch.setattr(
        target,
        "render_resolved_safe_lab_rgb",
        lambda rgb, *, style, **_kwargs: np.clip(
            rgb + offsets[style], 0.0, 1.0
        ).astype(np.float32),
    )

    canonical = target.evaluate(contract_path, tmp_path, tmp_path / "run_a", "canonical")
    reverse = target.evaluate(contract_path, tmp_path, tmp_path / "run_b", "reverse")
    left = dict(canonical)
    right = dict(reverse)
    left.pop("order")
    right.pop("order")
    assert left == right
    assert canonical["scientific_identity"] == reverse["scientific_identity"]
    assert canonical["target_closeness"] == "not_evaluable_without_controlled_stock_targets"
    assert canonical["stock_distinguishability"] == "not_evaluable_without_controlled_stock_targets"


def test_missing_gate_does_not_pass() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    del contract["comparison"]["minimum_pairwise_population_median_delta_e76"]
    with pytest.raises(target.ThreeStockProxyError, match="gate is missing"):
        target._validate_contract_payload(contract)


def test_public_engine_replays_all_frozen_three_stock_proxy_pixels() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    report_path = (
        ROOT
        / "outputs"
        / "eval"
        / "rf3_three_stock_proxy_baseline_d0_v1"
        / "run_a"
        / "report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stats = json.loads(
        (ROOT / contract["legacy_k1_operator"]["stats"]["path"]).read_text(
            encoding="utf-8"
        )
    )["styles"]
    profile_path = ROOT / contract["legacy_k1_operator"]["profile"]["path"]
    preset = contract["legacy_k1_operator"]["profile"]["preset"]
    guardrail_path = ROOT / contract["legacy_k1_operator"]["guardrails"]["path"]
    source_manifest = json.loads(
        (ROOT / contract["source"]["manifest"]).read_text(encoding="utf-8")
    )
    sources = {row["id"]: row for row in source_manifest}

    for row in report["rows"]:
        source = target._load_rgb8(ROOT / sources[row["source_id"]]["decoded_path"])
        for arm in contract["legacy_k1_operator"]["arms"]:
            style = arm["style"]
            parameters = load_profile_values(profile_path, preset, style)
            parameters.update({"grain": 0.0, "dither": 0.0})
            rendered = target.render_resolved_safe_lab_rgb(
                source.astype(np.float32) / 255.0,
                style=style,
                style_statistics=stats[style],
                style_parameters=parameters,
                guardrails=load_guardrail_config(guardrail_path, style),
                seed=1729,
            )
            actual = np.rint(rendered * 255.0).astype(np.uint8)
            frozen = target._load_rgb8(
                report_path.parent / row["outputs"][arm["arm_id"]]["relative_path"]
            )
            np.testing.assert_array_equal(actual, frozen)
