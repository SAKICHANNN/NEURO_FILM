from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.audit_u7_6i_three_stock_native_pointwise import _python_pointwise
from scripts.audit_u7_6j_three_stock_native_pointwise_exact import (
    CONTRACT_SCHEMA,
    ROOT,
    _load_contract,
)
from src.color_engine.safe_lab import safe_lab_context_from_lab
from src.eval.native_safe_lab_pointwise_v3 import (
    NativeSafeLabPointwiseV3Error,
    apply_native_safe_lab_pointwise_v3,
    build_native_safe_lab_pointwise_v3,
    load_native_safe_lab_pointwise_v3,
)


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory):
    build = build_native_safe_lab_pointwise_v3(
        root=ROOT, output_dir=tmp_path_factory.mktemp("u7_6j_native")
    )
    return load_native_safe_lab_pointwise_v3(build["dll_path"])


def _boundary_source() -> np.ndarray:
    return np.asarray(
        [
            [
                [77.235916, 0.31656027, -14.676989],
                [73.224915, -1.129508, -14.515758],
                [74.78917, -2.0660162, -14.22739],
                [55.0, 16.0, 22.0],
            ],
            [
                [81.74673, -2.8787851, -14.161563],
                [55.163345, 1.3277893, -30.644098],
                [64.70445, 0.3861189, -14.48648],
                [32.0, -18.0, 12.0],
            ],
        ],
        dtype=np.float32,
    )


def _parameters(source: np.ndarray) -> tuple[dict, dict]:
    context = safe_lab_context_from_lab(source)
    common = {
        "source_context": context,
        "destination_mean": np.asarray(
            [41.78746796, 2.28838992, -3.78775144], dtype=np.float32
        ),
        "destination_std": np.asarray(
            [26.84054565, 12.73180771, 23.40808678], dtype=np.float32
        ),
    }
    style = {"strength": 0.35, "luma_strength": 0.02, "chroma_curve_strength": 0.45}
    guard = {
        "neutral_protect": 0.55,
        "skin_protect": 0.35,
        "max_chroma_gain": 2.1,
        "max_chroma_boost": 16.0,
        "max_chroma_absolute": None,
    }
    return common, {"style": style, "guard": guard}


def test_u7_6j_contract_is_frozen_and_bound() -> None:
    path = ROOT / "configs/u7_6j_three_stock_native_pointwise_exact_v1.json"
    contract, digest = _load_contract(path)
    assert contract["schema"] == CONTRACT_SCHEMA
    assert contract["experiment_id"] == "U7.6J"
    assert len(digest) == 64
    assert contract["gates"]["maximum_lab_absolute_error"] == 0.0
    assert contract["styles"] == ["velvia_50", "portra_400", "ektar_100"]


def test_u7_6j_contract_preserves_parent_failure() -> None:
    contract = json.loads(
        (ROOT / "configs/u7_6j_three_stock_native_pointwise_exact_v1.json").read_text(
            "utf-8"
        )
    )
    parent = json.loads(
        (ROOT / contract["parent_negative"]["evidence_path"]).read_text("utf-8")
    )
    assert parent["status"] == "FAIL_CLOSED_EXACT_PARITY"
    assert parent["gates"]["maximum_lab_absolute_error"] is False
    assert contract["execution"]["full_renderer_integration_allowed"] is False


def test_u7_6j_matches_binary64_guardrail_boundary(native_library) -> None:
    source = _boundary_source()
    common, oracle_parameters = _parameters(source)
    context = common["source_context"]
    expected = _python_pointwise(
        source,
        source_mean=np.asarray(context.lab_mean, dtype=np.float32),
        source_std=np.asarray(context.lab_std, dtype=np.float32),
        destination_mean=common["destination_mean"],
        destination_std=common["destination_std"],
        **oracle_parameters,
    )
    actual = apply_native_safe_lab_pointwise_v3(
        native_library,
        source,
        **common,
        **oracle_parameters["style"],
        **oracle_parameters["guard"],
        thread_count=4,
    )
    assert np.array_equal(actual, expected)


def test_u7_6j_nonfinite_failure_is_atomic(native_library) -> None:
    source = _boundary_source()
    common, parameters = _parameters(source)
    source[0, 0, 0] = np.nan
    output = np.full_like(source, 17.0)
    with pytest.raises(NativeSafeLabPointwiseV3Error):
        apply_native_safe_lab_pointwise_v3(
            native_library,
            source,
            **common,
            **parameters["style"],
            **parameters["guard"],
            output=output,
        )
    assert np.all(output == 17.0)
