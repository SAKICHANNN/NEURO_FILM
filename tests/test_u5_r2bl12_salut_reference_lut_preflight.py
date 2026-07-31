from __future__ import annotations

from pathlib import Path

from src.eval.salut_reference_lut_preflight import (
    cpu_forward_max_index,
    cuda_forward_max_index,
    dense_attention_bytes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_published_cuda_index_matches_two_context_planes() -> None:
    dim = 17
    maximum = cuda_forward_max_index(
        dim=dim, context=0.5, red=0.5, green=0.5, blue=0.5
    )
    assert maximum < 2 * dim**3


def test_published_cpu_index_is_not_the_cuda_contract() -> None:
    dim = 17
    args = dict(dim=dim, context=0.5, red=0.5, green=0.5, blue=0.5)
    cuda_maximum = cuda_forward_max_index(**args)
    cpu_maximum = cpu_forward_max_index(**args)
    assert cpu_maximum != cuda_maximum
    assert cpu_maximum >= 2 * dim**3


def test_published_attention_exceeds_local_vram_before_other_tensors() -> None:
    assert dense_attention_bytes(height=512, width=512) == 17_179_869_184
    assert dense_attention_bytes(height=512, width=512) > 12 * 1024**3


def test_frozen_config_keeps_quality_comparison_closed_on_runtime_failure() -> None:
    text = (
        ROOT / "configs/u5_r2bl12_salut_reference_lut_preflight_v1.json"
    ).read_text(encoding="utf-8")
    assert "close direct reproduction" in text
    assert '"product_integration_allowed": false' in text
