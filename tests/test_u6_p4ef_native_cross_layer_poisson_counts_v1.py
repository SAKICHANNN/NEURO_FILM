from pathlib import Path

import pytest

from src.eval.native_cross_layer_poisson_conformance import evaluate

ROOT = Path(__file__).resolve().parents[1]
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM unavailable")
def test_p4ef_native_counts_are_exact(tmp_path: Path) -> None:
    report = evaluate(
        ROOT,
        ROOT / "configs/u6_p4ef_native_cross_layer_poisson_counts_v1.json",
        tmp_path,
        CLANG,
    )
    assert report["automatic_pass"] is True


def test_p4ef_parent_drift_fails_closed(tmp_path: Path) -> None:
    contract = ROOT / "configs/u6_p4ef_native_cross_layer_poisson_counts_v1.json"
    payload = contract.read_text(encoding="utf-8").replace(
        '"required_decision": "retain_native_cloud_typed_chain_parity_v1"',
        '"required_decision": "wrong"',
    )
    drifted = tmp_path / "contract.json"
    drifted.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent decision drift"):
        evaluate(ROOT, drifted, tmp_path / "out", CLANG)
