from __future__ import annotations

import pytest

from scripts.benchmark_u6_p6zg_scanner_glare_typed_chain import _probe, worker


@pytest.mark.parametrize(
    "algorithm",
    ["typed-chain-full-fft-glare", "typed-chain-channel-serial-glare"],
)
def test_small_worker_is_repeat_exact(algorithm: str) -> None:
    first = worker(algorithm, (31, 37, 3), seed=6206106)
    second = worker(algorithm, (31, 37, 3), seed=6206106)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["finite_bounded"] is True


def test_probe_matches_typed_reference() -> None:
    probe = _probe(6206106)
    assert probe["maximum_absolute_error"] <= 1e-11
    assert probe["rmse"] <= 1e-12


def test_worker_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        worker("unknown", (7, 9, 3), seed=6206106)
