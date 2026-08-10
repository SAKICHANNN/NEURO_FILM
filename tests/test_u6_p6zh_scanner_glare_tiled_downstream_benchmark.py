from __future__ import annotations

import pytest

from scripts.benchmark_u6_p6zh_scanner_glare_tiled_downstream import worker


@pytest.mark.parametrize("algorithm", ["full-downstream", "tiled-downstream"])
def test_small_worker_is_repeat_exact(algorithm: str) -> None:
    first = worker(algorithm, (31, 37, 3), seed=6206106)
    second = worker(algorithm, (31, 37, 3), seed=6206106)
    assert first["source_sha256"] == second["source_sha256"]
    assert first["output_sha256"] == second["output_sha256"]
    assert first["finite_bounded"] is True


def test_worker_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        worker("unknown", (7, 9, 3), seed=6206106)
