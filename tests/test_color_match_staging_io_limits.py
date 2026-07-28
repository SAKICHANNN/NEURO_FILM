from __future__ import annotations

from pathlib import Path

import pytest

import src.color_match.shared_runtime_staging_transaction as runtime_staging
from src.color_match import ReferenceMatchContractError
from src.color_match.staging_io import staging_output_paths


@pytest.mark.parametrize(
    "collector",
    [
        staging_output_paths,
        runtime_staging._runtime_output_paths,
    ],
)
def test_staging_path_collectors_do_not_exhaust_unbounded_iterables(
    collector,
    tmp_path: Path,
) -> None:
    expected_count = 2
    pulls = 0

    def unbounded_paths():
        nonlocal pulls
        while True:
            pulls += 1
            if pulls > expected_count + 1:
                raise AssertionError("staging collector over-consumed paths")
            yield tmp_path / f"output-{pulls}.png"

    with pytest.raises(
        ReferenceMatchContractError,
        match="output path count must match authorized sources",
    ):
        collector(unbounded_paths(), count=expected_count)

    assert pulls == expected_count + 1
