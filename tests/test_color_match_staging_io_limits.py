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


def test_runtime_collector_rejects_reparse_before_any_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("runtime-qualified path must not call Path.resolve")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    path = tmp_path / "output.png"

    assert runtime_staging._runtime_output_paths([path], count=1) == (path,)


def test_runtime_lock_checks_original_temp_root_without_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("runtime lock root must not call Path.resolve")

    monkeypatch.setattr(
        runtime_staging.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(Path, "resolve", forbidden_resolve)

    with runtime_staging._target_transaction_lock(
        (tmp_path / "destination.png",)
    ):
        pass
