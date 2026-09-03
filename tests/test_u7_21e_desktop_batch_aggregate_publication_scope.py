from __future__ import annotations

from pathlib import Path

import pytest

import src.inference.product_desktop as desktop_module
from src.inference.product_desktop import ProductDesktopError
from tests.test_u7_11a_desktop_single_look_batch import _inputs, _workflow


@pytest.mark.skipif(
    not desktop_module._batch_directory_rename_supported(),
    reason="formal directory publication is Windows-only",
)
def test_scope_drift_after_aggregate_receipt_rejects_before_directory_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _inputs(tmp_path)
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews((first, second), 0.625)
    destination = tmp_path / "batch"
    aggregate_staged = False
    original_write = desktop_module._write_bound_json

    def write_then_drift(path, payload):  # type: ignore[no-untyped-def]
        nonlocal aggregate_staged
        seal = original_write(path, payload)
        if Path(path).name == "batch.json":
            aggregate_staged = True
        return seal

    def reject_after_aggregate(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        if aggregate_staged:
            raise ProductDesktopError("runtime source scope changed")

    monkeypatch.setattr(desktop_module, "_write_bound_json", write_then_drift)
    monkeypatch.setattr(
        desktop_module, "_validate_runtime_source_scope", reject_after_aggregate
    )

    with pytest.raises(ProductDesktopError, match="runtime source scope changed"):
        workflow.export_batch(bound, "ektar_100", destination)
    assert aggregate_staged is True
    assert not destination.exists()
    assert not list(tmp_path.glob(".batch.u7-11a-*.stage"))
    assert workflow.close()
