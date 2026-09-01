from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import (
    ProductDesktopError,
    ProductDesktopWorkflow,
)

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:31, :47]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 2) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(tmp_path: Path, name: str, **kwargs: object) -> ProductDesktopWorkflow:
    scratch = tmp_path / name
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=2_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
        **kwargs,  # type: ignore[arg-type]
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_contract_freezes_optional_explicit_representative() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u7_15a_desktop_batch_representative_selection_v1.json"
        ).read_text("utf-8")
    )
    assert contract["interface"] == {
        "method": "ProductDesktopWorkflow.render_batch_previews",
        "new_keyword": "representative_path",
        "default": None,
        "default_semantics": "canonical-first-exact",
        "explicit_semantics": "one-exact-bound-member",
        "canonical_batch_order_unchanged": True,
        "receipt_schema_unchanged": True,
    }
    assert "automatic aesthetic routing" in contract["claim_ceiling"]


@pytest.mark.skipif(os.name != "nt", reason="batch publication is Windows-only")
def test_explicit_nonfirst_preview_preserves_canonical_batch_bytes(
    tmp_path: Path,
) -> None:
    later = tmp_path / "z-later.png"
    canonical_first = tmp_path / "a-first.png"
    _source(later, 1)
    _source(canonical_first, 2)
    selected = (later, canonical_first)
    destination = tmp_path / "batch"

    default = _workflow(tmp_path, "default-scratch")
    default_state, default_bound = default.render_batch_previews(selected, 0.625)
    assert default_state.input_path == canonical_first.resolve()
    default_receipt = default.export_batch(default_bound, "portra_400", destination)
    default_tree = _tree_bytes(destination)
    default.close()
    shutil.rmtree(destination)

    explicit = _workflow(tmp_path, "explicit-scratch")
    explicit_state, explicit_bound = explicit.render_batch_previews(
        selected, 0.625, representative_path=later
    )
    assert explicit_state.input_path == later.resolve()
    assert tuple(row.path for row in explicit_bound) == tuple(
        row.path for row in default_bound
    )
    explicit_receipt = explicit.export_batch(explicit_bound, "portra_400", destination)
    assert explicit_receipt.receipt_sha256 == default_receipt.receipt_sha256
    assert explicit_receipt.receipt == default_receipt.receipt
    assert _tree_bytes(destination) == default_tree
    explicit.close()


def test_invalid_or_changed_representative_rejects_before_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    external = tmp_path / "external.png"
    _source(first, 1)
    _source(second, 2)
    _source(external, 3)
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("preview renderer must not run")

    workflow = _workflow(tmp_path, "scratch", preview_renderer=forbidden)
    with pytest.raises(ProductDesktopError, match="one selected photo"):
        workflow.render_batch_previews(
            (first, second), 0.5, representative_path=external
        )
    with pytest.raises(ProductDesktopError, match="one selected photo"):
        workflow.render_batch_previews(
            (first, second), 0.5, representative_path=tmp_path / "missing.png"
        )

    stale = workflow.bind_batch_inputs((first, second))

    def bind_then_mutate(_paths: object) -> object:
        _source(second, 19)
        return stale

    monkeypatch.setattr(workflow, "bind_batch_inputs", bind_then_mutate)
    with pytest.raises(ProductDesktopError, match="changed before preview"):
        workflow.render_batch_previews((first, second), 0.5, representative_path=second)
    assert calls == 0
    assert workflow.preview_state is None
