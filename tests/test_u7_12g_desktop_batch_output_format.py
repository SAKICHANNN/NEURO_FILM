from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow
from src.inference.render_contract import verify_render_recipe_files
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:37, :53]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 2) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(tmp_path: Path, **kwargs: object) -> ProductDesktopWorkflow:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=3_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
        **kwargs,  # type: ignore[arg-type]
    )


def test_contract_freezes_versioned_formats_and_claim_ceiling() -> None:
    contract = json.loads(
        (ROOT / "configs/u7_12g_desktop_batch_output_format_v1.json").read_text("utf-8")
    )
    assert contract["product_contract"]["png16_legacy_receipt_and_member_bytes_exact"]
    assert contract["product_contract"]["recovery_core_remains_png16_only"]
    assert contract["formats"]["png16"]["legacy_receipt_schema"].endswith(".v1")
    assert contract["formats"]["tiff16"]["receipt_schema"].endswith(".v2")
    assert contract["formats"]["jpeg8"]["receipt_schema"].endswith(".v2")
    assert contract["claim_ceiling"]["calibrated_stock_response"] is False
    assert contract["claim_ceiling"]["physical_film_reproduction"] is False


def test_unknown_batch_format_rejects_before_stage_or_renderer(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _source(first, 1)
    _source(second, 2)
    calls = 0

    def forbidden(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess([], 1, "", "forbidden")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    calls = 0
    destination = tmp_path / "batch"
    with pytest.raises(ProductDesktopError, match="unknown desktop output format"):
        workflow.export_batch(
            bound,
            "ektar_100",
            destination,
            output_format_id="unknown",
        )
    assert calls == 0
    assert not destination.exists()
    assert not list(tmp_path.glob(".batch.u7-11a-*.stage"))
    workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal directory publication is Windows-only"
)
@pytest.mark.parametrize(
    ("format_id", "suffix", "recipe_format", "bit_depth"),
    [
        ("tiff16", ".tiff", "TIFF", 16),
        ("jpeg8", ".jpg", "JPEG", 8),
    ],
)
def test_non_png_batch_matches_direct_cli_and_strict_replay(
    tmp_path: Path,
    format_id: str,
    suffix: str,
    recipe_format: str,
    bit_depth: int,
) -> None:
    first = tmp_path / "B source.png"
    second = tmp_path / "a source.png"
    _source(first, 1)
    _source(second, 2)
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews((first, second), 0.625)
    destination = tmp_path / f"{format_id}-batch"
    receipt = workflow.export_batch(
        bound,
        "ektar_100",
        destination,
        output_format_id=format_id,
    )

    assert receipt.output_format_id == format_id
    assert receipt.receipt["schema_version"].endswith(".v2")
    assert receipt.receipt["output_format_id"] == format_id
    assert receipt.receipt["output_format"] == recipe_format
    assert receipt.receipt["output_bit_depth"] == bit_depth
    assert len(list(destination.glob(f"*{suffix}"))) == 2

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }
    for index, (source, row) in enumerate(zip(bound, receipt.receipt["jobs"]), 1):
        output = destination / row["output_path"]
        recipe_path = destination / row["recipe_path"]
        recipe = json.loads(recipe_path.read_text("utf-8"))
        assert output.suffix == suffix
        assert recipe["output"]["format"] == recipe_format
        assert recipe["output"]["bit_depth"] == bit_depth
        verify_render_recipe_files(
            recipe,
            profile_path=ROOT / "configs/render_profiles/safe_rich_product_v1.json",
            root=ROOT,
        )

        direct = tmp_path / f"direct-{index}{suffix}"
        completed = subprocess.run(
            list(
                workflow._build_export_command(
                    source.path,
                    "ektar_100",
                    direct,
                    0.625,
                    output_format_id=format_id,
                )
            ),
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert direct.read_bytes() == output.read_bytes()

        replay = tmp_path / f"replay-{index}{suffix}"
        replayed = replay_style_safe_recipe_to_file(
            recipe,
            profile_path=ROOT / "configs/render_profiles/safe_rich_product_v1.json",
            output_path=replay,
            root=ROOT,
        )
        assert replayed == row["output_sha256"]
        assert replay.read_bytes() == output.read_bytes()
    workflow.close()


def test_png16_default_keeps_legacy_receipt_shape(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _source(first, 1)
    _source(second, 2)
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews((first, second), 0.5)
    destination = tmp_path / "png-batch"
    receipt = workflow.export_batch(bound, "portra_400", destination)
    assert receipt.output_format_id == "png16"
    assert receipt.receipt["schema_version"].endswith(".v1")
    assert receipt.receipt["output_format"] == "PNG"
    assert receipt.receipt["output_bit_depth"] == 16
    assert "output_format_id" not in receipt.receipt
    assert all(row["output_path"].endswith(".png") for row in receipt.receipt["jobs"])
    workflow.close()
