from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.product_execution_policy import (
    PRODUCT_DEFAULT_TILE_SIZE,
    PRODUCT_DEFAULT_TILE_WORKERS,
    resolve_product_execution_policy,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"


def _source(path: Path) -> None:
    y, x = np.mgrid[:19, :23]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_fixed_product_default_and_explicit_override_are_distinct() -> None:
    automatic = resolve_product_execution_policy(
        profile_id="safe-rich-product-v1", tile_size=None, tile_workers=1
    )
    assert automatic.tile_size == PRODUCT_DEFAULT_TILE_SIZE == 256
    assert automatic.tile_workers == PRODUCT_DEFAULT_TILE_WORKERS == 1
    assert automatic.automatic is True

    explicit = resolve_product_execution_policy(
        profile_id="safe-rich-product-v1", tile_size=37, tile_workers=3
    )
    assert (explicit.tile_size, explicit.tile_workers, explicit.automatic) == (
        37,
        3,
        False,
    )

    legacy = resolve_product_execution_policy(
        profile_id="safe-rich-v1", tile_size=None, tile_workers=1
    )
    assert (legacy.tile_size, legacy.tile_workers, legacy.automatic) == (
        None,
        1,
        False,
    )


def test_public_product_cli_uses_fixed_default_and_preserves_explicit_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    _source(source)

    observations = []
    for label, extra in (
        ("automatic", []),
        ("explicit", ["--tile-size", "17", "--tile-workers", "2"]),
    ):
        output = tmp_path / f"{label}.png"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/render_film.py"),
                str(source),
                "--style",
                "ektar_100",
                "--use-render-profile",
                "--render-profile",
                str(PROFILE),
                "--write-metrics",
                *extra,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        metrics = json.loads(output.with_suffix(".metrics.json").read_text())
        observations.append((output.read_bytes(), metrics["tile_size"]))

    assert observations[0][1] == 256
    assert observations[1][1] == 17
    assert observations[0][0] == observations[1][0]
