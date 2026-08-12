from pathlib import Path

from src.eval.native_msvc import sha256_file
from src.eval.physical_native_f32_conformance import _build_component

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/film_physics/nf_gaussian_rgb_f32_v1.c"
HEADER = ROOT / "native/film_physics/nf_gaussian_rgb_f32_v1.h"


def test_frozen_gaussian_component_identity_is_unchanged(tmp_path: Path) -> None:
    assert sha256_file(SOURCE) == (
        "dcb26b664e1fd8cbd514246846070c257b5250994f996d0c4b8e4f30040116c9"
    )
    assert sha256_file(HEADER) == (
        "1f75a36fbbcc6944ad2dbe4daea634a8d3d4b2e1392d74ee40c2345b99a6f710"
    )
    built = _build_component(
        root=ROOT,
        output_dir=tmp_path,
        component={
            "source": "native/film_physics/nf_gaussian_rgb_f32_v1.c",
            "header": "native/film_physics/nf_gaussian_rgb_f32_v1.h",
        },
        basename="nf_gaussian_rgb_f32_v1",
    )
    assert built["dll_sha256"] == (
        "37686905def49425235a2d1625fb7728448263013cd376173171dec2e5c9693f"
    )
