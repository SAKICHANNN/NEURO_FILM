from __future__ import annotations

from pathlib import Path

from src.inference.romm_rec2020_velvia_staged import (
    render_supported_prophoto_velvia_rec2020_staged,
)
from src.inference.romm_rec2020_velvia_staged_native_v2 import (
    render_supported_prophoto_velvia_rec2020_staged_native_v2,
)
from tests.test_u1_4c19_staged_prophoto_render import _write_source

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/prophoto_rec2020_velvia_v1.json"


def test_native_v2_staged_renderer_preserves_exact_product_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tif"
    expected = tmp_path / "expected.png"
    actual = tmp_path / "actual.png"
    _write_source(source)
    expected_receipt = render_supported_prophoto_velvia_rec2020_staged(
        source,
        expected,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "expected_scratch",
        row_chunk=13,
    )
    actual_receipt = render_supported_prophoto_velvia_rec2020_staged_native_v2(
        source,
        actual,
        profile_path=PROFILE,
        root=ROOT,
        scratch_dir=tmp_path / "actual_scratch",
        build_dir=tmp_path / "native_build",
        row_chunk=13,
        thread_count=4,
    )
    assert actual.read_bytes() == expected.read_bytes()
    assert actual_receipt == expected_receipt
