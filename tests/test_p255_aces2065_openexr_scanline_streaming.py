from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/eval/p255_aces2065_openexr_scanline_writer.cpp"


def test_native_source_binds_ap0_transform_and_identity() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for token in (
        "0.6954522414",
        "0.1406786965",
        "0.1638690622",
        "0.7347F",
        "-0.077F",
        '"acesImageContainerFlag"',
        '"colorInteropID"',
        '"lin_ap0_scene"',
        "ZIP_COMPRESSION",
        "P255 row block must equal 16",
        "MOVEFILE_WRITE_THROUGH",
    ):
        assert token in text


def test_p248_parent_source_remains_separate() -> None:
    parent = ROOT / "src/eval/p248_acescg_openexr_scanline_writer.cpp"
    text = parent.read_text(encoding="utf-8")
    assert "acesImageContainerFlag" not in text
    assert "colorInteropID" not in text
    assert "P248 row block must equal 16" in text
