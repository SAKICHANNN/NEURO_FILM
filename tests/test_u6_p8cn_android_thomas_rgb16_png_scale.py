from __future__ import annotations

import hashlib
from pathlib import Path

from src.eval.native_thomas_rgb16_png_android_scale import (
    inspect_png,
    parse_facts,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cn_android_thomas_rgb16_png_scale_v1.json"


def test_p8cn_contract_and_probe_are_frozen() -> None:
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == (
        "b8a780ba6c7e3efe14789ad00b22304fb0b9e5850bfe1d3eddd3d1aeaa55f212"
    )
    probe = ROOT / "native/film_physics/nf_thomas_rgb16_png_scale_probe_v1.c"
    assert probe.is_file()
    assert "3000" not in probe.read_text(encoding="utf-8")


def test_scale_probe_facts_are_strict() -> None:
    facts = parse_facts(
        "status=0 height=3000 width=4000 rows=128 bytes=72054675 "
        "calls=12012 workspace=29744013 means=0x1p-12,0x1p-12,0x1p-12"
    )
    assert facts == {
        "status": 0,
        "height": 3000,
        "width": 4000,
        "rows": 128,
        "bytes": 72054675,
        "calls": 12012,
        "workspace": 29744013,
    }


def test_development_png_is_exact_when_available() -> None:
    png = Path(r"D:\nf-019f4b76-android\p8cn-scale\host_3000x4000_run1.png")
    if not png.is_file():
        return
    facts = inspect_png(png)
    assert facts["shape_hwc"] == [3000, 4000, 3]
    assert facts["dtype"] == "uint16"
    assert facts["icc_exact"] is True
    assert facts["png_sha256"] == (
        "7ee2819fe7d6cfc8ef1c8caf5d63fa8eacfe639c7cb0654e13726d9dfeed8cb3"
    )
