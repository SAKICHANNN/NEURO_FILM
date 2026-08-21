from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_p88_ultrahdr_pinned_decoder_consumption_v1 import (
    build_decoder_command,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p88_ultrahdr_pinned_decoder_consumption_v1.json"


def test_p88_contract_binds_exact_fixture_bytes_and_command() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = ROOT / "tests/fixtures/u1_5c_libultrahdr"
    assert [row["name"] for row in config["fixtures"]] == [
        "apple_gainmap_new.jpg",
        "apple_gainmap_old.jpg",
    ]
    for row in config["fixtures"]:
        assert (
            hashlib.sha256((root / row["name"]).read_bytes()).hexdigest()
            == row["sha256"]
        )
        assert (row["width"], row["height"]) == (384, 512)
    assert build_decoder_command(Path("decoder.exe"), "input.jpg", "output.raw") == [
        "decoder.exe",
        "-m",
        "1",
        "-j",
        "input.jpg",
        "-o",
        "0",
        "-O",
        "4",
        "-z",
        "output.raw",
    ]
