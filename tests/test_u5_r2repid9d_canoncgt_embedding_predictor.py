from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_contract_binds_all_implementation_files() -> None:
    config = json.loads(
        (ROOT / "configs/u5_r2repid9d_canoncgt_embedding_predictor_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for binding in config["implementation"].values():
        assert (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            == binding["sha256"]
        )
    assert (
        hashlib.sha256(
            (ROOT / config["canoncgt"]["contract_path"]).read_bytes()
        ).hexdigest()
        == config["canoncgt"]["contract_sha256"]
    )


def test_runner_requires_external_roots_and_freezes_before_original_read() -> None:
    source = (
        ROOT / "scripts/run_u5_r2repid9d_canoncgt_embedding_predictor.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--original-root", type=Path, required=True)' in source
    assert (
        'parser.add_argument("--closed-raw-stat-report", type=Path, required=True)'
        in source
    )
    assert "D:\\" not in source
    assert "P:\\" not in source
    freeze = source.index("freeze_sha =")
    assert freeze < source.index("original = load_original_srgb(", freeze)
