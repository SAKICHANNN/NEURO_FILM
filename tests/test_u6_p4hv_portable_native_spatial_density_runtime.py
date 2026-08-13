from __future__ import annotations

import json
from pathlib import Path

from src.eval.portable_native_spatial_density_runtime import (
    SCHEMA,
    SOURCES,
    _resolve_output_dir,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p4hv_contract_and_probe_are_fixed() -> None:
    contract = json.loads(
        (
            ROOT / "configs/u6_p4hv_portable_native_spatial_density_runtime_v1.json"
        ).read_text("utf-8")
    )
    assert contract["schema"] == SCHEMA
    assert contract["runtime"]["wipe_data_cold_boots"] == 2
    assert contract["runtime"]["fresh_processes_per_boot"] == 2
    assert (
        contract["fixture"]["profile_bundle_sha256"]
        == "8d20ce2874036befe78657b059ffafeb24e43d2e5bc1000512092a753e1e75df"
    )
    assert len(SOURCES) == 4
    assert all((ROOT / source).is_file() for source in SOURCES)


def test_p4hv_relative_output_is_anchored_to_repository(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _resolve_output_dir(ROOT, Path("outputs/p4hv")) == (
        ROOT / "outputs/p4hv"
    ).resolve()
