from __future__ import annotations

import json
from pathlib import Path

from scripts.acquire_p302_wildrelight_spatial_envmap_source import (
    _confirmation_allowed,
    _local_path,
)


def test_local_path_stays_below_small_aligned_root(tmp_path: Path) -> None:
    observed = _local_path(tmp_path, "small-aligned/scene/photo/time0_hdr.exr")
    assert observed == tmp_path / "scene/photo/time0_hdr.exr"


def test_confirmation_requires_exact_development_pass(tmp_path: Path) -> None:
    report = tmp_path / "development.json"
    report.write_text(json.dumps({"decision": "FAIL_CLOSED_P302_DEVELOPMENT"}))
    assert not _confirmation_allowed(report)
    report.write_text(json.dumps({"decision": "PASS_PRIVATE_P302_DEVELOPMENT"}))
    assert _confirmation_allowed(report)
