from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import run_p287_wildrelight_envmap_conditioned_hdr_d0 as p287


def test_spherical_rgb_mean_preserves_constant_radiance() -> None:
    values = np.empty((7, 11, 3), dtype=np.float32)
    values[...] = np.asarray([0.25, 1.5, 4.0], dtype=np.float32)

    observed = p287._spherical_rgb_mean(values)

    np.testing.assert_allclose(observed, [0.25, 1.5, 4.0], rtol=0.0, atol=4e-15)


def test_spherical_rgb_mean_rejects_negative_support() -> None:
    values = np.ones((3, 5, 3), dtype=np.float32)
    values[0, 0, 1] = -np.float32(1e-6)

    with pytest.raises(p287.P287Error, match="negative radiance"):
        p287._spherical_rgb_mean(values)


def test_log_metric_uses_central_crop_and_strict_support() -> None:
    source = np.ones((20, 20, 3), dtype=np.float32)
    target = source * np.float32(2.0)
    candidate = target.copy()
    candidate[0, :, :] = 1000.0
    candidate[:, 0, :] = 1000.0

    observed = p287._metric(candidate, target, source)

    assert observed["rmse"] == 0.0
    assert observed["valid_fraction"] == 1.0


def test_capture_separations_bind_matching_time_roles() -> None:
    metadata = {
        "photos": [
            {"time": "time0", "shooting_time": "2025:08:16 20:03:28"},
            {"time": "time1", "shooting_time": "2025:08:16 20:17:05"},
        ],
        "envmaps": [
            {"time": "time0", "shooting_time": "2025:08:16 20:03:04"},
            {"time": "time1", "shooting_time": "2025:08:16 20:16:48"},
        ],
    }

    assert p287._capture_separations(metadata) == [24.0, 17.0]


def test_existing_source_members_are_verified_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"p287-exact-member"
    source = tmp_path / "data" / "member.bin"
    source.parent.mkdir()
    source.write_bytes(body)
    config = {
        "source": {
            "local_root": "data",
            "members": [
                {
                    "relative_path": "member.bin",
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            ],
        }
    }
    monkeypatch.setattr(p287, "ROOT", tmp_path)

    observed = p287._exact_source_files(config)

    assert observed == [
        {
            "bytes": len(body),
            "exists": True,
            "relative_path": "member.bin",
            "sha256": hashlib.sha256(body).hexdigest(),
            "valid": True,
        }
    ]


def test_contract_remains_frozen_before_selected_exr_request() -> None:
    path = Path("docs/planning/P287_WILDRELIGHT_ENVMAP_CONDITIONED_HDR_D0_CONTRACT.md")
    text = path.read_text(encoding="utf-8")

    assert "before any selected EXR request or pixel decode" in text
    assert "Candidate 3 remains unconsumed" in text
    assert "lake" in text
    assert "108,447,512 bytes" in text
