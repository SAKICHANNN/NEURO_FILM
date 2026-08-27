from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.run_p302_wildrelight_spatial_envmap_explicit_operator import (
    P302Error,
    _metric,
    _paths,
    _verify_bindings,
)


def _config() -> dict[str, object]:
    return {"operator": {"crop_fraction": 0.9, "epsilon": 2.0**-16}}


def test_paths_use_disjoint_one_way_roles_and_frozen_cycle() -> None:
    observed = _paths("scene", 0, 1)
    assert observed["source_photo"].endswith("photo/time0_hdr.exr")
    assert observed["target_photo"].endswith("photo/time1_hdr.exr")
    assert observed["target_env"].endswith("envmap/time1_envmap.exr")
    assert observed["cyclic_env"].endswith("envmap/time3_envmap.exr")


def test_log_metric_is_zero_for_exact_candidate() -> None:
    source = np.full((10, 12, 3), 0.25, dtype=np.float32)
    error, valid = _metric(source, source.copy(), source, _config())
    assert error == 0.0
    assert valid == 1.0


def test_execution_binding_is_exact_and_tamper_rejects(tmp_path, monkeypatch) -> None:
    body = b"frozen-p302-binding\n"
    bound = tmp_path / "bound.txt"
    bound.write_bytes(body)
    monkeypatch.setattr(
        "scripts.run_p302_wildrelight_spatial_envmap_explicit_operator.ROOT",
        tmp_path,
    )
    config = {
        "bindings": [
            {
                "bytes": len(body),
                "path": "bound.txt",
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ]
    }
    assert _verify_bindings(config)[0]["sha256"] == hashlib.sha256(body).hexdigest()
    bound.write_bytes(body + b"tamper")
    with pytest.raises(P302Error, match="binding differs"):
        _verify_bindings(config)
