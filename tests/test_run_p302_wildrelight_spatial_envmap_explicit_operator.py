from __future__ import annotations

import numpy as np

from scripts.run_p302_wildrelight_spatial_envmap_explicit_operator import (
    _metric,
    _paths,
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
