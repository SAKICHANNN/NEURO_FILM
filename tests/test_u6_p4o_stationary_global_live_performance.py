from __future__ import annotations

import numpy as np

from scripts.run_u6_p4o_stationary_global_live_performance import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_stationary_global_live_performance import (
    compiled_profiles,
    load_contract,
    small_partition_parity,
    stationary_target,
    stream_hash,
)
from src.film_physics.density_conditioned_structure import (
    iter_density_conditioned_structure_rows,
)


CONFIG = ROOT / "configs/u6_p4o_stationary_global_live_performance_v1.json"


def test_contract_is_stationary_synthetic_and_local_only() -> None:
    contract, _ = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["photograph_access_allowed"] is False
    assert contract["executor"]["full_output_assembly_allowed"] is False
    assert contract["executor"]["full_target_allocation_allowed"] is False


def test_small_partitions_are_byte_exact() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    report = small_partition_parity(contract, parent)
    assert all(
        row["byte_exact"] for row in report["partitions"].values()
    )


def test_broadcast_target_stream_is_repeat_exact() -> None:
    contract, parent = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    target = stationary_target((23, 29), [0.8, 0.656, 0.512])
    assert target.flags.owndata is False
    profiles = compiled_profiles(contract, parent)
    first = stream_hash(target, profiles, row_tile_height=7)
    second = stream_hash(target, profiles, row_tile_height=11)
    assert first["output_rows"] == 23
    assert second["output_rows"] == 23
    arrays = list(
        iter_density_conditioned_structure_rows(
            target, profiles, row_tile_height=7
        )
    )
    density = np.concatenate(
        [row.density for _, row in arrays], axis=0
    )
    assert density.shape == (23, 29, 3)
