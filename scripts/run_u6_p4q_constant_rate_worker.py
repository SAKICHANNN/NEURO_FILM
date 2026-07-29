#!/usr/bin/env python
"""One bounded U6.P4Q scalar-CDF stationary worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from src.eval.physical_constant_rate_integration_live import (  # noqa: E402
    load_contract,
)
from src.eval.physical_stationary_global_live_performance import (  # noqa: E402
    compiled_profiles,
    stationary_target,
    stream_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract, parent = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    executor = contract["executor"]
    target = stationary_target(
        tuple(int(value) for value in executor["output_shape"]),
        executor["target_density"],
    )
    result = stream_hash(
        target,
        compiled_profiles(contract, parent),
        row_tile_height=int(executor["row_tile_height"]),
        executor_mode="scalar-cdf-v1",
    )
    result.update(
        {
            "schema": "neuro_film.u6_p4q_constant_rate_worker_result.v1",
            "executor_mode": "scalar-cdf-v1",
            "target_base_bytes": 24,
            "target_owns_data": bool(target.flags.owndata),
            "full_output_assembly_count": 0,
        }
    )
    _atomic_write(args.output, _canonical_json(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
