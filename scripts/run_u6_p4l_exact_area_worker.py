#!/usr/bin/env python
"""One bounded U6.P4L exact-area stream worker."""

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
from src.eval.physical_exact_area_live_performance import (  # noqa: E402
    load_contract,
)
from src.eval.physical_exact_area_streaming import (  # noqa: E402
    _load_exact,
    _profiles,
    _stream_hash,
    _target,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract, _ = load_contract(
        ROOT, args.config, args.expected_config_sha256
    )
    p4d = _load_exact(
        ROOT,
        "configs/u6_p4d_density_conditioned_structure_v1.json",
        "8c0af4e3e485a1b790294d868fa729f8aeee8bfd206b3488b5f862f43a12ddab",
    )
    worker = contract["worker"]
    target = _target(
        tuple(int(value) for value in worker["output_shape"]),
        maximum_density=2.0,
        channel_scales=[1.0, 0.82, 0.64],
    )
    profiles = _profiles(
        p4d, seed_offset=int(worker["seed_offset"])
    )
    result = _stream_hash(
        target,
        profiles,
        factor=int(worker["pixel_size_factor"]),
        row_tile_height=int(worker["row_tile_height"]),
    )
    payload = {
        "schema": "neuro_film.u6_p4l_exact_area_worker_result.v1",
        "stream_sha256": result[0],
        "output_rows": result[1],
        "maximum_virtual_output_rows": result[2],
        "minimum_density": result[3],
        "maximum_transmittance": result[4],
    }
    _atomic_write(args.output, _canonical_json(payload))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
