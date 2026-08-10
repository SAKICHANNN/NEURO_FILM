#!/usr/bin/env python
"""Fresh-process U6.P6ZF channel-serial block-FFT glare benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p6zd_scanner_glare_streaming import (
    _atomic_json,
    _field,
    _hash_array,
    _profile,
    _ratio,
    _sha256,
)
from scripts.benchmark_u6_p6ze_scanner_glare_block_fft import _launch
from src.film_physics.scanner_glare import (
    apply_scanner_glare,
    compile_scanner_glare_kernel,
)
from src.film_physics.scanner_glare_channel_serial_fft import (
    apply_scanner_glare_channel_serial_block_fft,
)

SCHEMA = "neuro_film.u6_p6zf_scanner_glare_channel_serial_fft_contract.v1"


def worker(
    algorithm: str, shape: tuple[int, int, int], *, seed: int, row_chunk: int
) -> dict[str, Any]:
    profile, kernel_size = _profile(ROOT)
    kernel = compile_scanner_glare_kernel(profile, kernel_size=kernel_size)
    values = _field(shape, seed)
    from time import perf_counter

    started = perf_counter()
    if algorithm == "fft-2d-reference":
        output = apply_scanner_glare(
            values, kernel, flare_fraction=profile.flare_fraction
        )
    elif algorithm == "channel-serial-block-fft":
        output = apply_scanner_glare_channel_serial_block_fft(
            values,
            kernel,
            flare_fraction=profile.flare_fraction,
            row_chunk=row_chunk,
        )
    else:
        raise ValueError("unsupported benchmark algorithm")
    apply_seconds = perf_counter() - started
    return {
        "algorithm": algorithm,
        "shape": list(shape),
        "source_sha256": _hash_array(values),
        "output_sha256": _hash_array(output),
        "apply_seconds": apply_seconds,
        "finite_bounded": bool(
            np.all(np.isfinite(output))
            and float(np.min(output)) >= 0.0
            and float(np.max(output)) <= 1.0
        ),
        "logical_input_bytes": int(values.nbytes),
        "logical_output_bytes": int(output.nbytes),
    }


def numerical_probe(seed: int, partitions: list[int]) -> dict[str, Any]:
    profile, kernel_size = _profile(ROOT)
    kernel = compile_scanner_glare_kernel(profile, kernel_size=kernel_size)
    values = _field((321, 513, 3), seed)
    reference = apply_scanner_glare(
        values, kernel, flare_fraction=profile.flare_fraction
    )
    rows = []
    outputs = []
    for row_chunk in partitions:
        output = apply_scanner_glare_channel_serial_block_fft(
            values,
            kernel,
            flare_fraction=profile.flare_fraction,
            row_chunk=row_chunk,
        )
        difference = output - reference
        outputs.append(output)
        rows.append(
            {
                "row_chunk": row_chunk,
                "output_sha256": _hash_array(output),
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "rmse": float(
                    np.sqrt(np.mean(np.square(difference), dtype=np.float64))
                ),
            }
        )
    return {
        "source_sha256": _hash_array(values),
        "reference_sha256": _hash_array(reference),
        "rows": rows,
        "maximum_reference_absolute_error": max(
            row["maximum_absolute_error"] for row in rows
        ),
        "maximum_reference_rmse": max(row["rmse"] for row in rows),
        "maximum_partition_absolute_error": max(
            float(np.max(np.abs(output - outputs[0]))) for output in outputs
        ),
    }


def parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6ZF contract")
    parent_hashes = {
        key.removesuffix("_path"): _sha256(ROOT / path)
        for key, path in config["parents"].items()
        if key.endswith("_path")
    }
    parent_hashes_exact = all(
        parent_hashes[key.removesuffix("_sha256")] == expected
        for key, expected in config["parents"].items()
        if key.endswith("_sha256")
    )
    benchmark = config["benchmark"]
    shape = tuple(int(value) for value in benchmark["shape"])
    counts = {algorithm: 0 for algorithm in set(benchmark["execution_order"])}
    runs = []
    for algorithm in benchmark["execution_order"]:
        counts[algorithm] += 1
        result_path = output_dir / f"{algorithm}_run{counts[algorithm]}.json"
        monitor = _launch(
            algorithm,
            shape,
            result_path,
            seed=int(benchmark["fixture_seed"]),
            row_chunk=int(config["mechanism"]["row_chunk"]),
            interval=float(benchmark["rss_sample_interval_seconds"]),
            timeout=float(benchmark["worker_timeout_seconds"]),
            worker_script=Path(__file__).resolve(),
        )
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        )
        runs.append({"monitor": monitor, "result": result})
        if monitor["exit_code"] != 0 or result is None:
            raise RuntimeError(f"worker failed: {algorithm}: {monitor['stderr']}")

    summaries = {}
    for algorithm in counts:
        selected = [row for row in runs if row["result"]["algorithm"] == algorithm]
        times = [float(row["result"]["apply_seconds"]) for row in selected]
        peaks = [
            float(row["monitor"]["peak_process_tree_rss_bytes"]) for row in selected
        ]
        summaries[algorithm] = {
            "median_apply_seconds": float(np.median(times)),
            "median_peak_process_tree_rss_bytes": float(np.median(peaks)),
            "maximum_peak_process_tree_rss_bytes": int(max(peaks)),
            "repeat_apply_time_ratio": _ratio(times),
            "repeat_peak_rss_ratio": _ratio(peaks),
            "repeat_output_exact": len(
                {row["result"]["output_sha256"] for row in selected}
            )
            == 1,
        }
    probe = numerical_probe(
        int(benchmark["fixture_seed"]),
        [int(value) for value in config["mechanism"]["functional_row_partitions"]],
    )
    reference = summaries["fft-2d-reference"]
    candidate = summaries["channel-serial-block-fft"]
    gates = config["automatic_gates"]
    measurements = {
        "candidate_to_reference_median_peak_rss_ratio": (
            candidate["median_peak_process_tree_rss_bytes"]
            / reference["median_peak_process_tree_rss_bytes"]
        ),
        "candidate_to_reference_median_apply_time_ratio": (
            candidate["median_apply_seconds"] / reference["median_apply_seconds"]
        ),
    }
    decisions = {
        "parent_hashes": parent_hashes_exact,
        "workers": all(
            row["monitor"]["exit_code"] == 0
            and not row["monitor"]["timed_out"]
            and row["monitor"]["result_exists"]
            for row in runs
        ),
        "finite_bounded": all(row["result"]["finite_bounded"] for row in runs),
        "source_identity": len({row["result"]["source_sha256"] for row in runs}) == 1,
        "repeat_output": all(row["repeat_output_exact"] for row in summaries.values()),
        "reference_absolute": probe["maximum_reference_absolute_error"]
        <= float(gates["maximum_reference_absolute_error"]),
        "reference_rmse": probe["maximum_reference_rmse"]
        <= float(gates["maximum_reference_rmse"]),
        "partition": probe["maximum_partition_absolute_error"]
        <= float(gates["maximum_partition_absolute_error"]),
        "candidate_memory": candidate["maximum_peak_process_tree_rss_bytes"]
        <= int(gates["maximum_candidate_peak_process_tree_rss_bytes"]),
        "candidate_memory_ratio": measurements[
            "candidate_to_reference_median_peak_rss_ratio"
        ]
        <= float(gates["maximum_candidate_to_reference_median_peak_rss_ratio"]),
        "candidate_time": candidate["median_apply_seconds"]
        <= float(gates["maximum_candidate_median_apply_seconds"]),
        "candidate_time_ratio": measurements[
            "candidate_to_reference_median_apply_time_ratio"
        ]
        <= float(gates["maximum_candidate_to_reference_median_apply_time_ratio"]),
        "repeat_rss": all(
            row["repeat_peak_rss_ratio"]
            <= float(gates["maximum_repeat_peak_rss_ratio"])
            for row in summaries.values()
        ),
        "repeat_time": all(
            row["repeat_apply_time_ratio"]
            <= float(gates["maximum_repeat_apply_time_ratio"])
            for row in summaries.values()
        ),
    }
    passed = all(decisions.values())
    stable_core = {
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_hashes": parent_hashes,
        "probe": probe,
        "output_hashes": {
            algorithm: next(
                row["result"]["output_sha256"]
                for row in runs
                if row["result"]["algorithm"] == algorithm
            )
            for algorithm in counts
        },
        "decisions": decisions,
        "automatic_pass": passed,
    }
    stable_id = hashlib.sha256(
        json.dumps(stable_core, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    report = {
        "schema": "neuro_film.u6_p6zf_scanner_glare_channel_serial_fft_report.v1",
        "node": config["node"],
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_hashes": parent_hashes,
        "runs": runs,
        "summaries": summaries,
        "numerical_probe": probe,
        "measurements": measurements,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": config["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
        "stable_evidence_id": stable_id,
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u6_p6zf_scanner_glare_channel_serial_fft_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-algorithm")
    parser.add_argument("--worker-shape", nargs=3, type=int)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--seed", type=int, default=6206105)
    parser.add_argument("--row-chunk", type=int, default=512)
    args = parser.parse_args()
    if args.worker_shape is not None:
        if args.worker_output is None or args.worker_algorithm is None:
            raise ValueError("worker algorithm/output are required")
        _atomic_json(
            args.worker_output,
            worker(
                args.worker_algorithm,
                tuple(args.worker_shape),
                seed=args.seed,
                row_chunk=args.row_chunk,
            ),
        )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required in parent mode")
    report = parent(args.config, args.output_dir)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    for key, summary in report["summaries"].items():
        print(
            f"{key} apply={summary['median_apply_seconds']:.3f}s "
            f"peak={summary['median_peak_process_tree_rss_bytes'] / 2**20:.1f}MiB"
        )


if __name__ == "__main__":
    main()
