#!/usr/bin/env python
"""Fresh-process resource audit for the experimental typed scanner-glare chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p6zd_scanner_glare_streaming import (
    _atomic_json,
    _field,
    _hash_array,
    _ratio,
    _sha256,
)
from scripts.benchmark_u6_p6ze_scanner_glare_block_fft import _launch
from src.eval.scanner_glare_typed_chain import (
    apply_typed_scanner_glare_chain,
    load_contract,
    scanner_profile,
)

SCHEMA = "neuro_film.u6_p6zg_scanner_glare_typed_chain_contract.v1"


def worker(algorithm: str, shape: tuple[int, int, int], *, seed: int) -> dict[str, Any]:
    contract = load_contract(ROOT / "configs/u6_p6zg_scanner_glare_typed_chain_v1.json")
    profile = scanner_profile(ROOT, contract)
    source = _field(shape, seed)
    glare_algorithm = {
        "typed-chain-full-fft-glare": "full-fft",
        "typed-chain-channel-serial-glare": "channel-serial-block-fft",
    }.get(algorithm)
    if glare_algorithm is None:
        raise ValueError("unsupported benchmark algorithm")
    started = perf_counter()
    output = apply_typed_scanner_glare_chain(
        source,
        profile,
        pixel_pitch_um=float(contract["chain"]["pixel_pitch_um"]),
        glare_algorithm=glare_algorithm,
        glare_row_chunk=int(contract["chain"]["glare_row_chunk"]),
    )
    return {
        "algorithm": algorithm,
        "shape": list(shape),
        "source_sha256": _hash_array(source),
        "output_sha256": _hash_array(output),
        "apply_seconds": perf_counter() - started,
        "finite_bounded": bool(
            np.all(np.isfinite(output))
            and float(np.min(output)) >= 0.0
            and float(np.max(output)) <= 1.0
        ),
    }


def _probe(seed: int) -> dict[str, Any]:
    shape = (321, 513, 3)
    reference = worker("typed-chain-full-fft-glare", shape, seed=seed)
    candidate = worker("typed-chain-channel-serial-glare", shape, seed=seed)
    contract = load_contract(ROOT / "configs/u6_p6zg_scanner_glare_typed_chain_v1.json")
    profile = scanner_profile(ROOT, contract)
    source = _field(shape, seed)
    outputs = [
        apply_typed_scanner_glare_chain(
            source,
            profile,
            pixel_pitch_um=float(contract["chain"]["pixel_pitch_um"]),
            glare_algorithm=algorithm,
            glare_row_chunk=int(contract["chain"]["glare_row_chunk"]),
        )
        for algorithm in ("full-fft", "channel-serial-block-fft")
    ]
    difference = outputs[1] - outputs[0]
    return {
        "source_sha256": reference["source_sha256"],
        "reference_sha256": reference["output_sha256"],
        "candidate_sha256": candidate["output_sha256"],
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference), dtype=np.float64))),
    }


def parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = config_path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6ZG contract")
    hashes = {
        key.removesuffix("_path"): _sha256(ROOT / path)
        for key, path in config["parents"].items()
        if key.endswith("_path")
    }
    hashes_exact = all(
        hashes[key.removesuffix("_sha256")] == value
        for key, value in config["parents"].items()
        if key.endswith("_sha256")
    )
    benchmark = config["benchmark"]
    shape = tuple(int(value) for value in benchmark["shape"])
    counts = {name: 0 for name in set(benchmark["execution_order"])}
    runs = []
    for algorithm in benchmark["execution_order"]:
        counts[algorithm] += 1
        path = output_dir / f"{algorithm}_run{counts[algorithm]}.json"
        monitor = _launch(
            algorithm,
            shape,
            path,
            seed=int(benchmark["fixture_seed"]),
            row_chunk=int(config["chain"]["glare_row_chunk"]),
            interval=float(benchmark["rss_sample_interval_seconds"]),
            timeout=float(benchmark["worker_timeout_seconds"]),
            worker_script=Path(__file__).resolve(),
        )
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        runs.append({"monitor": monitor, "result": result})
        if monitor["exit_code"] != 0 or result is None:
            raise RuntimeError(
                f"typed-chain worker failed: {algorithm}: {monitor['stderr']}"
            )
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
    probe = _probe(int(benchmark["fixture_seed"]))
    reference = summaries["typed-chain-full-fft-glare"]
    candidate = summaries["typed-chain-channel-serial-glare"]
    gates = config["automatic_gates"]
    metrics = {
        "candidate_to_reference_median_peak_rss_ratio": candidate[
            "median_peak_process_tree_rss_bytes"
        ]
        / reference["median_peak_process_tree_rss_bytes"],
        "candidate_to_reference_median_apply_time_ratio": candidate[
            "median_apply_seconds"
        ]
        / reference["median_apply_seconds"],
    }
    decisions = {
        "parent_hashes": hashes_exact,
        "workers": all(
            row["monitor"]["exit_code"] == 0 and not row["monitor"]["timed_out"]
            for row in runs
        ),
        "finite_bounded": all(row["result"]["finite_bounded"] for row in runs),
        "source_identity": len({row["result"]["source_sha256"] for row in runs}) == 1,
        "repeat_output": all(row["repeat_output_exact"] for row in summaries.values()),
        "numeric_absolute": probe["maximum_absolute_error"]
        <= float(gates["maximum_cross_chain_absolute_error"]),
        "numeric_rmse": probe["rmse"] <= float(gates["maximum_cross_chain_rmse"]),
        "candidate_memory": candidate["maximum_peak_process_tree_rss_bytes"]
        <= int(gates["maximum_candidate_peak_process_tree_rss_bytes"]),
        "candidate_memory_ratio": metrics[
            "candidate_to_reference_median_peak_rss_ratio"
        ]
        <= float(gates["maximum_candidate_to_reference_median_peak_rss_ratio"]),
        "candidate_time": candidate["median_apply_seconds"]
        <= float(gates["maximum_candidate_median_apply_seconds"]),
        "candidate_time_ratio": metrics[
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
    stable = {
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_hashes": hashes,
        "probe": probe,
        "output_hashes": {
            name: next(
                row["result"]["output_sha256"]
                for row in runs
                if row["result"]["algorithm"] == name
            )
            for name in counts
        },
        "decisions": decisions,
        "automatic_pass": passed,
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    report = {
        "schema": "neuro_film.u6_p6zg_scanner_glare_typed_chain_report.v1",
        "node": config["node"],
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_hashes": hashes,
        "runs": runs,
        "summaries": summaries,
        "numerical_probe": probe,
        "measurements": metrics,
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
        default=Path("configs/u6_p6zg_scanner_glare_typed_chain_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-algorithm")
    parser.add_argument("--worker-shape", nargs=3, type=int)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--seed", type=int, default=6206106)
    parser.add_argument("--row-chunk", type=int, default=512)
    args = parser.parse_args()
    if args.worker_shape is not None:
        if args.worker_output is None or args.worker_algorithm is None:
            raise ValueError("worker algorithm/output are required")
        _atomic_json(
            args.worker_output,
            worker(args.worker_algorithm, tuple(args.worker_shape), seed=args.seed),
        )
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    report = parent(args.config, args.output_dir)
    print(f"automatic_pass={report['automatic_pass']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")


if __name__ == "__main__":
    main()
