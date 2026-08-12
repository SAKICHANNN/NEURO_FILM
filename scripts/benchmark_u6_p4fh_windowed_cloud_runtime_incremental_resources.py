#!/usr/bin/env python3
"""P4FH incremental-RSS wrapper around the exact P4FG worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.benchmark_u6_p4fg_windowed_cloud_runtime_resources import (
    benchmark as base_benchmark,
)
from src.eval.native_msvc import sha256_file


def benchmark(contract_path:Path,output_dir:Path)->dict:
    contract=json.loads(contract_path.read_text());parent=ROOT/contract["parent"]["path"]
    if sha256_file(parent)!=contract["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=contract["parent"]["required_decision"]:raise RuntimeError("P4FH parent drift")
    base={"schema":"neuro_film.u6_p4fg_windowed_cloud_runtime_resources_contract.v1","parent":contract["parent"],"scenario":contract["scenario"],"gates":{"minimum_median_peak_rss_reduction_bytes":0,"maximum_median_peak_rss_ratio":2,"maximum_median_worker_wall_ratio":contract["gates"]["maximum_median_worker_wall_ratio"]},
        "decision_if_pass":"base-pass","decision_if_fail":"base-fail","claim_ceiling":contract["claim_ceiling"]}
    base_path=output_dir/"derived-base-contract.json";base_path.parent.mkdir(parents=True,exist_ok=True);base_path.write_text(json.dumps(base))
    # The base harness checks its own P4FG parent decision; P4FH already checked the P4FG evidence.
    original=json.loads(parent.read_text());base["parent"]["required_decision"]=original["decision"];base_path.write_text(json.dumps(base))
    result=base_benchmark(base_path,output_dir/"base")
    runs=result["stable"]["runs"];by={name:[r for r in runs if r["worker"]["variant"]==name] for name in ("full","windowed")}
    median=lambda v:float(np.median(np.asarray(v,np.float64)))
    increments={name:[max(0,r["peak_process_tree_rss_bytes"]-r["worker"]["render_entry_rss_bytes"]) for r in rows] for name,rows in by.items()}
    full=median(increments["full"]);window=median(increments["windowed"]);g=contract["gates"]
    source_hashes={r["worker"]["input_sha256"] for r in runs};output_hashes={r["worker"]["output_sha256"] for r in runs}
    gates={"source":source_hashes=={"d2647ee01ef91402a181d6e183af2b326bd4cade340131878f1e49464b9a0b46"},"output":output_hashes=={"a3eb79bfe0aeada767721ab321df8aed68c97649154f1278b4cdf95e98943ee4"},
        "rss_bytes":full-window>=g["minimum_median_render_incremental_peak_reduction_bytes"],"rss_ratio":window/full<=g["maximum_median_render_incremental_peak_ratio"],
        "wall":result["stable"]["metrics"]["wall_ratio"]<=g["maximum_median_worker_wall_ratio"],"runs":all(len(v)==2 for v in by.values())}
    stable={"contract_sha256":sha256_file(contract_path),"runs":runs,"incremental_peak_bytes":increments,
        "metrics":{"full_median_render_incremental_peak_bytes":full,"windowed_median_render_incremental_peak_bytes":window,"reduction_bytes":full-window,"ratio":window/full,
            "wall_ratio":result["stable"]["metrics"]["wall_ratio"]},"gates":gates,"decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    stable_id=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"schema":"neuro_film.u6_p4fh_windowed_cloud_runtime_incremental_resources.v1","automatic_pass":all(gates.values()),"stable":stable,"stable_evidence_id":stable_id}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);args=parser.parse_args();print(json.dumps(benchmark(args.contract,args.output_dir),sort_keys=True,separators=(",",":")))


if __name__=="__main__":main()
