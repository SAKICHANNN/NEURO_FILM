#!/usr/bin/env python3
"""P4FI 24MP windowed cloud runtime confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.benchmark_u6_p4fg_windowed_cloud_runtime_resources import (
    benchmark as base_benchmark,
)
from src.eval.native_msvc import sha256_file


def benchmark(contract_path:Path,output_dir:Path)->dict:
    contract=json.loads(contract_path.read_text());parent=ROOT/contract["parent"]["path"]
    if sha256_file(parent)!=contract["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=contract["parent"]["required_decision"]:raise RuntimeError("P4FI parent drift")
    scenario=contract["scenario"]
    derived={"schema":"neuro_film.u6_p4fg_windowed_cloud_runtime_resources_contract.v1","parent":contract["parent"],"scenario":scenario,
        "gates":{"minimum_median_peak_rss_reduction_bytes":0,"maximum_median_peak_rss_ratio":2,"maximum_median_worker_wall_ratio":10},"decision_if_pass":"base-pass","decision_if_fail":"base-fail","claim_ceiling":contract["claim_ceiling"]}
    path=output_dir/"derived-contract.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(derived))
    result=base_benchmark(path,output_dir/"base");runs=result["stable"]["runs"];g=contract["gates"]
    peaks=[r["peak_process_tree_rss_bytes"] for r in runs];increments=[max(0,r["peak_process_tree_rss_bytes"]-r["worker"]["render_entry_rss_bytes"]) for r in runs]
    walls=[r["worker"]["wall_seconds"] for r in runs];hashes={r["worker"]["output_sha256"] for r in runs}
    gates={"runs":len(runs)==2,"output":len(hashes)==1,"peak":max(peaks)<=g["maximum_peak_process_tree_rss_bytes"],"increment":max(increments)<=g["maximum_render_incremental_peak_bytes"],
        "wall":max(walls)<=g["maximum_worker_wall_seconds"],"repeat":max(peaks)/min(peaks)<=g["maximum_peak_repeat_ratio"],"windowed":all(not r["worker"]["full_forward_frame_retained"] for r in runs)}
    stable={"contract_sha256":sha256_file(contract_path),"runs":runs,"metrics":{"peak_process_tree_rss_bytes":peaks,"render_incremental_peak_bytes":increments,
        "worker_wall_seconds":walls,"peak_repeat_ratio":max(peaks)/min(peaks),"output_sha256":next(iter(hashes)) if len(hashes)==1 else None},"gates":gates,
        "decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4fi_windowed_cloud_runtime_24mp.v1","automatic_pass":all(gates.values()),"stable":stable,
        "stable_evidence_id":hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--contract",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);args=parser.parse_args();print(json.dumps(benchmark(args.contract,args.output_dir),sort_keys=True,separators=(",",":")))


if __name__=="__main__":main()
