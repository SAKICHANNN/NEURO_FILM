#!/usr/bin/env python3
"""Fresh-process P4FD versus P4FF cloud runtime resource benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure,
    render_physical_partition,
)
from src.eval.native_msvc import sha256_file
from src.film_physics.native_cloud_scan_runtime import (
    NativeCloudScanRuntime,
)
from src.film_physics.native_cloud_scan_runtime_v2 import (
    WindowedNativeCloudScanRuntime,
)
from tests.test_u6_p4fc_opt_in_cloud_scan_runtime_v1 import _profile


def source_rows(height:int,width:int,block_rows:int|None=None)->np.ndarray:
    if block_rows is None:
        y=np.arange(height,dtype=np.uint64)[:,None,None];x=np.arange(width,dtype=np.uint64)[None,:,None];c=np.arange(3,dtype=np.uint64)[None,None,:]
        return np.ascontiguousarray(((y*17+x*31+c*101+3)%997)/996.,np.float32)
    result=np.empty((height,width,3),np.float32);x=np.arange(width,dtype=np.uint64)[None,:,None];c=np.arange(3,dtype=np.uint64)[None,None,:]
    for y0 in range(0,height,block_rows):
        y1=min(height,y0+block_rows);y=np.arange(y0,y1,dtype=np.uint64)[:,None,None]
        result[y0:y1]=((y*17+x*31+c*101+3)%997)/996.
    return result


def worker(*,variant:str,dll:Path,height:int,width:int,tile_rows:int,output:Path,source_block_rows:int|None=None)->None:
    lib=_configure(dll);p4fb=json.loads((ROOT/"configs/u6_p4fb_native_cloud_spatial_partition_v1.json").read_text())
    p4fb["fixture"]["full_height"]=height;p4fb["fixture"]["width"]=width
    source=source_rows(height,width,source_block_rows);profile=_profile();component=sha256_file(ROOT/"native/film_physics/nf_cloud_post_spatial_f32_v1.c")
    render_entry_rss_bytes=psutil.Process().memory_info().rss
    started=time.perf_counter()
    if variant=="full":
        def provider(values:np.ndarray,y0:int,count:int)->np.ndarray:return render_physical_partition(lib,p4fb,values,y0,count)
        runtime=NativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,physical_rows=provider,physical_component_sha256=component,tile_rows=tile_rows)
    elif variant=="windowed":
        def provider(get_rows,y0:int,count:int)->np.ndarray:return render_physical_partition(lib,p4fb,get_rows,y0,count)
        runtime=WindowedNativeCloudScanRuntime(gaussian_library=dll,forward_scatter_profile=profile,physical_rows=provider,physical_component_sha256=component,tile_rows=tile_rows)
    else:raise ValueError("invalid P4FG variant")
    receipt=runtime.render_to_sink(source,output_sink=lambda y0,y1,rows:None)
    result={"variant":variant,"height":height,"width":width,"tile_rows":tile_rows,"wall_seconds":time.perf_counter()-started,
        "output_sha256":receipt["output_sha256"],"input_sha256":receipt["input_sha256"],"full_forward_frame_retained":bool(receipt.get("full_forward_frame_retained",True)),
        "maximum_forward_workspace_values":int(receipt.get("maximum_forward_workspace_values",height*width*3)),"render_entry_rss_bytes":render_entry_rss_bytes}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))


def monitor(command:list[str],output:Path,timeout:float,interval:float)->dict:
    process=subprocess.Popen(command,cwd=ROOT);root=psutil.Process(process.pid);peak=0;started=time.perf_counter()
    try:
        while process.poll() is None:
            if time.perf_counter()-started>timeout:raise TimeoutError("P4FG worker timeout")
            total=0
            try:
                observed=[root,*root.children(recursive=True)]
            except psutil.NoSuchProcess:
                observed=[]
            for item in observed:
                try:total+=item.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            peak=max(peak,total);time.sleep(interval)
        if process.returncode!=0:raise RuntimeError(f"P4FG worker exit {process.returncode}")
        return {"peak_process_tree_rss_bytes":peak,"worker":json.loads(output.read_text())}
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=10)


def benchmark(contract_path:Path,output_dir:Path)->dict:
    contract=json.loads(contract_path.read_text());parent=ROOT/contract["parent"]["path"]
    if sha256_file(parent)!=contract["parent"]["sha256"] or json.loads(parent.read_text())["decision"]!=contract["parent"]["required_decision"]:raise RuntimeError("P4FG parent drift")
    dll=_build(ROOT,output_dir/"build",None);scenario=contract["scenario"];runs=[]
    for index,variant in enumerate(scenario["run_order"]):
        worker_output=output_dir/f"worker-{index+1}-{variant}.json"
        command=[sys.executable,str(Path(__file__).resolve()),"--worker","--variant",variant,"--dll",str(dll),"--height",str(scenario["height"]),"--width",str(scenario["width"]),"--tile-rows",str(scenario["tile_rows"]),"--output",str(worker_output)]
        if "source_construction_rows" in scenario:command.extend(["--source-block-rows",str(scenario["source_construction_rows"])])
        runs.append(monitor(command,worker_output,scenario["timeout_seconds"],scenario["sample_interval_seconds"]))
    by_variant={name:[r for r in runs if r["worker"]["variant"]==name] for name in ("full","windowed")}
    median=lambda values:float(np.median(np.asarray(values,dtype=np.float64)))
    full_rss=median([r["peak_process_tree_rss_bytes"] for r in by_variant["full"]]);window_rss=median([r["peak_process_tree_rss_bytes"] for r in by_variant["windowed"]])
    full_wall=median([r["worker"]["wall_seconds"] for r in by_variant["full"]]);window_wall=median([r["worker"]["wall_seconds"] for r in by_variant["windowed"]])
    hashes={r["worker"]["output_sha256"] for r in runs};g=contract["gates"]
    gates={"output":len(hashes)==1,"runs":all(len(by_variant[v])==2 for v in by_variant),"rss_bytes":full_rss-window_rss>=g["minimum_median_peak_rss_reduction_bytes"],
        "rss_ratio":window_rss/full_rss<=g["maximum_median_peak_rss_ratio"],"wall":window_wall/full_wall<=g["maximum_median_worker_wall_ratio"],
        "cleanup":not any(r.is_running() for r in []),"windowed_no_full":all(not r["worker"]["full_forward_frame_retained"] for r in by_variant["windowed"])}
    stable={"contract_sha256":sha256_file(contract_path),"runs":runs,"metrics":{"full_median_peak_rss_bytes":full_rss,"windowed_median_peak_rss_bytes":window_rss,
        "rss_reduction_bytes":full_rss-window_rss,"rss_ratio":window_rss/full_rss,"full_median_wall_seconds":full_wall,"windowed_median_wall_seconds":window_wall,"wall_ratio":window_wall/full_wall},
        "gates":gates,"decision":contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"],"claim_ceiling":contract["claim_ceiling"]}
    return {"schema":"neuro_film.u6_p4fg_windowed_cloud_runtime_resources.v1","automatic_pass":all(gates.values()),"stable":stable}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--worker",action="store_true");parser.add_argument("--variant");parser.add_argument("--dll",type=Path);parser.add_argument("--height",type=int);parser.add_argument("--width",type=int);parser.add_argument("--tile-rows",type=int);parser.add_argument("--output",type=Path);parser.add_argument("--source-block-rows",type=int);parser.add_argument("--contract",type=Path);parser.add_argument("--output-dir",type=Path);args=parser.parse_args()
    if args.worker:worker(variant=args.variant,dll=args.dll,height=args.height,width=args.width,tile_rows=args.tile_rows,output=args.output,source_block_rows=args.source_block_rows)
    else:print(json.dumps(benchmark(args.contract,args.output_dir),sort_keys=True,separators=(",",":")))


if __name__=="__main__":main()
