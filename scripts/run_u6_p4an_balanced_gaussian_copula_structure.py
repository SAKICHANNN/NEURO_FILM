#!/usr/bin/env python
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.eval.physical_balanced_gaussian_copula_structure import evaluate_structure,load_contract  # noqa:E402
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=ROOT/"configs/u6_p4an_balanced_gaussian_copula_structure_v1.json");p.add_argument("--report",type=Path,required=True);a=p.parse_args();r=evaluate_structure(load_contract(a.config),ROOT);a.report.parent.mkdir(parents=True,exist_ok=True);t=a.report.with_name(a.report.name+".tmp");t.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8",newline="\n");
 with t.open("r+b") as f:os.fsync(f.fileno())
 os.replace(t,a.report);print(json.dumps({"passed":r["passed"],"stable_evidence_id":r["stable_evidence_id"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
