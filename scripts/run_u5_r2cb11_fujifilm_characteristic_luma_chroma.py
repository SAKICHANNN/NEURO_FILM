#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.eval.fujifilm_characteristic_luma_chroma import (
    evaluate,
    load_contract,
    write_report,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u5_r2cb11_fujifilm_characteristic_luma_chroma_v1.json",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    r = evaluate(load_contract(a.contract), ROOT, a.output_dir)
    d = write_report(r, a.report)
    print(f"automatic_pass={r['automatic_pass']}")
    print(f"failed_gates={','.join(k for k, v in r['checks'].items() if not v)}")
    print(f"stable_evidence_id={r['stable_evidence_id']}")
    print(f"report_sha256={d}")


if __name__ == "__main__":
    main()
