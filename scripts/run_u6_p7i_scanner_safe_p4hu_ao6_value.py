from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_u6_p7h_p4hu_ao6_value as runner
from src.eval.scanner_safe_p4hu_ao6_value import evaluate, load_contract

runner.evaluate = evaluate
runner.load_contract = load_contract
runner.WORKER_REPORT_SCHEMA = "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-worker-report.v1"
runner.FINAL_REPORT_SCHEMA = "neuro-film.u6-p7i-scanner-safe-p4hu-ao6-value-run-report.v1"


if __name__ == "__main__":
    raise SystemExit(runner.main())
