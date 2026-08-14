"""Run the frozen U6.P7I scanner-safe P4HU value experiment."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["NEURO_FILM_P7_VALUE_EVALUATOR"] = (
    "src.eval.scanner_safe_p4hu_ao6_value"
)

main = import_module("scripts.run_u6_p7h_p4hu_ao6_value").main


if __name__ == "__main__":
    raise SystemExit(main())
