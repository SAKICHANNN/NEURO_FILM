"""Run the frozen BW2.D0 HP5/Tri-X baseline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.bw_two_stock_proxy_baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
