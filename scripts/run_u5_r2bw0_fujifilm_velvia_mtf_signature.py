"""Repository entry point for U5.R2BW0."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.fujifilm_velvia_mtf_signature_run import main

if __name__ == "__main__":
    raise SystemExit(main())
