"""Run the separately frozen structural comparator, not a prior-run rescue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_ai_deep_photo_pilot import main


if __name__ == "__main__":
    main("configs/ai_structural_photo_pilot_v1.json", structural=True)
