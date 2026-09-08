import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tst_oracle", ROOT / "scripts/run_tst100k_global_oracle.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.CONFIG = ROOT / "configs/tst100k_global_oracle_recovery_v1.json"
module.__file__ = __file__

if __name__ == "__main__":
    raise SystemExit(module.main())
