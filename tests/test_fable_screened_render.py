import json
from pathlib import Path

import pytest

from src.preprocess import fable_screened_render as module


ROOT = Path(__file__).resolve().parents[1]


def test_bad_raw_digest_rejects_before_metadata_read(tmp_path, monkeypatch):
    seal = json.loads((ROOT / 'configs/fable_cdfe68_screen_lock_v1.json').read_text())
    raw = tmp_path / 'input.dng'
    raw.write_bytes(b'wrong input')
    def forbidden(*args):
        pytest.fail('metadata read before identity check')
    monkeypatch.setattr(module, 'inspect_numeric_metadata', forbidden)
    with pytest.raises(ValueError, match='RAW identity'):
        module.render_screened_raw(raw, root=ROOT, seal=seal, expected_raw_sha256='0' * 64)


def test_screen_drift_rejects_before_input_read():
    seal = json.loads((ROOT / 'configs/fable_cdfe68_screen_lock_v1.json').read_text())
    seal['project_files']['src/preprocess/fable_raw_eligibility.py'] = '0' * 64
    with pytest.raises(ValueError, match='drift'):
        module.render_screened_raw(Path('absent.dng'), root=ROOT, seal=seal, expected_raw_sha256='0' * 64)
