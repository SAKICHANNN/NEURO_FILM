import copy
import json
from pathlib import Path

import pytest

from src.preprocess.fable_render_lock import verify_render_lock, render_locked_canonical_raw


ROOT = Path(__file__).resolve().parents[1]


def lock():
    return json.loads((ROOT / 'configs/fable_cdfe68_render_lock_v1.json').read_text())


def test_live_render_lock():
    assert verify_render_lock(ROOT, lock())['params']['no_auto_bright'] is True


@pytest.mark.parametrize('kind', ['version', 'binary', 'config'])
def test_drift_rejected_before_raw_open(kind, monkeypatch):
    value = copy.deepcopy(lock())
    if kind == 'version':
        value['runtime']['rawpy'] = 'different'
    elif kind == 'binary':
        value['runtime_files'][0]['sha256'] = '0' * 64
    else:
        value['project_files'][value['config']] = '0' * 64
    def forbidden(*args, **kwargs):
        pytest.fail('RAW opened despite failed lock')
    monkeypatch.setattr('src.preprocess.fable_render_lock.render_canonical_raw', forbidden)
    with pytest.raises(ValueError, match='drift'):
        render_locked_canonical_raw(Path('not-opened.dng'), root=ROOT, lock=value)
