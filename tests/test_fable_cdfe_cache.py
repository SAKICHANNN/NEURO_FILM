import numpy as np
import pytest

from src.data.fable_cdfe_cache import load_completed_donor, save_completed_donor


def test_cache_receipt_reuse_and_corruption_detection(tmp_path):
    example = {'treatment_id': 't0', 'input': np.zeros((3,128,128), np.float32),
               'target': np.arange(4.), 'after_mu': np.zeros(3), 'after_scale': 1.}
    assert load_completed_donor(tmp_path, 'fit/a', 'contract') is None
    saved = save_completed_donor(tmp_path, 'fit/a', 'contract', [example], ['t0'])
    assert load_completed_donor(tmp_path, 'fit/a', 'contract') == saved
    with pytest.raises(ValueError, match='contract'):
        load_completed_donor(tmp_path, 'fit/a', 'changed')
    with pytest.raises(ValueError, match='reused'):
        save_completed_donor(tmp_path, 'fit/a', 'contract', [example], ['t0'])
    next(tmp_path.glob('*.npz')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        load_completed_donor(tmp_path, 'fit/a', 'contract')


def test_partial_file_is_not_completed(tmp_path):
    (tmp_path / 'orphan.npz.partial').write_bytes(b'incomplete')
    assert load_completed_donor(tmp_path, 'fit/a', 'contract') is None
    with pytest.raises(ValueError, match='ordered'):
        save_completed_donor(tmp_path, 'fit/a', 'contract', [], ['t0'])
