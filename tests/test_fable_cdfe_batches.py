import numpy as np
import pytest

from src.training.fable_cdfe_batches import shuffled_epoch_batches


def test_fixed_training_frame_replays_without_epoch_omissions():
    kwargs = dict(examples=8192, batch_size=32, steps=4000, seed=2026091403)
    first = np.stack(list(shuffled_epoch_batches(**kwargs)))
    replay = np.stack(list(shuffled_epoch_batches(**kwargs)))
    assert first.shape == (4000, 32)
    np.testing.assert_array_equal(first, replay)
    for epoch in range(15):
        np.testing.assert_array_equal(np.sort(first[epoch*256:(epoch+1)*256].ravel()),
                                      np.arange(8192))
    assert len(np.unique(first[15*256:])) == 160*32
    assert not np.array_equal(first[:256], first[256:512])
    different = next(shuffled_epoch_batches(**(kwargs | {'seed': 2026091404})))
    assert not np.array_equal(first[0], different)


def test_incomplete_batch_is_not_silently_dropped():
    with pytest.raises(ValueError, match='complete'):
        next(shuffled_epoch_batches(examples=33, batch_size=32, steps=1, seed=1))
