from types import SimpleNamespace

import numpy as np
import pytest

from src.preprocess.fable_raw_eligibility import tag_numbers


def test_signed_and_unsigned_rational_decoding():
    for dtype in (5, 10):
        tag = SimpleNamespace(dtype=dtype, count=2, value=(0, 256, -1, 4), name='test')
        np.testing.assert_array_equal(tag_numbers(tag), [0., -.25])


@pytest.mark.parametrize('values,count', [((1, 0), 1), ((1, 2, 3), 2), ((float('nan'), 1), 1)])
def test_invalid_rational_rejected(values, count):
    with pytest.raises(ValueError):
        tag_numbers(SimpleNamespace(dtype=10, count=count, value=values, name='test'))
