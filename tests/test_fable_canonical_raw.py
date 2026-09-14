import numpy as np
import pytest

from src.preprocess.fable_canonical_raw import linear16_to_q8


def test_linear16_encoding_endpoints_and_breakpoint():
    codes = np.array([0, 1, 205, 206, 32768, 65535], dtype=np.uint16)
    rgb = np.broadcast_to(codes[:, None, None], (len(codes), 1, 3))
    output = linear16_to_q8(rgb)
    np.testing.assert_array_equal(output[:, 0, 0], [0, 0, 10, 10, 188, 255])
    assert np.all(np.diff(output[:, 0, 0].astype(int)) >= 0)
    with pytest.raises(ValueError, match='uint16'):
        linear16_to_q8(rgb.astype(np.float32))
