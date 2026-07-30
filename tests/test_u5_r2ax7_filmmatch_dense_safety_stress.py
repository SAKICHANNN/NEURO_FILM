import numpy as np

from src.eval.filmmatch_dense_safety_stress import _chunked_apply


class _Square:
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return rgb**2


def test_dense_stress_chunking_is_partition_exact() -> None:
    values = np.random.default_rng(101).uniform(size=(31, 3))
    assert np.array_equal(_chunked_apply(_Square(), values, 7), values**2)
