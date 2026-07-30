import numpy as np

from src.eval.filmmatch_validation_scene import _apply_rows


class _Double:
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return rgb * 2.0


def test_validation_row_execution_is_partition_exact() -> None:
    image = np.random.default_rng(9).uniform(size=(17, 11, 3))
    assert np.array_equal(_apply_rows(_Double(), image, 4), image * 2.0)
