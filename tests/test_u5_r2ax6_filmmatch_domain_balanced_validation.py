import numpy as np

from src.eval.filmmatch_validation_scene import _apply_rows


class _Offset:
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return rgb + 0.1


def test_ax6_row_execution_is_partition_exact() -> None:
    image = np.random.default_rng(91).uniform(0.0, 0.8, size=(19, 13, 3))
    assert np.array_equal(_apply_rows(_Offset(), image, 5), image + 0.1)
