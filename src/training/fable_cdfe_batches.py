from collections.abc import Iterator

import numpy as np


def shuffled_epoch_batches(*, examples: int, batch_size: int, steps: int,
                           seed: int) -> Iterator[np.ndarray]:
    if examples <= 0 or batch_size <= 0 or examples % batch_size or steps <= 0:
        raise ValueError('positive complete batches and steps required')
    rng = np.random.Generator(np.random.PCG64(seed))
    emitted = 0
    while emitted < steps:
        order = rng.permutation(examples)
        for start in range(0, examples, batch_size):
            yield order[start:start + batch_size].copy()
            emitted += 1
            if emitted == steps:
                return
