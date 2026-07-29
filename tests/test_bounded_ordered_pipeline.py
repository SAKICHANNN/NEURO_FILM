from __future__ import annotations

import threading
import time

import pytest

from src.eval.bounded_ordered_pipeline import BoundedOrderedPipeline


def test_pipeline_consumes_out_of_order_completion_in_submit_order() -> None:
    consumed: list[int] = []
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def worker(value: int) -> int:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep((7 - value) * 0.001)
            return value
        finally:
            with lock:
                active -= 1

    pipeline = BoundedOrderedPipeline(
        worker=worker,
        consumer=consumed.append,
        max_workers=3,
        max_in_flight=4,
    )
    for value in range(8):
        pipeline.submit(value)
    stats = pipeline.finish()

    assert consumed == list(range(8))
    assert stats.submitted == stats.consumed == 8
    assert stats.maximum_in_flight == 4
    assert maximum_active <= 3


def test_pipeline_failure_stops_ordered_consumption_and_closes() -> None:
    consumed: list[int] = []

    def worker(value: int) -> int:
        if value == 2:
            raise RuntimeError("injected")
        return value

    pipeline = BoundedOrderedPipeline(
        worker=worker,
        consumer=consumed.append,
        max_workers=2,
        max_in_flight=3,
    )
    for value in range(4):
        pipeline.submit(value)

    with pytest.raises(RuntimeError, match="injected"):
        pipeline.finish()
    assert consumed == [0, 1]
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.submit(5)


@pytest.mark.parametrize(
    ("max_workers", "max_in_flight"),
    ((0, 1), (1, 0), (2, 1), (True, 2)),
)
def test_pipeline_rejects_invalid_bounds(
    max_workers: int, max_in_flight: int
) -> None:
    with pytest.raises(ValueError):
        BoundedOrderedPipeline(
            worker=lambda value: value,
            consumer=lambda value: None,
            max_workers=max_workers,
            max_in_flight=max_in_flight,
        )
