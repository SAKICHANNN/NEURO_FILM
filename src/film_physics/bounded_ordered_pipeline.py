"""Bounded parallel work with deterministic submission-order consumption."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class BoundedOrderedPipelineStats:
    submitted: int
    consumed: int
    maximum_in_flight: int


class BoundedOrderedPipeline(Generic[InputT, OutputT]):
    """Run independent work in parallel and consume results in submit order."""

    def __init__(
        self,
        *,
        worker: Callable[[InputT], OutputT],
        consumer: Callable[[OutputT], None],
        max_workers: int,
        max_in_flight: int,
    ) -> None:
        if (
            isinstance(max_workers, bool)
            or isinstance(max_in_flight, bool)
            or not isinstance(max_workers, int)
            or not isinstance(max_in_flight, int)
            or max_workers <= 0
            or max_in_flight <= 0
            or max_in_flight < max_workers
        ):
            raise ValueError(
                "max_in_flight must be at least max_workers and both "
                "must be positive integers"
            )
        self._worker = worker
        self._consumer = consumer
        self._max_in_flight = max_in_flight
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: Deque[Future[OutputT]] = deque()
        self._submitted = 0
        self._consumed = 0
        self._maximum_in_flight = 0
        self._closed = False

    def submit(self, value: InputT) -> None:
        if self._closed:
            raise RuntimeError("pipeline is closed")
        try:
            if len(self._pending) >= self._max_in_flight:
                self._consume_oldest()
        except BaseException:
            self._closed = True
            self._cancel_pending()
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise
        self._pending.append(self._executor.submit(self._worker, value))
        self._submitted += 1
        self._maximum_in_flight = max(
            self._maximum_in_flight, len(self._pending)
        )

    def finish(self) -> BoundedOrderedPipelineStats:
        if self._closed:
            raise RuntimeError("pipeline is closed")
        try:
            while self._pending:
                self._consume_oldest()
        except BaseException:
            self._cancel_pending()
            raise
        finally:
            self._closed = True
            self._executor.shutdown(
                wait=True, cancel_futures=True
            )
        return BoundedOrderedPipelineStats(
            submitted=self._submitted,
            consumed=self._consumed,
            maximum_in_flight=self._maximum_in_flight,
        )

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_pending()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _consume_oldest(self) -> None:
        future = self._pending.popleft()
        try:
            result = future.result()
            self._consumer(result)
        except BaseException:
            self._cancel_pending()
            raise
        self._consumed += 1

    def _cancel_pending(self) -> None:
        while self._pending:
            self._pending.popleft().cancel()
