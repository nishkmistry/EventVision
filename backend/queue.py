import queue
from typing import Optional
from events.event import Event

class EventQueue:
    def __init__(self, maxsize: int = 1000):
        self._queue = queue.Queue(maxsize=maxsize)

    def put(self, event: Event, block: bool = True, timeout: Optional[float] = None):
        self._queue.put(event, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Event:
        return self._queue.get(block=block, timeout=timeout)

    def size(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
