import cv2
import threading
import queue
import time
from typing import Optional, Union, Tuple

class ThreadedCameraReader:
    """
    Background thread for reading video/camera frames asynchronously.
    Prevents UI locking, buffer build-up, and eliminates video lag.
    """
    def __init__(self, src: Union[int, str]):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None

    def start(self):
        if not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            # Keep queue fresh with latest frame
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[any]]:
        if not self.running or self.queue.empty():
            return False, None
        return True, self.queue.get()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
