"""OpenCV sources: one latest webcam frame, or time-based recording replay."""
import threading
import time

import cv2


class LatestWebcam:
    """Drain the camera continuously, replacing a single frame instead of queuing."""

    def __init__(self, index=0):
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("Cannot open webcam. Check Windows camera permissions and index.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.last_read = 0
        self.error = None
        self.stopped = threading.Event()
        self.last_access = time.monotonic()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        """Release even on disconnect; inactivity also handles an abandoned browser."""
        try:
            while not self.stopped.is_set():
                if time.monotonic() - self.last_access > 30:
                    break
                ok, frame = self.capture.read()
                if not ok:
                    self.error = "Webcam disconnected or could not read a frame."
                    break
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
        finally:
            self.capture.release()
            self.stopped.set()

    def read(self):
        """Return each new frame at most once, so repeats cannot confirm a stale image."""
        self.last_access = time.monotonic()
        if self.stopped.is_set():
            raise ValueError(self.error or "Camera closed after inactivity. Press Start again.")
        with self.lock:
            if self.frame is None or self.last_read == self.sequence:
                return None, None
            self.last_read = self.sequence
            return self.frame.copy(), None

    def close(self):
        """Signal the owning reader to release the camera and wait briefly for it."""
        self.stopped.set()
        self.thread.join(timeout=2)


class VideoReplay:
    """Seek to current footage time, skipping frames when inference falls behind."""

    def __init__(self, path, loop=False):
        self.capture = cv2.VideoCapture(str(path))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not self.capture.isOpened() or fps <= 0 or count <= 0:
            self.capture.release()
            raise ValueError("Cannot read video timing. Try an MP4/H.264 recording.")
        self.fps = fps
        self.frame_count = int(count)
        self.started = time.monotonic()
        self.last_index = -1
        self.loop = loop
        self.completed_loops = 0

    def start_at(self, monotonic_seconds: float):
        """Share one replay epoch across cameras, after browser media is ready."""
        self.started = monotonic_seconds
        self.last_index = -1
        self.completed_loops = 0

    def playback_seconds(self):
        """Current replay clock, including inference time; used only for display sync."""
        elapsed = max(0.0, time.monotonic() - self.started)
        duration = self.frame_count / self.fps
        return elapsed % duration if self.loop else min(elapsed, duration)

    def read(self):
        """Replay at approximately original speed; media time is relative, not a date."""
        index = int((time.monotonic() - self.started) * self.fps)
        if index >= self.frame_count:
            if not self.loop:
                raise EOFError("Replay finished.")
            # A loop resets playback position, but the alert clock stays continuous.
            self.completed_loops += index // self.frame_count
            index %= self.frame_count
            self.started = time.monotonic() - index / self.fps
            self.last_index = -1
        if index <= self.last_index:
            return None, None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            raise EOFError("Replay ended or the next frame could not be decoded.")
        self.last_index = index
        return frame, index / self.fps

    def event_seconds(self, footage_seconds):
        """Keep cooldowns moving forward when looped footage returns to zero."""
        return self.completed_loops * self.frame_count / self.fps + footage_seconds

    def close(self):
        """Release the file so Windows can remove the temporary upload."""
        self.capture.release()
