"""One shared model, sequential inference, and independent per-camera alert rules."""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
import threading
import logging
from uuid import uuid4

import cv2
import numpy as np

from wildlife.detection import detect_frame, load_model
from wildlife.events import AlertGate, save_event
from wildlife.media import SessionMedia
from wildlife.protocol import CAMERA_NAMES, VIDEO_SUFFIXES, StartConfig, footage_label, notification, trusted_file
from wildlife.sources import VideoReplay

MAX_JPEG_BYTES = 500_000


class LatestFrameSlot:
    """A replaceable single pending JPEG, never an accumulating frame queue."""

    def __init__(self):
        self.pending = None
        self.dropped = 0
        self.lock = threading.Lock()

    def put(self, jpeg: bytes, now: float):
        with self.lock:
            if self.pending is not None:
                self.dropped += 1  # A newer frame is more useful than an old frame.
            self.pending = (jpeg, now)

    def take(self, now: float, max_age=2.0):
        with self.lock:
            item, self.pending = self.pending, None
        if item is not None and now - item[1] > max_age:
            self.dropped += 1
            return None
        return item[0] if item else None


class UploadStore:
    """Track only files created by this process; never delete trusted source files."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.files = {}
        self.pending_delete = set()

    def create(self, filename: str):
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError("Use MP4, AVI, MOV, or MKV.")
        if len(self.files) >= 5:
            raise ValueError("Stop the current run before uploading more videos.")
        self.folder.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        path = self.folder / (token + suffix)
        self.files[token] = path
        return token, path

    def get(self, token):
        if token not in self.files:
            raise ValueError("Upload expired. Select and upload the video again.")
        return self.files[token]

    def remove(self, token):
        path = self.files.get(token)
        if path:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                # Stop can arrive during an upload on Windows. Retry once its
                # writing handle closes instead of leaking the temporary copy.
                self.pending_delete.add(token)
                return
            self.files.pop(token, None)
            self.pending_delete.discard(token)

    def clear(self):
        for token in list(self.files):
            self.remove(token)


class ModelRunner:
    """All model and video work uses one worker, keeping the web server responsive."""

    def __init__(self):
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wildlife")
        self.model = None
        self.model_path = None

    async def call(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.worker, function, *args)

    def select_model(self, model_path):
        path = str(trusted_file(model_path, {".pt"})) if model_path != "yolo11n.pt" else model_path
        if self.model_path != path:
            self.model = None  # Keep one selected model rather than a growing cache.
            self.model_path = None
            self.model = load_model(path)
            self.model_path = path
        return self.model

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.model = None


@dataclass
class CameraState:
    """Each camera owns its playback position, alert counts, cooldowns and error."""
    id: str
    name: str
    source_name: str
    gate: AlertGate
    source: VideoReplay | None = None
    upload_id: str = ""
    slot: LatestFrameSlot = field(default_factory=LatestFrameSlot)
    active: bool = True
    error: str | None = None
    next_due: float = 0
    last_processed: float | None = None
    media_url: str | None = None


class DetectionSession:
    """One local dashboard run. One scheduler visits its cameras sequentially."""

    def __init__(self, runner: ModelRunner, uploads: UploadStore, output: Path):
        self.runner, self.uploads, self.output = runner, uploads, output
        self.cameras = {}
        self.config = None
        self.stopping = False
        self.media = SessionMedia()

    @property
    def live_view(self):
        return self.config.mode == "demo" and self.config.display_mode == "live"

    def configure(self, config: StartConfig):
        """Open sources once; a bad feed does not disable the other feeds."""
        self.config = config
        self.stopping = False
        self.runner.select_model(config.model)
        for item in config.cameras:
            name = CAMERA_NAMES[item.id]
            live = item.kind != "video"
            source_name = ("LIVE screen capture" if item.kind == "screen" else
                           f"LIVE {item.name or 'Webcam'}") if live else f"REPLAY {item.name or name}"
            camera = CameraState(item.id, name, source_name,
                                 AlertGate(config.repeats, config.cooldown), upload_id=item.upload_id)
            self.cameras[item.id] = camera
            if not live:
                try:
                    path = self.uploads.get(item.upload_id) if item.upload_id else trusted_file(item.path, VIDEO_SUFFIXES)
                    camera.source = VideoReplay(path, loop=config.loop)
                    if self.live_view and not item.upload_id:
                        camera.media_url = self.media.register(item.id, path)
                except (ValueError, OSError):
                    camera.error = "Cannot open video. Check the local path, upload, and codec."
                    self.close_camera(camera)

    def close_camera(self, camera):
        self.media.remove_camera(camera.id)
        camera.active = False
        camera.slot.pending = None
        if camera.source:
            camera.source.close()
            camera.source = None
        if camera.upload_id:
            self.uploads.remove(camera.upload_id)
            camera.upload_id = ""

    def close(self):
        """Wait for inference before closing, so a file is never closed mid-read."""
        self.stopping = True
        for camera in self.cameras.values():
            self.close_camera(camera)
        self.media.clear()

    def start_playback(self, delay_seconds=.25):
        """Start together after the browser has loaded media; no inference rate change."""
        epoch = time.monotonic() + delay_seconds
        for camera in self.cameras.values():
            if camera.active and camera.source:
                camera.source.start_at(epoch)
                camera.next_due = epoch

    def process_camera(self, camera: CameraState):
        """Decode a sampled frame, reuse the CNN, and apply existing alert rules."""
        footage = None
        if camera.source:
            frame, footage = camera.source.read()
        else:
            jpeg = camera.slot.take(time.monotonic())
            if jpeg is None:
                return {"type": "frame_dropped", "camera_id": camera.id}
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or max(frame.shape[:2]) > 1920:
                raise ValueError("Send a valid JPEG resized to at most 1920 pixels.")
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if max(height, width) > 960:
            frame = cv2.resize(frame, (int(width * 960 / max(height, width)),
                                       int(height * 960 / max(height, width))))
        started = time.perf_counter()
        annotated, animals, boxes = detect_frame(self.runner.model, frame,
                                                self.config.confidence, include_boxes=True)
        inference_ms = (time.perf_counter() - started) * 1000
        now = time.monotonic()
        fps = 0 if camera.last_processed is None else 1 / max(now - camera.last_processed, .001)
        camera.last_processed = now
        # A continuous footage clock keeps cooldowns moving forward across loops.
        timeline = camera.source.event_seconds(footage) if camera.source else now
        events = []
        for label, score in camera.gate.update(animals, timeline).items():
            row = save_event(annotated, label, score, camera.source_name, footage,
                             self.config.model, self.output, camera_name=camera.name)
            events.append(notification(row, camera.id))
        image = None
        if not self.live_view:
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise OSError("Cannot encode annotated frame.")
            image = base64.b64encode(encoded).decode("ascii")
        return {"type": "frame", "camera_id": camera.id, "camera_name": camera.name,
                "source": camera.source_name, "status": "Replaying" if footage is not None else "Live",
                "processing_time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "footage_seconds": footage, "footage_time": footage_label(footage),
                "inference_ms": round(inference_ms, 1), "fps": round(fps, 2),
                "dropped_frames": camera.slot.dropped, "detections": boxes,
                "animals": animals, "events": events,
                "display_mode": "live" if self.live_view else "detection",
                "playback_seconds": float(camera.source.playback_seconds()) if self.live_view else None,
                "image": image}

    async def run(self, send):
        """No concurrent GPU jobs: await each camera before visiting the next one."""
        logging.getLogger("uvicorn.error").info("DetectionSession.run started display_mode=%s cameras=%d",
                                               self.config.display_mode, len(self.cameras))
        while not self.stopping:
            for camera in self.cameras.values():
                if self.stopping:
                    break
                now = time.monotonic()
                if not camera.active or now < camera.next_due:
                    continue
                if camera.source is None and camera.slot.pending is None:
                    continue
                camera.next_due = now + 1 / self.config.rate
                try:
                    message = await self.runner.call(self.process_camera, camera)
                    if message:
                        await send(message)
                except EOFError:
                    await self.runner.call(self.close_camera, camera)
                    await send({"type": "camera_status", "camera_id": camera.id, "status": "Replay finished"})
                except Exception:
                    logging.getLogger(__name__).exception("Camera processing failed: %s", camera.id)
                    camera.error = "Processing stopped. Check video/weights, memory, and output-folder access."
                    await self.runner.call(self.close_camera, camera)
                    await send({"type": "error", "camera_id": camera.id, "message": camera.error})
            if not any(camera.active for camera in self.cameras.values()):
                await send({"type": "finished"})
                return
            await asyncio.sleep(.02)
