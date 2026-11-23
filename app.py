"""Local server: python -m uvicorn app:app --host 127.0.0.1 --port 8000"""
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
from pathlib import Path
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from anyio import CancelScope, ClosedResourceError, BrokenResourceError

from wildlife.protocol import StartConfig
from wildlife.runtime import DetectionSession, MAX_JPEG_BYTES, ModelRunner, UploadStore

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
logger = logging.getLogger(__name__)
playback_logger = logging.getLogger("uvicorn.error")
PLAYBACK_PROTOCOL = "demo-display-mode-1"


@asynccontextmanager
async def lifespan(application):
    """Load weights lazily; release sources, uploads, sockets and the worker on exit."""
    application.state.runner = ModelRunner()
    application.state.uploads = UploadStore(OUTPUT / "uploads")
    application.state.connection = None
    try:
        yield
    finally:
        connection = application.state.connection
        if connection:
            await connection.stop()
            with suppress(RuntimeError, WebSocketDisconnect, ClosedResourceError, BrokenResourceError):
                await connection.socket.close(code=1001)
        application.state.uploads.clear()
        application.state.runner.close()


app = FastAPI(title="Texas Wildlife Camera", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


def check_origin(headers):
    """A local page can select trusted files; unrelated websites must not control it."""
    origin = headers.get("origin")
    if origin and origin != "http://" + headers.get("host", ""):
        raise ValueError("Open this dashboard directly on localhost.")


@app.middleware("http")
async def local_writes(request: Request, call_next):
    if request.method in {"POST", "DELETE"}:
        try:
            check_origin(request.headers)
        except ValueError as error:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": str(error)}, status_code=403)
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        # This local editable app must not mix an old controller with a new page.
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
def health():
    return {"status": "ready", "model": app.state.runner.model_path or "Not loaded",
            "playback_protocol": PLAYBACK_PROTOCOL}


@app.post("/api/uploads")
async def upload_video(file: UploadFile = File(...)):
    """Copy bounded chunks to disk, rather than keeping a whole video in RAM."""
    token = None
    try:
        token, path = app.state.uploads.create(file.filename or "")
        size = 0
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 200 * 1024 * 1024:
                    raise ValueError("Video exceeds 200 MB. Use a trusted local path in Demo.")
                await asyncio.to_thread(target.write, chunk)
        if not size:
            raise ValueError("The uploaded video is empty.")
        if token in app.state.uploads.pending_delete:
            raise ValueError("Upload was cancelled by Stop.")
        return {"upload_id": token, "name": Path(file.filename).name}
    except (ValueError, OSError) as error:
        if token:
            app.state.uploads.remove(token)
        raise HTTPException(400, "Video upload failed. Check file type, size, and output-folder access.") from error
    finally:
        await file.close()


@app.delete("/api/uploads/{token}")
def delete_upload(token: str):
    connection = app.state.connection
    if connection and connection.session and any(c.upload_id == token for c in connection.session.cameras.values()):
        raise HTTPException(409, "Stop playback before removing this upload.")
    app.state.uploads.remove(token)
    return {"status": "removed"}


@app.get("/snapshots/{filename}")
def snapshot(filename: str):
    folder = (OUTPUT / "snapshots").resolve()
    path = (folder / filename).resolve()
    if path.parent != folder or path.suffix.lower() != ".jpg" or not path.is_file():
        raise HTTPException(404, "Snapshot not found.")
    return FileResponse(path)


@app.api_route("/api/media/{token}", methods=["GET", "HEAD"])
def session_video(token: str):
    """Byte-range video delivery, limited to current validated Demo selections.

    Starlette FileResponse implements Range/If-Range, partial (206) responses,
    and HEAD without reading a complete video into application memory.
    """
    connection = app.state.connection
    session = connection.session if connection else None
    path = session.media.get(token) if session and not session.stopping else None
    if path is None:
        raise HTTPException(404, "Video reference is invalid or expired.")
    media_type = {".mp4": "video/mp4", ".mov": "video/quicktime",
                  ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}[path.suffix.lower()]
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store",
                        "Cross-Origin-Resource-Policy": "same-origin", "X-Content-Type-Options": "nosniff"})


class DashboardConnection:
    """Own one WebSocket/run; serialize sends and wait for safe Stop cleanup."""

    def __init__(self, socket, runner, uploads):
        self.socket, self.runner, self.uploads = socket, runner, uploads
        self.session = None
        self.task = None
        self.send_lock = asyncio.Lock()
        self.stop_lock = asyncio.Lock()

    async def send(self, message):
        async with self.send_lock:
            await self.socket.send_json(message)

    async def stop(self):
        async with self.stop_lock:
            if self.session:
                self.session.stopping = True
                if self.task:
                    await asyncio.gather(self.task, return_exceptions=True)
                await self.runner.call(self.session.close)
            self.session, self.task = None, None
            if app.state.connection is self:
                self.uploads.clear()

    async def command(self, data):
        action = data.get("action") if isinstance(data, dict) else None
        if action == "stop":
            await self.stop()
            if app.state.connection is self:
                app.state.connection = None
            await self.send({"type": "stopped"})
            await self.socket.close(code=1000)
        elif action == "start":
            if data.get("mode") == "demo" and "display_mode" not in data:
                # Old JS omits this field. Defaulting it to live silently waits
                # for a play command that the old client never knows to send.
                raise ValueError("Demo display mode is missing. Fully restart the server and hard-refresh the dashboard.")
            config = StartConfig.model_validate(data)
            if config.mode == "demo":
                playback_logger.info("Demo start requested display_mode=%s", config.display_mode)
            if self.session:
                raise ValueError("Stop the current run before starting another.")
            self.session = DetectionSession(self.runner, self.uploads, OUTPUT)
            try:
                await self.runner.call(self.session.configure, config)
            except Exception:
                await self.stop()
                raise ValueError("Cannot load model. Choose yolo11n.pt or a trusted local detection .pt file.")
            await self.send({"type": "started", "model": config.model, "mode": config.mode,
                             "display_mode": config.display_mode,
                             "cameras": [{"id": c.id, "status": c.error or "Ready", "error": bool(c.error),
                                          "media_url": c.media_url}
                                         for c in self.session.cameras.values()]})
            if config.mode == "demo":
                playback_logger.info("Demo started display_mode=%s media_registered=%s", config.display_mode,
                                     {c.id: bool(c.media_url) for c in self.session.cameras.values()})
            # Live view waits for the browser's loaded media before starting the
            # existing detection scheduler. Detection view retains immediate start.
            if not self.session.live_view:
                if config.mode == "demo":
                    playback_logger.info("Demo detection scheduler starting immediately (no play handshake)")
                self.task = asyncio.create_task(self.session.run(self.send))
        elif action == "play":
            if not self.session or not self.session.live_view or self.task is not None:
                raise ValueError("Playback is not waiting to start.")
            await self.runner.call(self.session.start_playback)
            await self.send({"type": "playing", "starts_in_ms": 250})
            playback_logger.info("Demo play accepted; starting detection scheduler")
            self.task = asyncio.create_task(self.session.run(self.send))
        else:
            raise ValueError("Unknown action. Use start or stop.")

    async def browser_frame(self, jpeg):
        if not self.session or "debug" not in self.session.cameras:
            raise ValueError("Start a browser webcam or screen source first.")
        camera = self.session.cameras["debug"]
        if camera.source or not camera.active or len(jpeg) > MAX_JPEG_BYTES:
            raise ValueError("Invalid browser frame or frame exceeds 500 KB.")
        # Replace one pending frame while the single worker processes the current one.
        camera.slot.put(jpeg, time.monotonic())


@app.websocket("/ws")
async def dashboard(socket: WebSocket):
    try:
        check_origin(socket.headers)
    except ValueError:
        await socket.close(code=1008)
        return
    await socket.accept()
    if app.state.connection is not None:
        await socket.send_json({"type": "error", "message": "Another dashboard is connected. Close that tab first."})
        await socket.close(code=1008)
        return
    connection = DashboardConnection(socket, app.state.runner, app.state.uploads)
    app.state.connection = connection
    try:
        await connection.send({"type": "connected"})
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                break
            try:
                if message.get("bytes") is not None:
                    await connection.browser_frame(message["bytes"])
                else:
                    text = message.get("text", "")
                    if len(text) > 8192:
                        raise ValueError("Control message is too large.")
                    await connection.command(json.loads(text))
            except (ValidationError, json.JSONDecodeError):
                await connection.send({"type": "error", "message": "Invalid request. Check source, paths and settings; stop before restarting."})
            except ValueError as error:
                await connection.send({"type": "error", "message": str(error)})
            except Exception:
                logger.exception("Dashboard command failed")
                await connection.stop()
                await connection.send({"type": "error", "message": "Request failed. Stop and check local files or server output."})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        # A dropped browser (or cancelled ASGI test client) must not cancel file
        # cleanup halfway through, leaving usable media tokens or OpenCV handles.
        with CancelScope(shield=True):
            await connection.stop()
        if app.state.connection is connection:
            app.state.connection = None
