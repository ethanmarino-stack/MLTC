"""Validated messages shared by the server and browser dashboard."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

CAMERA_NAMES = {"north": "North Trail", "creek": "Creek Crossing",
                "feeder": "Feeder Area", "gate": "Main Gate", "debug": "Debug"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


class CameraConfig(BaseModel):
    """Paths are explicit user-selected files, never URLs or capture-device IDs."""
    model_config = ConfigDict(extra="forbid")
    id: Literal["debug", "north", "creek", "feeder", "gate"]
    kind: Literal["webcam", "screen", "video"]
    path: str = Field(default="", max_length=500)
    upload_id: str = Field(default="", max_length=40)
    name: str = Field(default="", max_length=120)


class StartConfig(BaseModel):
    """Validate settings in Python too; browser sliders are only convenience UI."""
    model_config = ConfigDict(extra="forbid")
    action: Literal["start"]
    mode: Literal["debug", "demo"]
    model: str = Field(default="yolo11n.pt", min_length=1, max_length=500)
    confidence: float = Field(default=0.5, ge=0.05, le=0.95)
    rate: int = Field(default=3, ge=1, le=10)
    repeats: int = Field(default=3, ge=2, le=20)
    cooldown: float = Field(default=30, ge=1, le=3600)
    loop: bool = False
    display_mode: Literal["live", "detection"] = "live"
    cameras: list[CameraConfig] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def valid_sources(self):
        ids = [camera.id for camera in self.cameras]
        if len(set(ids)) != len(ids):
            raise ValueError("Camera IDs must be unique.")
        if self.mode == "debug" and ids != ["debug"]:
            raise ValueError("Debug uses one debug camera.")
        if self.mode == "debug":
            self.display_mode = "detection"  # Debug keeps its existing annotated monitor.
        if self.mode == "demo" and (self.rate > 3 or "debug" in ids
                                   or any(c.kind != "video" for c in self.cameras)):
            raise ValueError("Demo uses replay cameras at 1–3 FPS each.")
        return self


def trusted_file(value: str, suffixes: set[str]) -> Path:
    """Accept existing local files only; OpenCV must never interpret a URL here."""
    path = Path(value).expanduser()
    if "://" in value or path.suffix.lower() not in suffixes or not path.is_file():
        raise ValueError("Select an existing trusted local file with a supported extension.")
    return path.resolve()


def footage_label(seconds: float | None) -> str | None:
    """Relative footage time, never an invented capture date."""
    if seconds is None:
        return None
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def notification(row: dict, camera_id: str) -> dict:
    """Format Debug and combined Demo notifications consistently."""
    return {"id": row["event_id"], "camera_id": camera_id,
            "camera_name": row["camera_name"], "title": f"{row['label'].capitalize()} spotted",
            "label": row["label"], "confidence": row["confidence"], "source": row["source"],
            "footage_seconds": row["footage_seconds"],
            "footage_time": footage_label(row["footage_seconds"]),
            "processing_time": row["processing_time"],
            "snapshot_url": "/snapshots/" + Path(row["snapshot"]).name}
