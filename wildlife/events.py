"""Class-level alert confirmation, timing, snapshots, and plain CSV logging."""
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2


class AlertGate:
    """Require consecutive processed frames; apply a cooldown per animal class."""

    def __init__(self, repeats=3, cooldown=30.0):
        self.repeats = repeats
        self.cooldown = cooldown
        self.counts = {}
        self.last_alert = {}

    def update(self, animals, event_seconds):
        """Use monotonic time for live video and footage seconds for replay."""
        self.counts = {name: self.counts.get(name, 0) + 1 for name in animals}
        confirmed = {}
        for name, score in animals.items():
            elapsed = event_seconds - self.last_alert.get(name, float("-inf"))
            # Timing is per class, not per tracked animal or bounding box.
            if self.counts[name] >= self.repeats and elapsed >= self.cooldown:
                confirmed[name] = score
                self.last_alert[name] = event_seconds
                self.counts[name] = 0
        return confirmed


def save_event(frame, label, confidence, source, footage_seconds, model_file,
               output_dir="outputs", camera_name="Debug"):
    """Write one snapshot and one CSV row; never claim replay is a live sighting."""
    root = Path(output_dir)
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    event_id = uuid4().hex
    snapshot = snapshots / f"{now:%Y%m%d_%H%M%S}_{event_id}.jpg"
    if not cv2.imwrite(str(snapshot), frame):
        raise OSError("Could not save event snapshot.")
    row = dict(event_id=event_id, processing_time=now.isoformat(timespec="seconds"),
               source=source, footage_seconds=footage_seconds, label=label,
               confidence=round(confidence, 4), model=str(model_file),
               snapshot=str(snapshot), camera_name=camera_name)
    log = root / "events.csv"
    needs_header = not log.exists() or log.stat().st_size == 0
    if not needs_header:
        # Upgrade old logs once, preserving every existing record and snapshot.
        with log.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_fields = reader.fieldnames or []
            if "camera_name" not in old_fields:
                upgraded = log.with_suffix(".upgrade.csv")
                with upgraded.open("w", newline="", encoding="utf-8") as target:
                    writer = csv.DictWriter(target, fieldnames=old_fields + ["camera_name"])
                    writer.writeheader()
                    for old_row in reader:
                        writer.writerow({**old_row, "camera_name": "Debug"})
            else:
                upgraded = None
        if upgraded is not None:
            upgraded.replace(log)
    with log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    return row
