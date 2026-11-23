"""Scheduling, independent camera state, protocol formatting and resource cleanup."""
import asyncio
import csv
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from wildlife.protocol import StartConfig, footage_label, notification, trusted_file
from wildlife.runtime import DetectionSession, LatestFrameSlot, ModelRunner, UploadStore


def demo_config(cameras, **settings):
    return StartConfig(action="start", mode="demo", cameras=cameras, rate=1, **settings)


class PendingFrameTests(unittest.TestCase):
    def test_only_one_pending_frame_and_newest_wins(self):
        slot = LatestFrameSlot()
        for index in range(100):
            slot.put(str(index).encode(), index / 100)
        self.assertEqual(slot.take(1), b"99")
        self.assertIsNone(slot.take(1))
        self.assertEqual(slot.dropped, 99)

    def test_stale_frames_are_discarded(self):
        slot = LatestFrameSlot()
        slot.put(b"old", 0)
        self.assertIsNone(slot.take(2.01))
        self.assertEqual(slot.dropped, 1)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.uploads = UploadStore(self.output / "uploads")
        self.runner = ModelRunner()
        self.session = DetectionSession(self.runner, self.uploads, self.output)

    def tearDown(self):
        self.session.close()
        self.uploads.clear()
        self.runner.close()
        self.temp.cleanup()

    @patch("wildlife.runtime.load_model")
    @patch("wildlife.runtime.VideoReplay")
    @patch("wildlife.runtime.detect_frame")
    def test_camera_alert_isolation_and_camera_name_csv(self, detect, video, load):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detect.return_value = (frame, {"cow": .9}, [{"label": "cow", "confidence": .9, "xyxy": [1, 2, 8, 9]}])
        cameras = []
        for id in ("north", "creek"):
            token, path = self.uploads.create(id + ".avi")
            path.write_bytes(b"mock video")
            cameras.append({"id": id, "kind": "video", "upload_id": token})
        source1, source2 = MagicMock(), MagicMock()
        for source in (source1, source2):
            source.read.return_value = (frame, 18.4)
            source.event_seconds.return_value = 18.4
        video.side_effect = [source1, source2]
        self.session.configure(demo_config(cameras, repeats=2))
        north, creek = self.session.cameras.values()
        self.assertEqual(self.session.process_camera(north)["events"], [])
        self.assertEqual(self.session.process_camera(creek)["events"], [])
        result = self.session.process_camera(north)
        self.assertEqual(result["events"][0]["camera_name"], "North Trail")
        self.assertEqual(result["events"][0]["footage_time"], "00:18.40")
        self.assertEqual(creek.gate.last_alert, {})
        self.assertEqual(len(self.session.process_camera(creek)["events"]), 1)
        self.assertEqual(self.session.process_camera(north)["events"], [])
        with (self.output / "events.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["camera_name"] for row in rows], ["North Trail", "Creek Crossing"])
        self.assertTrue(all(row["source"].startswith("REPLAY") for row in rows))
        self.assertEqual(result["detections"][0]["xyxy"], [1, 2, 8, 9])
        self.assertIn("processing_time", result)

    @patch("wildlife.runtime.load_model")
    @patch("wildlife.runtime.VideoReplay")
    def test_four_sources_close_and_uploads_are_deleted(self, video, load):
        sources = [MagicMock() for _ in range(4)]
        video.side_effect = sources
        cameras, paths = [], []
        for id in ("north", "creek", "feeder", "gate"):
            token, path = self.uploads.create(id + ".mp4")
            path.write_bytes(b"mock")
            paths.append(path)
            cameras.append({"id": id, "kind": "video", "upload_id": token})
        self.session.configure(demo_config(cameras))
        self.session.close()
        self.session.close()  # Shutdown and disconnect cleanup may both run.
        for source in sources:
            source.close.assert_called_once()
        self.assertTrue(all(not path.exists() for path in paths))
        self.assertFalse(self.uploads.files)

    @patch("wildlife.runtime.load_model")
    @patch("wildlife.runtime.VideoReplay")
    def test_invalid_video_does_not_disable_other_camera(self, video, load):
        token, path = self.uploads.create("good.mp4")
        path.write_bytes(b"mock")
        self.session.configure(demo_config([
            {"id": "north", "kind": "video", "path": str(self.output / "missing.mp4")},
            {"id": "creek", "kind": "video", "upload_id": token}]))
        self.assertFalse(self.session.cameras["north"].active)
        self.assertTrue(self.session.cameras["north"].error)
        self.assertTrue(self.session.cameras["creek"].active)

    @patch("wildlife.runtime.load_model")
    def test_model_reuse_and_invalid_paths(self, load):
        first = self.runner.select_model("yolo11n.pt")
        self.assertIs(first, self.runner.select_model("yolo11n.pt"))
        load.assert_called_once()
        for path in ("https://example.com/model.pt", str(self.output / "missing.pt"), str(self.output)):
            with self.assertRaises(ValueError):
                self.runner.select_model(path)
        self.assertIs(self.runner.model, first)
        with self.assertRaises(ValueError):
            trusted_file("https://example.com/video.mp4", {".mp4"})

    def test_one_worker_never_runs_parallel_inference_jobs(self):
        count = 0
        maximum = 0
        lock = threading.Lock()

        def job():
            nonlocal count, maximum
            with lock:
                count += 1
                maximum = max(maximum, count)
            time.sleep(.01)
            with lock:
                count -= 1

        async def exercise():
            await asyncio.gather(*(self.runner.call(job) for _ in range(4)))
        asyncio.run(exercise())
        self.assertEqual(maximum, 1)

    def test_combined_notification_contains_both_timelines_and_snapshot(self):
        row = dict(event_id="test", camera_name="Main Gate", label="cow", confidence=.9,
                   source="REPLAY gate.mp4", footage_seconds=78.4,
                   processing_time="2026-11-09T12:22:00-06:00", snapshot="outputs/snapshots/test.jpg")
        message = notification(row, "gate")
        self.assertEqual(message["footage_time"], "01:18.40")
        self.assertEqual(message["snapshot_url"], "/snapshots/test.jpg")
        self.assertEqual(message["processing_time"], row["processing_time"])
        self.assertEqual(message["camera_name"], "Main Gate")
        self.assertIsNone(footage_label(None))


if __name__ == "__main__":
    unittest.main()
