"""Small behavioral checks; no real wildlife data or webcam is used."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from wildlife.dataset import check_dataset
from wildlife.events import AlertGate, save_event
from wildlife.sources import LatestWebcam, VideoReplay


class AlertTests(unittest.TestCase):
    def test_old_csv_schema_upgrades_without_losing_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fields = ["event_id", "processing_time", "source", "footage_seconds", "label", "confidence", "model", "snapshot"]
            with (root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(event_id="existing", source="REPLAY old.mp4", label="cow"))
            save_event(np.zeros((16, 16, 3), dtype=np.uint8), "cow", .9,
                       "REPLAY new.mp4", 18.4, "yolo11n.pt", folder, camera_name="North Trail")
            with (root / "events.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["event_id"], "existing")
            self.assertEqual(rows[0]["camera_name"], "Debug")
            self.assertEqual(rows[1]["camera_name"], "North Trail")

    def test_consecutive_frames_and_independent_class_cooldowns(self):
        gate = AlertGate(repeats=2, cooldown=10)
        self.assertEqual(gate.update({"cow": .9}, 0), {})
        self.assertEqual(gate.update({}, 1), {})
        self.assertEqual(gate.update({"cow": .9}, 2), {})
        self.assertEqual(gate.update({"cow": .9}, 3), {"cow": .9})
        self.assertEqual(gate.update({"cow": .9, "deer": .8}, 4), {})
        self.assertEqual(gate.update({"cow": .9, "deer": .8}, 5), {"deer": .8})
        self.assertEqual(gate.update({"cow": .9}, 12.9), {})
        self.assertEqual(gate.update({"cow": .9}, 13), {"cow": .9})

    def test_csv_appends_and_unique_snapshot_per_event(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            first = save_event(frame, "cow", .9, "REPLAY clip.mp4", 12.0, "yolo11n.pt", folder)
            second = save_event(frame, "cow", .8, "LIVE webcam 0", None, "yolo11n.pt", folder)
            with (Path(folder) / "events.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["footage_seconds"], "12.0")
            self.assertEqual(rows[1]["footage_seconds"], "")
            self.assertNotEqual(first["snapshot"], second["snapshot"])
            self.assertEqual(len(list((Path(folder) / "snapshots").glob("*.jpg"))), 2)


class DatasetTests(unittest.TestCase):
    def test_missing_labels_rejected_and_duplicate_splits_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_folder = root / "images/test"
            image_folder.mkdir(parents=True)
            image = image_folder / "fixture.jpg"
            image.write_bytes(b"test fixture only; not training data")
            config = root / "dataset.yaml"
            settings = dict(path=str(root), test="images/test",
                            names={0: "deer", 1: "wild_boar", 2: "cow"})
            config.write_text(yaml.safe_dump(settings), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing annotation"):
                check_dataset(config, ("test",))
            labels = root / "labels/test"
            labels.mkdir(parents=True)
            (labels / "fixture.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            check_dataset(config, ("test",))
            with self.assertRaisesRegex(ValueError, "all three classes"):
                check_dataset(config, ("test",), require_all_classes=True)
            train = root / "images/train"
            train.mkdir(parents=True)
            (train / "copied.jpg").write_bytes(image.read_bytes())
            settings["train"] = "images/train"
            config.write_text(yaml.safe_dump(settings), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate image"):
                check_dataset(config, ("test",))


class ReplayTests(unittest.TestCase):
    @patch("wildlife.sources.cv2.VideoCapture")
    @patch("wildlife.sources.time.monotonic")
    def test_loop_rewinds_footage_without_rewinding_alert_clock(self, clock, factory):
        import cv2
        capture = factory.return_value
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 90
        capture.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))
        clock.return_value = 100
        replay = VideoReplay("fixture.avi", loop=True)
        replay.read()
        clock.return_value = 103
        _, seconds = replay.read()
        self.assertEqual(seconds, 0)
        self.assertEqual(replay.event_seconds(seconds), 3)
        clock.return_value = 106
        _, seconds = replay.read()
        self.assertEqual(replay.event_seconds(seconds), 6)
        replay.close()

    @patch("wildlife.sources.cv2.VideoCapture")
    @patch("wildlife.sources.time.monotonic")
    def test_skips_ahead_and_ends_without_reading_past_file(self, clock, factory):
        import cv2
        capture = factory.return_value
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 90
        capture.read.return_value = (True, np.zeros((4, 4, 3), dtype=np.uint8))
        clock.return_value = 100
        replay = VideoReplay("fixture.avi")
        _, seconds = replay.read()
        self.assertEqual(seconds, 0)
        self.assertEqual(replay.read(), (None, None))
        clock.return_value = 102
        _, seconds = replay.read()
        self.assertEqual(seconds, 2)
        capture.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 60)
        clock.return_value = 103
        with self.assertRaises(EOFError):
            replay.read()
        replay.close()
        capture.release.assert_called_once()

    @patch("wildlife.sources.cv2.VideoCapture")
    def test_webcam_releases_on_open_failure(self, factory):
        factory.return_value.isOpened.return_value = False
        with self.assertRaisesRegex(ValueError, "Cannot open webcam"):
            LatestWebcam(9)
        factory.return_value.release.assert_called_once()

    @patch("wildlife.sources.cv2.VideoCapture")
    def test_webcam_releases_after_read_failure(self, factory):
        factory.return_value.isOpened.return_value = True
        factory.return_value.read.return_value = (False, None)
        source = LatestWebcam()
        source.thread.join(timeout=2)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            source.read()
        source.close()
        factory.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
