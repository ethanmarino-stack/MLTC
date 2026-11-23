"""CPU-only Live view/API checks. No model weights, GPU jobs or real server."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app as server
from wildlife.protocol import StartConfig
from wildlife.sources import VideoReplay


class FakeReplay:
    """Deterministic replay clock for transport tests; not a wildlife recording."""
    instances = []

    def __init__(self, path, loop=False):
        self.path, self.loop, self.closed = path, loop, False
        self.started, self.start_count, self.read_count = None, 0, 0
        self.instances.append(self)

    def read(self):
        self.read_count += 1
        return np.zeros((16, 16, 3), dtype=np.uint8), 2.5

    def playback_seconds(self):
        return 3.0  # The display clock may advance while a detection is computed.

    def event_seconds(self, seconds):
        return seconds

    def start_at(self, epoch):
        self.started = epoch
        self.start_count += 1

    def close(self):
        self.closed = True


class LiveViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.video = self.folder / "selected.mp4"
        self.content = bytes(range(256))
        self.video.write_bytes(self.content)
        self.arbitrary = self.folder / "not-selected.mp4"
        self.arbitrary.write_bytes(b"private unselected file")
        self.output_patch = patch.object(server, "OUTPUT", self.folder / "output")
        self.output_patch.start()
        self.model_patch = patch("wildlife.runtime.load_model", return_value=object())
        self.load = self.model_patch.start()
        self.replay_patch = patch("wildlife.runtime.VideoReplay", side_effect=FakeReplay)
        self.replay_patch.start()
        FakeReplay.instances = []
        annotated = np.full((16, 16, 3), 255, dtype=np.uint8)
        self.detect_patch = patch("wildlife.runtime.detect_frame", return_value=(annotated, {"cow": .9},
                                  [{"label": "cow", "confidence": .9, "xyxy": [1, 1, 15, 15]}]))
        self.detect = self.detect_patch.start()

    def tearDown(self):
        self.detect_patch.stop()
        self.replay_patch.stop()
        self.model_patch.stop()
        self.output_patch.stop()
        self.temp.cleanup()

    def config(self, display_mode="live", **overrides):
        return {"action": "start", "mode": "demo", "display_mode": display_mode,
                "rate": 3, "repeats": 2, "loop": True,
                "cameras": [{"id": "north", "kind": "video", "path": str(self.video)}], **overrides}

    def stop_socket(self, socket):
        socket.send_json({"action": "stop"})
        while socket.receive_json()["type"] != "stopped":
            pass

    def test_configuration_defaults_and_rate_limits(self):
        config = self.config()
        config.pop("display_mode")
        self.assertEqual(StartConfig.model_validate(config).display_mode, "live")
        self.assertEqual(StartConfig.model_validate(self.config("detection")).display_mode, "detection")
        for changes in ({"display_mode": "anything"}, {"rate": 30}, {"rate": 4}):
            with self.assertRaises(ValidationError):
                StartConfig.model_validate({**config, **changes})
        debug = {"action": "start", "mode": "debug", "cameras": [{"id": "debug", "kind": "webcam"}]}
        self.assertEqual(StartConfig.model_validate(debug).display_mode, "detection")

    def test_legacy_start_cannot_silently_default_to_live_and_wait_at_ready(self):
        with TestClient(server.app) as client:
            for visible_selector in ("live", "detection"):
                with self.subTest(selector=visible_selector), client.websocket_connect("/ws") as socket:
                    socket.receive_json()
                    legacy_request = self.config(visible_selector)
                    legacy_request.pop("display_mode")  # Cached JS before the playback-mode field.
                    socket.send_json(legacy_request)
                    message = socket.receive_json()
                    self.assertEqual(message["type"], "error")
                    self.assertIn("hard-refresh", message["message"])
                    self.assertIsNone(server.app.state.connection.session)
                    self.assertIsNone(server.app.state.connection.task)
                    self.load.assert_not_called()
                    self.stop_socket(socket)

    def test_full_startup_order_for_each_mode_and_source_type(self):
        with TestClient(server.app) as client:
            for display_mode in ("detection", "live"):
                for source_type in ("path", "upload"):
                    with self.subTest(mode=display_mode, source=source_type), client.websocket_connect("/ws") as socket:
                        socket.receive_json()
                        camera = {"id": "north", "kind": "video"}
                        if source_type == "path":
                            camera["path"] = str(self.video)
                        else:
                            token = client.post("/api/uploads", files={"file": ("local.mp4", self.content)}).json()["upload_id"]
                            camera["upload_id"] = token
                        trace = ["start"]
                        socket.send_json(self.config(display_mode, cameras=[camera]))
                        started = socket.receive_json()
                        trace.append(started["type"])
                        self.assertEqual(started["display_mode"], display_mode)
                        if display_mode == "live":
                            self.assertIsNone(server.app.state.connection.task)
                            if source_type == "path":
                                url = started["cameras"][0]["media_url"]
                                partial = client.get(url, headers={"range": "bytes=0-15"})
                                self.assertEqual(partial.status_code, 206)
                                self.assertEqual(partial.content, self.content[:16])
                            trace.append("play")
                            socket.send_json({"action": "play"})
                            trace.append(socket.receive_json()["type"])
                        else:
                            self.assertIsNotNone(server.app.state.connection.task)
                        first, second = socket.receive_json(), socket.receive_json()
                        trace.extend([first["type"], second["type"]])
                        self.assertEqual(trace, ["start", "started"] +
                                         (["play", "playing"] if display_mode == "live" else []) + ["frame", "frame"])
                        self.assertEqual(bool(first["image"]), display_mode == "detection")
                        self.assertEqual(first["footage_seconds"], 2.5)
                        self.assertTrue(second["events"])
                        self.stop_socket(socket)

    def test_current_asset_responses_and_backend_restart_marker(self):
        with TestClient(server.app) as client:
            page = client.get("/")
            self.assertEqual(page.headers["cache-control"], "no-store")
            self.assertEqual(client.get("/api/health").json()["playback_protocol"], "demo-display-mode-1")
            for name in ("app.js", "demo-playback.js", "styles.css"):
                response = client.get("/static/" + name + "?v=live-view-3")
                self.assertEqual(response.content, (server.ROOT / "static" / name).read_bytes())
                self.assertEqual(response.headers["cache-control"], "no-store")
                self.assertIn(name + "?v=live-view-3", page.text)
            for id in ("north", "creek", "feeder", "gate"):
                for suffix in ("video", "frame", "empty", "view-badge", "playback-note"):
                    self.assertIn(f'id="{id}-{suffix}"', page.text)

    def test_selected_token_ranges_head_and_expiry(self):
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json(self.config())
            started = socket.receive_json()
            url = started["cameras"][0]["media_url"]
            self.assertTrue(url.startswith("/api/media/"))
            self.assertNotIn("selected", url)
            self.assertNotIn(str(self.folder), url)
            self.assertEqual(client.get(url).content, self.content)
            head = client.head(url)
            self.assertEqual(head.status_code, 200)
            self.assertEqual(head.content, b"")
            self.assertEqual(head.headers["accept-ranges"], "bytes")
            partial = client.get(url, headers={"range": "bytes=10-19"})
            self.assertEqual(partial.status_code, 206)
            self.assertEqual(partial.content, self.content[10:20])
            self.assertEqual(partial.headers["content-range"], "bytes 10-19/256")
            self.assertEqual(partial.headers["cache-control"], "no-store")
            self.assertEqual(client.get(url, headers={"range": "bytes=9999-"}).status_code, 416)
            self.assertEqual(client.get("/api/media/not-a-token", params={"path": str(self.arbitrary)}).status_code, 404)
            self.assertEqual(client.get("/api/media/" + self.arbitrary.name).status_code, 404)
            self.assertEqual(client.get("/api/media/%2E%2E%2Fnot-selected.mp4").status_code, 404)
            self.stop_socket(socket)
            self.assertEqual(client.get(url).status_code, 404)
            self.assertTrue(FakeReplay.instances[0].closed)
            self.assertTrue(self.video.exists())

    def test_live_view_handshake_events_and_annotated_snapshots_without_display_jpeg(self):
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json(self.config())
            started = socket.receive_json()
            self.assertEqual(started["display_mode"], "live")
            self.detect.assert_not_called()  # No scheduler work before browser readiness.
            self.assertEqual(FakeReplay.instances[0].read_count, 0)
            socket.send_json({"action": "play"})
            self.assertEqual(socket.receive_json()["type"], "playing")
            first, second = socket.receive_json(), socket.receive_json()
            self.assertIsNone(first["image"])
            self.assertEqual(second["display_mode"], "live")
            self.assertEqual(second["events"][0]["camera_name"], "North Trail")
            self.assertEqual(second["events"][0]["footage_seconds"], 2.5)
            self.assertEqual(second["playback_seconds"], 3.0)
            self.assertEqual(second["events"][0]["confidence"], .9)
            self.assertIn("processing_time", second["events"][0])
            snapshot = client.get(second["events"][0]["snapshot_url"])
            decoded = cv2.imdecode(np.frombuffer(snapshot.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(decoded.mean(), 250)  # The annotated array, not the raw zero array.
            with (self.folder / "output/events.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["camera_name"], "North Trail")
            self.assertEqual(row["footage_seconds"], "2.5")
            self.assertEqual(len(server.app.state.connection.session.media.references), 1)
            self.assertEqual(FakeReplay.instances[0].start_count, 1)
            socket.send_json({"action": "play"})
            while socket.receive_json()["type"] != "error":
                pass
            self.assertEqual(FakeReplay.instances[0].start_count, 1)
            self.stop_socket(socket)

    def test_detection_view_keeps_immediate_annotated_frames(self):
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json(self.config("detection"))
            started = socket.receive_json()
            self.assertIsNone(started["cameras"][0]["media_url"])
            frame = socket.receive_json()
            self.assertEqual(frame["type"], "frame")
            self.assertEqual(frame["display_mode"], "detection")
            self.assertTrue(frame["image"])
            self.assertIsNone(frame["playback_seconds"])
            self.assertEqual(len(server.app.state.connection.session.media.references), 0)
            self.stop_socket(socket)

    def test_stop_restart_invalidates_old_tokens_and_reuses_mock_model(self):
        with TestClient(server.app) as client:
            urls = []
            for _ in range(2):
                with client.websocket_connect("/ws") as socket:
                    socket.receive_json()
                    socket.send_json(self.config())
                    urls.append(socket.receive_json()["cameras"][0]["media_url"])
                    self.stop_socket(socket)
            self.assertNotEqual(*urls)
            self.assertTrue(all(client.get(url).status_code == 404 for url in urls))
            self.load.assert_called_once()
            self.assertTrue(all(source.closed for source in FakeReplay.instances))

    def test_invalid_video_gets_no_media_reference(self):
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json(self.config(cameras=[{"id": "north", "kind": "video", "path": str(self.folder / "missing.mp4")}]))
            camera = socket.receive_json()["cameras"][0]
            self.assertTrue(camera["error"])
            self.assertIsNone(camera["media_url"])
            self.assertFalse(server.app.state.connection.session.media.references)
            self.stop_socket(socket)

    def test_uploaded_live_view_keeps_server_detection_and_cleans_upload(self):
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            uploaded = client.post("/api/uploads", files={"file": ("local.mp4", self.content)}).json()
            token = uploaded["upload_id"]
            path = server.app.state.uploads.get(token)
            socket.send_json(self.config(cameras=[{"id": "north", "kind": "video", "upload_id": token}]))
            started = socket.receive_json()
            # Local browser files use object URLs; upload IDs are not media capabilities.
            self.assertIsNone(started["cameras"][0]["media_url"])
            self.assertEqual(client.get("/api/media/" + token).status_code, 404)
            socket.send_json({"action": "play"})
            self.assertEqual(socket.receive_json()["type"], "playing")
            frame = socket.receive_json()
            self.assertIsNone(frame["image"])
            self.assertEqual(frame["animals"], {"cow": .9})
            self.stop_socket(socket)
            self.assertFalse(path.exists())

    def test_all_cameras_share_epoch_and_disconnect_removes_tokens(self):
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.receive_json()
                socket.send_json(self.config(cameras=[{"id": id, "kind": "video", "path": str(self.video)}
                                                      for id in ("north", "creek", "feeder", "gate")]))
                urls = [c["media_url"] for c in socket.receive_json()["cameras"]]
                socket.send_json({"action": "play"})
                socket.receive_json()
                self.assertEqual(len({source.started for source in FakeReplay.instances}), 1)
                self.assertEqual(len(server.app.state.connection.session.media.references), 4)
                self.assertTrue(all(source.loop for source in FakeReplay.instances))
            self.assertTrue(all(client.get(url).status_code == 404 for url in urls))
            self.assertTrue(all(source.closed for source in FakeReplay.instances))

    @patch("wildlife.sources.time.monotonic")
    @patch("wildlife.sources.cv2.VideoCapture")
    def test_replay_clock_includes_processing_delay_and_loops(self, capture, clock):
        capture.return_value.isOpened.return_value = True
        capture.return_value.get.side_effect = [10, 30]
        capture.return_value.read.return_value = (True, np.zeros((16, 16, 3), dtype=np.uint8))
        clock.return_value = 100
        replay = VideoReplay("fixture.mp4", loop=True)
        replay.start_at(101)
        clock.return_value = 102
        _, footage = replay.read()
        self.assertEqual(footage, 1)
        clock.return_value = 102.8
        self.assertAlmostEqual(replay.playback_seconds(), 1.8)
        clock.return_value = 104.5
        _, footage = replay.read()
        self.assertEqual(footage, .5)
        self.assertEqual(replay.event_seconds(footage), 3.5)
        replay.close()


if __name__ == "__main__":
    unittest.main()
