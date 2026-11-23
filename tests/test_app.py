"""HTTP/WebSocket behavior with fake inference; no physical browser stream needed."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
import app as server


def start_message(**overrides):
    return {"action": "start", "mode": "debug", "model": "yolo11n.pt",
            "rate": 10, "repeats": 3,
            "cameras": [{"id": "debug", "kind": "webcam"}], **overrides}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        self.output_patch = patch.object(server, "OUTPUT", self.output)
        self.output_patch.start()
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.jpeg = cv2.imencode(".jpg", self.frame)[1].tobytes()

    def tearDown(self):
        self.output_patch.stop()
        self.temporary.cleanup()

    def test_startup_main_page_and_static_files(self):
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/api/health").json()["status"], "ready")
            self.assertIn("Texas Wildlife Camera", client.get("/").text)
            for path in ("/static/app.js", "/static/demo-playback.js", "/static/styles.css"):
                self.assertEqual(client.get(path).status_code, 200)
        self.assertIsNone(server.app.state.connection)

    def test_upload_validation_delete_and_shutdown_cleanup(self):
        with TestClient(server.app) as client:
            for filename, content in (("bad.txt", b"bad"), ("empty.mp4", b"")):
                self.assertEqual(client.post("/api/uploads", files={"file": (filename, content)}).status_code, 400)
            response = client.post("/api/uploads", files={"file": ("clip.mp4", b"fixture")})
            token = response.json()["upload_id"]
            path = server.app.state.uploads.get(token)
            self.assertTrue(path.is_file())
            self.assertEqual(client.delete("/api/uploads/" + token).status_code, 200)
            self.assertFalse(path.exists())
            token = client.post("/api/uploads", files={"file": ("clip.mp4", b"fixture")}).json()["upload_id"]
            orphan = server.app.state.uploads.get(token)
        self.assertFalse(orphan.exists())

    @patch("wildlife.runtime.load_model")
    @patch("wildlife.runtime.detect_frame")
    def test_uploaded_replay_stop_releases_temporary_copy(self, detect, load):
        video = self.output / "sample.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
        self.assertTrue(writer.isOpened())
        for _ in range(50):
            writer.write(self.frame)
        writer.release()
        detect.return_value = (self.frame, {}, [])
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            response = client.post("/api/uploads", files={"file": ("clip.avi", video.read_bytes())})
            token = response.json()["upload_id"]
            temporary = server.app.state.uploads.get(token)
            socket.send_json(start_message(cameras=[{"id": "debug", "kind": "video", "upload_id": token}]))
            self.assertEqual(socket.receive_json()["type"], "started")
            message = socket.receive_json()
            self.assertEqual(message["type"], "frame")
            self.assertEqual(message["source"], "REPLAY Debug")
            self.assertIsNotNone(message["footage_time"])
            self.assertEqual(client.delete("/api/uploads/" + token).status_code, 409)
            socket.send_json({"action": "stop"})
            while socket.receive_json()["type"] != "stopped":
                pass
            self.assertFalse(temporary.exists())
            self.assertTrue(video.exists())  # Only the temporary copy is removed.

    def test_local_origin_and_snapshot_path_checks(self):
        with TestClient(server.app) as client:
            response = client.post("/api/uploads", files={"file": ("clip.mp4", b"fixture")},
                                   headers={"origin": "https://unrelated.example"})
            self.assertEqual(response.status_code, 403)
            self.assertEqual(client.get("/snapshots/missing.jpg").status_code, 404)
            self.assertEqual(client.get("/snapshots/settings.json").status_code, 404)

    def test_missing_model_path_returns_helpful_error(self):
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json(start_message(model=str(self.output / "missing.pt")))
            message = socket.receive_json()
            self.assertEqual(message["type"], "error")
            self.assertIn("Cannot load model", message["message"])
            self.assertIsNone(server.app.state.connection.session)

    def test_invalid_messages_are_safe_and_socket_survives(self):
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            self.assertEqual(socket.receive_json()["type"], "connected")
            for text in ("invalid json", "[]", '{"action":"nonsense"}', '{"action":"start","rate":999}'):
                socket.send_text(text)
                result = socket.receive_json()
                self.assertEqual(result["type"], "error")
                self.assertNotIn("Traceback", result["message"])
            socket.send_bytes(self.jpeg)
            self.assertEqual(socket.receive_json()["type"], "error")
            socket.send_json({"action": "stop"})
            self.assertEqual(socket.receive_json()["type"], "stopped")

    @patch("wildlife.runtime.load_model")
    @patch("wildlife.runtime.detect_frame")
    def test_start_confirm_stop_and_reuse_model(self, detect, load):
        detect.return_value = (self.frame, {"cow": .9},
                               [{"label": "cow", "confidence": .9, "xyxy": [1, 1, 20, 20]}])
        with TestClient(server.app) as client, client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json(start_message())
            self.assertEqual(socket.receive_json()["type"], "started")
            for index in range(3):
                socket.send_bytes(self.jpeg)
                message = socket.receive_json()
                self.assertEqual(message["type"], "frame")
                self.assertEqual(message["detections"][0]["label"], "cow")
                self.assertIsNone(message["footage_time"])
                self.assertTrue(message["image"])
                self.assertEqual(len(message["events"]), int(index == 2))
            event = message["events"][0]
            self.assertEqual(event["title"], "Cow spotted")
            self.assertEqual(client.get(event["snapshot_url"]).status_code, 200)
            socket.send_json({"action": "stop"})
            self.assertEqual(socket.receive_json()["type"], "stopped")
            self.assertEqual(socket.receive()["type"], "websocket.close")
            with client.websocket_connect("/ws") as restarted:
                restarted.receive_json()
                restarted.send_json(start_message())
                self.assertEqual(restarted.receive_json()["type"], "started")
                restarted.send_json({"action": "stop"})
                restarted.receive_json()
            load.assert_called_once()
            self.assertTrue((self.output / "events.csv").is_file())


if __name__ == "__main__":
    unittest.main()
