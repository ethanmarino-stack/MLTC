"""Model loading and single-frame inference using the official library."""
from pathlib import Path
import os

import torch

# Keep library settings with project outputs rather than in a user-wide folder.
config_folder = Path(__file__).resolve().parents[1] / "outputs" / "ultralytics"
config_folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_folder))
from ultralytics import YOLO, settings

settings.update({"sync": False})  # Local project: disable optional analytics sync.

# COCO has cow, but no deer or wild boar. Never rename other predictions as deer.
ANIMAL_NAMES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "deer", "wild_boar"}


def choose_device():
    """Use the first NVIDIA GPU when PyTorch can access CUDA; otherwise use CPU."""
    return 0 if torch.cuda.is_available() else "cpu"


def load_model(model_file):
    """Load trusted local detection weights, or download official YOLO11n once."""
    path = Path(model_file)
    if model_file != "yolo11n.pt" and not path.is_file():
        raise ValueError("Choose yolo11n.pt or an existing local .pt file.")
    model = YOLO(model_file)  # Import the package; no copied model source code.
    if model.task != "detect":
        raise ValueError("This project needs an object-detection model.")
    return model


def detect_frame(model, frame, confidence=0.5, image_size=640, include_boxes=False):
    """Return an annotated BGR image and the strongest score for each animal class."""
    # Confidence filtering removes weak predictions before drawing or alerting.
    result = model.predict(frame, conf=confidence, imgsz=image_size,
                           device=choose_device(), verbose=False, max_det=30)[0]
    animals = {}
    detections = []
    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        score = float(box.conf.item())
        detections.append({"label": name, "confidence": score,
                           "xyxy": box.xyxy[0].tolist()})
        if name in ANIMAL_NAMES:
            animals[name] = max(score, animals.get(name, 0.0))
    # YOLO's plot includes bounding boxes, original labels and confidence scores.
    if include_boxes:
        return result.plot(), animals, detections
    return result.plot(), animals  # Preserve the CLI and course notebook API.
