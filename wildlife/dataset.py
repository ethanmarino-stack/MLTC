"""Check the simple folder-based dataset before training or evaluation."""
import hashlib
import math
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {0: "deer", 1: "wild_boar", 2: "cow"}
BOUNDARY_TOLERANCE = 1e-5


def validate_box(values, tolerance=BOUNDARY_TOLERANCE):
    """Validate normalized xywh; clip only tiny rounding errors at image edges.

    This also serves preparation, so training and conversion use the same rules.
    Larger errors are rejected rather than silently changing an annotation.
    """
    if len(values) != 4:
        raise ValueError("A detection box needs four coordinates.")
    x, y, width, height = map(float, values)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("Box coordinates must be finite.")
    if width <= 0 or height <= 0:
        raise ValueError("Box width and height must be positive.")
    left, right = x - width / 2, x + width / 2
    top, bottom = y - height / 2, y + height / 2
    if min(left, top) < -tolerance or max(right, bottom) > 1 + tolerance:
        raise ValueError("Box extends outside the normalized image.")
    if not (0 <= left <= right <= 1 and 0 <= top <= bottom <= 1):
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(1.0, right), min(1.0, bottom)
        x, y = (left + right) / 2, (top + bottom) / 2
        width, height = right - left, bottom - top
    if not (0 <= x <= 1 and 0 <= y <= 1 and width > 0 and height > 0):
        raise ValueError("Box has no positive area inside the image.")
    return x, y, width, height


def check_dataset(config_file, splits, require_all_classes=False):
    """Require explicit labels, valid normalized boxes, and distinct split folders.

    Empty label files mean a human verified a negative image. These checks cannot
    prove annotation quality or identify related scenes; group splitting is manual.
    """
    config = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    if config.get("names") != CLASS_NAMES:
        raise ValueError("Use class IDs 0=deer, 1=wild_boar, 2=cow in this project.")
    root = Path(config["path"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Set path in dataset.yaml to your existing ABSOLUTE dataset folder.")
    hashes = {}
    split_paths = set()
    for split in ("train", "val", "test"):
        folder_name = config.get(split)
        if not folder_name:
            if split in splits:
                raise ValueError(f"Missing {split} configuration.")
            continue
        folder = (root / folder_name).resolve()
        if folder in split_paths:
            raise ValueError("Each split needs a separate image folder.")
        split_paths.add(folder)
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if split in splits and not images:
            raise ValueError(f"No images found in {folder}")
        found_classes = set()
        for image in images:
            # Exact duplicates across splits leak information into evaluation.
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"Duplicate image across splits: {image}")
            hashes[digest] = split
            if split not in splits:
                continue  # Do not use test labels to tune training.
            relative = image.relative_to(root)
            if relative.parts[:2] != ("images", split):
                raise ValueError("Use images/train, images/val, images/test folder layout.")
            label = root / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing annotation: {label}. Use an empty file only for a verified negative.")
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5 or fields[0] not in {"0", "1", "2"}:
                    raise ValueError(f"Invalid class/box at {label}:{line_number}")
                try:
                    validate_box(fields[1:])
                except ValueError as error:
                    raise ValueError(f"Invalid box at {label}:{line_number}: {error}") from error
                found_classes.add(int(fields[0]))
        if split in splits and (require_all_classes or split in ("train", "val")):
            if found_classes != {0, 1, 2}:
                raise ValueError(f"{split} must contain annotated examples of all three classes.")
    return config
