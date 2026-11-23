"""Stage 1: predict an image or sample a saved video without a web interface."""
import argparse
from pathlib import Path

import cv2

from wildlife.detection import detect_frame, load_model


def main():
    """Write annotated outputs, holding only one decoded frame at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=3, help="Sampled output video FPS")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if not args.source.is_file() or not 0 < args.confidence <= 1 or args.fps <= 0:
        parser.error("Provide an existing file, confidence in (0,1], and positive FPS.")
    model = load_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(args.source))
    if image is not None:
        annotated, animals = detect_frame(model, image, args.confidence)
        output = args.output / f"{args.source.stem}_detected.jpg"
        if not cv2.imwrite(str(output), annotated):
            raise OSError("Cannot write image.")
        print(output, animals)
        return
    capture = cv2.VideoCapture(str(args.source))
    writer = None
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or source_fps <= 0:
            raise ValueError("Cannot open video or read its FPS.")
        output_fps = min(args.fps, source_fps)
        next_sample = 0.0
        frame_index = 0
        output = args.output / f"{args.source.stem}_REPLAY.avi"
        while True:
            # Grab skips unneeded frames without constructing a Python queue.
            if not capture.grab():
                break
            seconds = frame_index / source_fps
            frame_index += 1
            if seconds < next_sample:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                break
            next_sample += 1 / output_fps
            annotated, animals = detect_frame(model, frame, args.confidence)
            cv2.putText(annotated, f"REPLAY | footage {seconds:.2f} s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"),
                                         output_fps, (width, height))
                if not writer.isOpened():
                    raise OSError("Cannot create output video.")
            writer.write(annotated)
            print(f"REPLAY footage {seconds:.2f} s: {animals}")
        if writer is None:
            raise ValueError("No video frames decoded.")
        print(f"Saved {output}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
