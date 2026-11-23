"""Stage 3: supervised transfer learning, separate from the running interface."""
import argparse

from wildlife.detection import choose_device, load_model
from wildlife.dataset import check_dataset


def main():
    """Fine-tune pretrained YOLO11n on real, manually checked box annotations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/wildlife_dataset/dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default=None, help="0 for GPU or cpu")
    args = parser.parse_args()
    if min(args.imgsz, args.batch, args.epochs) < 1:
        parser.error("Image size, batch and epochs must be positive.")
    check_dataset(args.data, ("train", "val"))
    model = load_model(args.model)
    # workers=0 avoids Windows subprocess overhead; cache=False limits RAM usage.
    model.train(data=args.data, imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
                device=args.device if args.device is not None else choose_device(),
                workers=0, cache=False, amp=True, seed=42, deterministic=True,
                project="runs/train", name="wildlife")
    print("Use best.pt from the printed run folder in the app. Do not train while the app runs.")


if __name__ == "__main__":  # Windows requires this guard around training.
    main()
