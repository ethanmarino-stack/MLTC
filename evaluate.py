"""Evaluate once on untouched test data, exporting per-class detection metrics."""
import argparse
import csv
from pathlib import Path

from wildlife.dataset import check_dataset
from wildlife.detection import choose_device, load_model


def main():
    """Keep test data out of tuning; validation data is for development decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/wildlife_dataset/dataset.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = check_dataset(args.data, ("test",), require_all_classes=True)
    model = load_model(args.model)
    if model.names != config["names"]:
        raise ValueError("Model class names must match dataset.yaml. Choose your trained best.pt.")
    # Low confidence retains candidates for the precision-recall curve, unlike
    # the app threshold. mAP evaluates both class prediction and box overlap.
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz,
                        batch=args.batch, workers=0, device=args.device or choose_device(),
                        conf=0.001, plots=True, project="runs/test", name="wildlife")
    rows = []
    indices = list(metrics.box.ap_class_index)
    for class_id, name in model.names.items():
        if class_id in indices:
            index = indices.index(class_id)
            precision, recall, map50, map95 = metrics.box.class_result(index)
            rows.append(dict(class_id=class_id, label=name, precision=precision,
                             recall=recall, mAP50=map50, mAP50_95=map95))
        else:
            # Never substitute aggregate mAP for a class without valid metrics.
            rows.append(dict(class_id=class_id, label=name, precision="N/A",
                             recall="N/A", mAP50="N/A", mAP50_95="N/A"))
    output = Path(metrics.save_dir) / "per_class_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Test mAP50={metrics.box.map50:.4f}; mAP50-95={metrics.box.map:.4f}")
    print(f"Per-class precision, recall and mAP: {output}")


if __name__ == "__main__":
    main()
