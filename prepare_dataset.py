"""Copy downloaded YOLO data into a balanced detection dataset; never train a model.

Run with the project Python. The three downloads are read-only inputs. The
output is always data/wildlife_dataset, and replacing it needs --rebuild.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont
import yaml

from wildlife.dataset import BOUNDARY_TOLERANCE, CLASS_NAMES, IMAGE_SUFFIXES, check_dataset, validate_box

ROOT = Path(__file__).resolve().parent
SEED = 42
SPLITS = ("train", "val", "test")
TARGET_COUNTS = {0: {"train": 231, "val": 50, "test": 50},
                 1: {"train": 231, "val": 50, "test": 49},
                 2: {"train": 231, "val": 50, "test": 50}}
SOURCES = {"deer": {"final_id": 0, "names": ["deer"], "license": "CC BY 4.0"},
           "boar": {"final_id": 1, "names": ["boar", "dog", "human"], "license": "Public Domain"},
           "cow": {"final_id": 2, "names": ["cow"], "license": "CC BY 4.0"}}
VIDEO_FRAME_CLASSES = {"boar": 1, "cow": 2, "deer": 0}
VIDEO_FRAME_SOURCE_NAMES = {0: "boar", 1: "cow", 2: "deer"}
MARKER = ".prepared_by_wildlife.json"


def windows_path(path: Path) -> Path:
    """Extended absolute paths let Python read long downloaded Windows filenames."""
    absolute = str(path.absolute())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
    return Path(absolute)


def readable_path(path: Path) -> str:
    """Remove Windows's internal path prefix for a useful provenance record."""
    name = str(path)
    if name.startswith("\\\\?\\UNC\\"):
        name = "\\\\" + name[8:]
    elif name.startswith("\\\\?\\"):
        name = name[4:]
    try:
        return Path(name).relative_to(ROOT).as_posix()
    except ValueError:
        return Path(name).as_posix()


def polygon_to_box(coordinates):
    """Enclose polygon vertices in a rectangular normalized xywh detection box."""
    if len(coordinates) < 6 or len(coordinates) % 2:
        raise ValueError("A polygon needs at least three x/y coordinate pairs.")
    values = list(map(float, coordinates))
    if not all(math.isfinite(value) and -BOUNDARY_TOLERANCE <= value <= 1 + BOUNDARY_TOLERANCE for value in values):
        raise ValueError("Polygon vertices must be normalized to 0–1.")
    xs, ys = values[::2], values[1::2]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    return validate_box(((left + right) / 2, (top + bottom) / 2,
                         right - left, bottom - top))


def convert_annotations(text: str, source: str):
    """Remap class 0; filter boar's dog/human rows without discarding mixed images.

    Invalid target annotations reject the whole image later. Keeping an image
    with a silently missing target box would teach the detector a false negative.
    """
    boxes, formats, ignored = [], Counter(), Counter()
    final_id = SOURCES[source]["final_id"]
    names = SOURCES[source]["names"]
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if fields[0] not in {str(index) for index in range(len(names))}:
            raise ValueError(f"Unknown source class on line {line_number}.")
        original_id = int(fields[0])
        if original_id != 0:
            ignored[names[original_id]] += 1
            continue
        if len(fields) == 5:
            coordinates = validate_box(fields[1:])
            formats["existing_boxes"] += 1
            if coordinates != tuple(map(float, fields[1:])):
                formats["clamped_boxes"] += 1
        else:
            coordinates = polygon_to_box(fields[1:])
            formats["converted_polygons"] += 1
        boxes.append((final_id, *coordinates))
    return boxes, dict(formats), dict(ignored)


@dataclass
class SourceImage:
    """Keep small annotation records in RAM, not decoded images."""
    image: Path
    source: str
    original_split: str
    final_id: int
    boxes: list
    sha256: str
    family: str
    can_promote: bool
    formats: dict
    ignored: dict
    source_url: str


def skipped_record(record: SourceImage, reason: str):
    return {"original_filename": readable_path(record.image), "source_dataset": record.source,
            "original_split": record.original_split, "reason": reason}


def collect_sources(data_dir: Path):
    """Read metadata, labels and hashes; record exclusions instead of hiding them."""
    records, skipped, inventory = [], [], {}
    for source, specification in SOURCES.items():
        folder = windows_path(data_dir / source)
        metadata = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))
        names = metadata["names"]
        if isinstance(names, dict):
            names = [names[index] for index in sorted(names)]
        if names != specification["names"]:
            raise ValueError(f"Unexpected class order in {source}/data.yaml.")
        if metadata.get("roboflow", {}).get("license") != specification["license"]:
            raise ValueError(f"Check the source license for {source}; metadata does not match.")
        readme = (folder / "README.roboflow.txt").read_text(encoding="utf-8")
        # Only known unaugmented training exports may be moved into holdouts.
        can_promote = "No image augmentation techniques were applied." in readme
        inventory[source] = {}
        for split in ("train", "valid", "test"):
            images = sorted(p for p in (folder / split / "images").iterdir()
                            if p.suffix.lower() in IMAGE_SUFFIXES)
            inventory[source][split] = len(images)
            for image in images:
                info = {"original_filename": readable_path(image), "source_dataset": source,
                        "original_split": split}
                label = folder / split / "labels" / (image.stem + ".txt")
                if not label.is_file():
                    skipped.append({**info, "reason": "missing_label"})
                    continue
                try:
                    boxes, formats, ignored = convert_annotations(label.read_text(encoding="utf-8"), source)
                except (ValueError, OSError) as error:
                    skipped.append({**info, "reason": "invalid_annotation", "detail": str(error)})
                    continue
                if not boxes:
                    skipped.append({**info, "reason": "no_target_annotations", "ignored_annotations": ignored})
                    continue
                try:
                    with image.open("rb") as handle:
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    with Image.open(image) as picture:
                        picture.load()  # Decode one image to reject corrupt input before publication.
                except (OSError, ValueError) as error:
                    skipped.append({**info, "reason": "unreadable_image", "detail": str(error)})
                    continue
                records.append(SourceImage(image, source, split, specification["final_id"], boxes,
                                           digest, image.stem.split(".rf.")[0], can_promote,
                                           formats, ignored, metadata.get("roboflow", {}).get("url", "")))
    return records, skipped, inventory


def collect_video_frame_export(data_dir: Path, export_name: str, seen_hashes: set):
    """Read the optional auto-labeled video export and remap its class IDs.

    Roboflow exported 0=boar, 1=cow, 2=deer, while this project requires
    0=deer, 1=wild_boar, 2=cow. All accepted frames become training examples;
    neighboring video frames must not be used as independent holdout evidence.
    """
    folder = windows_path(data_dir / export_name)
    if not folder.is_dir():
        return [], [], {"train": 0, "valid": 0, "test": 0}
    metadata = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))
    if metadata.get("names") != ["boar", "cow", "deer"]:
        raise ValueError("Unexpected class order in video_frames/data.yaml.")
    records, skipped = [], []
    inventory = {}
    for split in ("train", "valid", "test"):
        image_folder = folder / split / "images"
        images = sorted(p for p in image_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        inventory[split] = len(images)
        for image in images:
            info = {"original_filename": readable_path(image), "source_dataset": export_name,
                    "original_split": split}
            label = folder / split / "labels" / (image.stem + ".txt")
            if not label.is_file():
                skipped.append({**info, "reason": "missing_label"})
                continue
            expected_name = next((name for name in VIDEO_FRAME_CLASSES if image.name.startswith(name + "_")), None)
            boxes, source_names = [], set()
            try:
                for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    fields = line.split()
                    if not fields:
                        continue
                    source_id = int(fields[0])
                    if source_id not in VIDEO_FRAME_SOURCE_NAMES or len(fields) != 5:
                        raise ValueError(f"Invalid video-frame annotation on line {line_number}.")
                    source_name = VIDEO_FRAME_SOURCE_NAMES[source_id]
                    source_names.add(source_name)
                    if expected_name is not None and source_name != expected_name:
                        raise ValueError(f"{image.name} is labeled {source_name}, expected {expected_name}.")
                    boxes.append((VIDEO_FRAME_CLASSES[source_name], *validate_box(fields[1:])))
                if not boxes:
                    raise ValueError("Video frame has no target annotations.")
                if len(source_names) != 1:
                    raise ValueError("Each source video frame must contain one animal class.")
                with image.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                with Image.open(image) as picture:
                    picture.load()
            except (ValueError, OSError) as error:
                skipped.append({**info, "reason": "invalid_annotation", "detail": str(error)})
                continue
            if digest in seen_hashes:
                skipped.append({**info, "reason": "duplicate_hash"})
                continue
            seen_hashes.add(digest)
            expected_name = source_names.pop()
            final_id = VIDEO_FRAME_CLASSES[expected_name]
            records.append(SourceImage(image, "video_frames", split, final_id, boxes, digest,
                                       image.stem.split(".rf.")[0], False,
                                       {"existing_boxes": len(boxes)}, {},
                                       metadata.get("roboflow", {}).get("url", "")))
    return records, skipped, inventory


def collect_video_frames(data_dir: Path):
    """Combine versioned exports while keeping the first copy of each image."""
    records, skipped, inventory, seen_hashes = [], [], {}, set()
    for export_name in ("video_frames", "video_frames_v2"):
        new_records, new_skipped, counts = collect_video_frame_export(
            data_dir, export_name, seen_hashes)
        records.extend(new_records)
        skipped.extend(new_skipped)
        inventory[export_name] = counts
    return records, skipped, inventory


def remove_duplicates(records, seed=SEED):
    """Prefer source holdouts; keep one member of each known augmentation family.

    Hashes catch byte-identical images. Roboflow's pre-.rf. stem groups exported
    variants conservatively; it cannot identify every related video frame.
    """
    priority = {"test": 0, "valid": 1, "train": 2}
    ordered = sorted(records, key=lambda r: (priority[r.original_split], r.source, str(r.image)))
    unique, skipped, seen = [], [], set()
    for record in ordered:
        if record.sha256 in seen:
            skipped.append(skipped_record(record, "duplicate_hash"))
        else:
            seen.add(record.sha256)
            unique.append(record)
    families = defaultdict(list)
    for record in unique:
        families[(record.source, record.family)].append(record)
    selected, generator = [], random.Random(seed)
    for key in sorted(families):
        group = families[key]
        preferred = min(priority[r.original_split] for r in group)
        choices = [r for r in group if priority[r.original_split] == preferred]
        chosen = generator.choice(choices) if len(choices) > 1 else choices[0]
        selected.append(chosen)
        skipped.extend(skipped_record(r, "related_variant") for r in group if r is not chosen)
    return selected, skipped


def assign_splits(records, targets=TARGET_COUNTS, seed=SEED):
    """Keep original holdouts; fill deficits from unaugmented train with seed 42."""
    selected, skipped = [], []
    generator = random.Random(seed)
    for class_id in CLASS_NAMES:
        pools = {split: sorted((r for r in records if r.final_id == class_id and
                               ("val" if r.original_split == "valid" else r.original_split) == split),
                              key=lambda r: str(r.image)) for split in SPLITS}
        for pool in pools.values():
            generator.shuffle(pool)
        # Fill validation and test first; holdout surplus never goes into training.
        for split in ("val", "test"):
            desired = targets[class_id][split]
            keep, surplus = pools[split][:desired], pools[split][desired:]
            skipped.extend(skipped_record(r, "target_surplus") for r in surplus)
            deficit = desired - len(keep)
            promoted = [r for r in pools["train"] if r.can_promote][:deficit]
            if len(promoted) != deficit:
                raise ValueError(f"Insufficient unaugmented {CLASS_NAMES[class_id]} images for {split}.")
            promoted_ids = {r.sha256 for r in promoted}
            pools["train"] = [r for r in pools["train"] if r.sha256 not in promoted_ids]
            selected.extend((r, split) for r in keep + promoted)
        desired = targets[class_id]["train"]
        if len(pools["train"]) < desired:
            raise ValueError(f"Need {desired} unique {CLASS_NAMES[class_id]} train images; have {len(pools['train'])}.")
        selected.extend((r, "train") for r in pools["train"][:desired])
        skipped.extend(skipped_record(r, "target_surplus") for r in pools["train"][desired:])
    return selected, skipped


def short_filename(class_id: int, split: str, index: int) -> str:
    return f"{CLASS_NAMES[class_id]}_{split}_{index:04d}.jpg"


def write_examples(folder: Path, selected):
    """Copy JPEG bytes when possible; write remapped rectangular detection labels."""
    counters, rows = Counter(), []
    for record, split in selected:
        counters[(record.final_id, split)] += 1
        filename = short_filename(record.final_id, split, counters[(record.final_id, split)])
        image_path = folder / "images" / split / filename
        label_path = folder / "labels" / split / (Path(filename).stem + ".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if record.image.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copyfile(record.image, image_path)
        else:
            with Image.open(record.image) as picture:
                picture.convert("RGB").save(image_path, quality=95)
        lines = [" ".join([str(box[0]), *(format(value, ".12g") for value in box[1:])])
                 for box in record.boxes]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        source_name = (CLASS_NAMES[record.final_id] if record.source == "video_frames"
                       else SOURCES[record.source]["names"][0])
        source_license = ("Private Roboflow project" if record.source == "video_frames"
                          else SOURCES[record.source]["license"])
        rows.append({"generated_filename": filename, "generated_image": image_path.relative_to(folder).as_posix(),
                     "original_filename": readable_path(record.image), "source_dataset": record.source,
                     "original_split": record.original_split, "final_split": split,
                     "original_class": source_name, "original_class_id": "varies" if record.source == "video_frames" else 0,
                     "final_class": CLASS_NAMES[record.final_id], "final_class_id": record.final_id,
                     "source_license": source_license, "source_url": record.source_url,
                     "image_sha256": record.sha256})
    with (folder / "source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def create_contact_sheets(folder: Path, rows, seed=SEED):
    """Draw actual generated ground-truth boxes: four random images per class/split."""
    generator = random.Random(seed)
    output = folder / "contact_sheets"
    output.mkdir()
    font = ImageFont.load_default(size=14)
    for split in SPLITS:
        examples = []
        for class_id in CLASS_NAMES:
            pool = [row for row in rows if row["final_split"] == split and row["final_class_id"] == class_id]
            examples.extend(generator.sample(pool, min(4, len(pool))))
        sheet = Image.new("RGB", (960, 40 + 250 * math.ceil(len(examples) / 3)), "#f2f4ef")
        drawing = ImageDraw.Draw(sheet)
        drawing.text((12, 10), f"{split.upper()} · generated ground-truth annotations · seed {seed}", fill="black", font=font)
        for index, row in enumerate(examples):
            with Image.open(folder / row["generated_image"]) as picture:
                thumbnail = picture.convert("RGB")
            thumbnail.thumbnail((304, 208))
            x, y = (index % 3) * 320 + 8, (index // 3) * 250 + 40
            sheet.paste(thumbnail, (x, y))
            label = folder / "labels" / split / (Path(row["generated_filename"]).stem + ".txt")
            for line in label.read_text(encoding="utf-8").splitlines():
                class_id, cx, cy, width, height = map(float, line.split())
                left, top = x + (cx - width/2)*thumbnail.width, y + (cy-height/2)*thumbnail.height
                right, bottom = left + width*thumbnail.width, top + height*thumbnail.height
                drawing.rectangle((left, top, right, bottom), outline="#00ff80", width=2)
                text = CLASS_NAMES[int(class_id)]
                drawing.rectangle(drawing.textbbox((left, top), text, font=font), fill="#17382c")
                drawing.text((left, top), text, fill="white", font=font)
            drawing.text((x, y + 214), row["generated_filename"], fill="black", font=font)
        sheet.save(output / f"{split}.jpg", quality=90)


def validate_output_path(folder: Path, data_dir: Path):
    """Never follow output junctions/symlinks or replace a source download."""
    if folder.is_symlink() or folder.is_junction() or folder.resolve().parent != data_dir.resolve():
        raise ValueError("Generated output must be a real folder directly inside the selected data directory.")
    if folder.name != "wildlife_dataset" and not folder.name.startswith(".wildlife_dataset_"):
        raise ValueError("Refusing to remove a source or unrelated folder.")


def remove_generated(folder: Path, data_dir: Path):
    """Delete only a checked, script-owned generated folder, including on Windows."""
    validate_output_path(folder, data_dir)
    marker = folder / MARKER
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("tool") != "prepare_dataset.py":
        raise ValueError("Refusing to replace a folder without this script's ownership marker.")
    shutil.rmtree(folder)


def prepare_dataset(data_dir=ROOT / "data", rebuild=False, targets=TARGET_COUNTS, seed=SEED):
    """Build/check a staging copy before publishing; a failed build preserves old output."""
    data_dir = Path(data_dir).resolve()
    output = data_dir / "wildlife_dataset"
    validate_output_path(output, data_dir)
    if output.exists():
        if not rebuild:
            raise ValueError(f"{output} exists. Use --rebuild to replace this generated dataset.")
        marker = output / MARKER
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("tool") != "prepare_dataset.py":
            raise ValueError("Existing output is not marked as script-generated; refusing to replace it.")
    records, skipped, inventory = collect_sources(data_dir)
    unique, excluded = remove_duplicates(records, seed)
    skipped.extend(excluded)
    selected, excluded = assign_splits(unique, targets, seed)
    skipped.extend(excluded)
    video_records, video_skipped, video_inventory = collect_video_frames(data_dir)
    skipped.extend(video_skipped)
    selected.extend((record, "train") for record in video_records)
    inventory["video_frames"] = video_inventory
    staging = data_dir / (".wildlife_dataset_build_" + uuid4().hex)
    staging.mkdir()
    (staging / MARKER).write_text(json.dumps({"tool": "prepare_dataset.py", "seed": seed}), encoding="utf-8")
    try:
        rows = write_examples(staging, selected)
        configuration = {"path": str(staging), "train": "images/train", "val": "images/val",
                         "test": "images/test", "names": CLASS_NAMES}
        config_path = staging / "dataset.yaml"
        config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
        # The same checker used by training validates labels and cross-split hashes.
        check_dataset(config_path, SPLITS, require_all_classes=True)
        create_contact_sheets(staging, rows, seed)
        for source in SOURCES:
            provenance = staging / "source_licenses" / source
            provenance.mkdir(parents=True)
            for filename in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
                shutil.copyfile(windows_path(data_dir / source / filename), provenance / filename)
        counts = {name: {split: sum(r["final_class_id"] == class_id and r["final_split"] == split for r in rows)
                         for split in SPLITS} for class_id, name in CLASS_NAMES.items()}
        formats, ignored = Counter(), Counter()
        for record, _ in selected:
            formats.update(record.formats)
            ignored.update(record.ignored)
        summary = {"seed": seed, "classes": CLASS_NAMES, "counts": counts, "total_images": len(rows),
                   "source_counts": inventory, "annotation_counts": dict(formats),
                   "excluded_annotations_in_selected_images": dict(ignored),
                   "source_split_transitions": dict(Counter(f"{r.source}:{r.original_split}->{split}" for r, split in selected)),
                   "skipped_images": len(skipped), "skipped_by_reason": dict(Counter(r["reason"] for r in skipped)),
                   "skipped_by_source": dict(Counter(r["source_dataset"] for r in skipped)), "skipped": skipped,
                   "grouping_note": "Exact hashes and pre-.rf. filename families are separated. Related visits/video frames still require human review."}
        (staging / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        configuration["path"] = str(output)
        config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
        if output.exists():
            backup = data_dir / (".wildlife_dataset_backup_" + uuid4().hex)
            validate_output_path(backup, data_dir)
            output.rename(backup)
            try:
                staging.rename(output)
            except OSError:
                backup.rename(output)
                raise
            remove_generated(backup, data_dir)
        else:
            staging.rename(output)
        return summary
    finally:
        if staging.exists():
            remove_generated(staging, data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Safely replace an existing script-generated dataset.")
    args = parser.parse_args()
    try:
        summary = prepare_dataset(rebuild=args.rebuild)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Preparation stopped: {error}\n")
    print(json.dumps({key: summary[key] for key in ("counts", "total_images", "skipped_images", "skipped_by_reason")}, indent=2))
    print("Saved data/wildlife_dataset, manifest, summary and annotated contact sheets. No training was run.")


if __name__ == "__main__":
    main()
