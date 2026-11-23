"""Preparation checks using tiny software fixtures, never real training examples."""
import csv
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image
import yaml

from prepare_dataset import (SOURCES, SourceImage, assign_splits, convert_annotations,
                             polygon_to_box, prepare_dataset, remove_duplicates, short_filename)
from wildlife.dataset import CLASS_NAMES, check_dataset, validate_box
from wildlife.detection import ANIMAL_NAMES


class AnnotationTests(unittest.TestCase):
    def test_polygon_uses_all_vertex_extrema(self):
        box = polygon_to_box([.2, .3, .8, .3, .7, .7])
        for actual, expected in zip(box, (.5, .5, .6, .4)):
            self.assertAlmostEqual(actual, expected)

    def test_remapping_filters_only_non_boar_annotations(self):
        text = "0 0.5 0.5 0.2 0.4\n1 0.5 0.5 0.2 0.4\n2 0.5 0.5 0.2 0.4\n"
        boxes, formats, ignored = convert_annotations(text, "boar")
        self.assertEqual(boxes, [(1, .5, .5, .2, .4)])
        self.assertEqual(formats, {"existing_boxes": 1})
        self.assertEqual(ignored, {"dog": 1, "human": 1})
        self.assertEqual(convert_annotations("1 0.5 0.5 0.2 0.4", "boar")[0], [])
        for source, final_id in (("deer", 0), ("cow", 2)):
            self.assertEqual(convert_annotations(text.splitlines()[0], source)[0][0][0], final_id)
        self.assertIn("cow", ANIMAL_NAMES)  # Preserve pretrained cow alerts.
        self.assertIn("wild_boar", ANIMAL_NAMES)
        self.assertNotIn("cattle", ANIMAL_NAMES)

    def test_mixed_boxes_and_polygons_all_become_five_values(self):
        text = "0 0.5 0.5 0.2 0.4\n0 0.2 0.3 0.8 0.3 0.7 0.7"
        boxes, formats, _ = convert_annotations(text, "cow")
        self.assertTrue(all(len(box) == 5 and box[0] == 2 for box in boxes))
        self.assertEqual(formats, {"existing_boxes": 1, "converted_polygons": 1})

    def test_box_validity_and_tiny_boundary_clamping(self):
        self.assertEqual(validate_box((.5, .5, 1.0000002, 1)), (.5, .5, 1, 1))
        # Observed deer export error: the left edge is about -0.00000244.
        box = validate_box((.0935986328125, .5914388020833333, .1872021484375, .43034505208333335))
        self.assertAlmostEqual(box[0] - box[2]/2, 0)
        for values in ((.5, .5, 0, 1), (.5, .5, 1.01, 1), (-.2, .5, .1, .1),
                       (.5, .5, float("nan"), .1), (.5, .5, .1)):
            with self.assertRaises(ValueError):
                validate_box(values)
        for polygon in ([0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1.01, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0]):
            with self.assertRaises(ValueError):
                polygon_to_box(polygon)

    def test_generated_filenames_are_short(self):
        self.assertEqual(short_filename(0, "train", 1), "deer_train_0001.jpg")
        self.assertEqual(short_filename(1, "val", 1), "wild_boar_val_0001.jpg")
        self.assertEqual(short_filename(2, "test", 1), "cow_test_0001.jpg")
        self.assertLessEqual(len(short_filename(1, "train", 231)), 25)


def record(name, split, digest, family=None):
    return SourceImage(Path(name), "deer", split, 0, [(0, .5, .5, .2, .2)],
                       digest, family or name, True, {"existing_boxes": 1}, {}, "fixture")


class SelectionTests(unittest.TestCase):
    def test_duplicate_hash_keeps_source_test_and_cannot_cross_splits(self):
        train = record("train.jpg", "train", "same")
        test = record("test.jpg", "test", "same")
        selected, skipped = remove_duplicates([train, test])
        self.assertEqual(selected, [test])
        self.assertEqual(skipped[0]["reason"], "duplicate_hash")

    def test_filename_family_cannot_leak_augmented_variants(self):
        train = record("train.rf.a.jpg", "train", "first", "related")
        val = record("val.rf.b.jpg", "valid", "second", "related")
        selected, skipped = remove_duplicates([train, val])
        self.assertEqual(selected, [val])
        self.assertEqual(skipped[0]["reason"], "related_variant")

    def test_exact_requested_deer_split_and_original_holdout_preservation(self):
        records = [record(f"train_{i}.jpg", "train", str(i)) for i in range(321)]
        records += [record(f"val_{i}.jpg", "valid", f"val{i}") for i in range(7)]
        records += [record(f"test_{i}.jpg", "test", f"test{i}") for i in range(3)]
        targets = {0: {"train": 231, "val": 50, "test": 50},
                   1: {"train": 0, "val": 0, "test": 0}, 2: {"train": 0, "val": 0, "test": 0}}
        selected, skipped = assign_splits(records, targets)
        self.assertEqual(skipped, [])
        self.assertEqual({split: sum(final == split for _, final in selected)
                          for split in ("train", "val", "test")}, targets[0])
        self.assertEqual(sum(r.original_split == "train" and s == "val" for r, s in selected), 43)
        self.assertEqual(sum(r.original_split == "train" and s == "test" for r, s in selected), 47)
        self.assertTrue(all(s == "val" for r, s in selected if r.original_split == "valid"))
        self.assertTrue(all(s == "test" for r, s in selected if r.original_split == "test"))
        self.assertEqual(assign_splits(records, targets)[0], selected)


class PreparationTests(unittest.TestCase):
    def make_sources(self, root):
        generator = random.Random(123)
        for source, spec in SOURCES.items():
            folder = root / source
            folder.mkdir()
            metadata = {"names": spec["names"], "roboflow": {"license": spec["license"], "url": "software-fixture"}}
            (folder / "data.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
            (folder / "README.roboflow.txt").write_text("No image augmentation techniques were applied.", encoding="utf-8")
            (folder / "README.dataset.txt").write_text("Software fixtures only, not wildlife training data.", encoding="utf-8")
            for split, count in (("train", 5), ("valid", 1), ("test", 1)):
                images, labels = folder / split / "images", folder / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                for index in range(count):
                    # Random pixel fixtures make byte-distinct files without photos.
                    pixels = generator.randbytes(16 * 16 * 3)
                    Image.frombytes("RGB", (16, 16), pixels).save(images / f"fixture_{split}_{index}.jpg")
                    (labels / f"fixture_{split}_{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    def test_manifest_counts_labels_contact_sheets_and_safe_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_sources(root)
            originals = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                         for source in SOURCES for path in (root / source).rglob("*") if path.is_file()}
            targets = {class_id: {"train": 3, "val": 2, "test": 2} for class_id in CLASS_NAMES}
            summary = prepare_dataset(root, targets=targets)
            output = root / "wildlife_dataset"
            self.assertEqual(summary["total_images"], 21)
            self.assertEqual(summary["skipped_images"], 0)
            with (output / "source_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 21)
            required = {"generated_filename", "original_filename", "source_dataset", "original_split",
                        "final_split", "original_class", "final_class", "source_license", "image_sha256"}
            self.assertTrue(required <= set(rows[0]))
            for row in rows:
                image = output / row["generated_image"]
                self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), row["image_sha256"])
                self.assertLess(len(row["generated_filename"]), 30)
                label = output / "labels" / row["final_split"] / (image.stem + ".txt")
                fields = label.read_text().split()
                self.assertEqual(len(fields), 5)
                self.assertEqual(int(fields[0]), int(row["final_class_id"]))
                validate_box(fields[1:])
            check_dataset(output / "dataset.yaml", ("train", "val", "test"), require_all_classes=True)
            for split in ("train", "val", "test"):
                self.assertTrue((output / "contact_sheets" / (split + ".jpg")).is_file())
                self.assertTrue(all(summary["counts"][name][split] == targets[index][split]
                                    for index, name in CLASS_NAMES.items()))
            manifest = (output / "source_manifest.csv").read_bytes()
            with self.assertRaisesRegex(ValueError, "--rebuild"):
                prepare_dataset(root, targets=targets)
            prepare_dataset(root, rebuild=True, targets=targets)
            self.assertEqual((output / "source_manifest.csv").read_bytes(), manifest)
            self.assertEqual(originals, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals})
            # A failed rebuild must leave the previous complete dataset intact.
            (root / "deer" / "data.yaml").write_text("names: [wrong]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_dataset(root, rebuild=True, targets=targets)
            self.assertEqual((output / "source_manifest.csv").read_bytes(), manifest)

    def test_rebuild_refuses_unmarked_existing_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "wildlife_dataset"
            output.mkdir()
            original = output / "personal.txt"
            original.write_text("Keep me", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not marked"):
                prepare_dataset(root, rebuild=True)
            self.assertEqual(original.read_text(), "Keep me")


if __name__ == "__main__":
    unittest.main()
