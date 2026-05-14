"""
prepare_dataset.py

Converts the VisDrone detection dataset annotation format into YOLO format.

Project target classes:
    0 = human  -> VisDrone pedestrian + people
    1 = car    -> VisDrone car

Original VisDrone DET annotation format per line:
    bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion

Important:
    - score = 0 means ignored region/object, so we skip it.
    - Bounding boxes are clamped inside image boundaries.
    - Only pedestrian, people, and car are kept.
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# VisDrone original category IDs
# 0 = ignored region
# 1 = pedestrian
# 2 = people
# 3 = bicycle
# 4 = car
# 5 = van
# 6 = truck
# 7 = tricycle
# 8 = awning-tricycle
# 9 = bus
# 10 = motor
VISDRONE_CLASS_NAMES = {
    0: "ignored-region",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


# Convert VisDrone classes into our 2 assessment classes
# pedestrian + people -> human
# car -> car
CLASS_MAPPING = {
    1: 0,  # pedestrian -> human
    2: 0,  # people -> human
    4: 1,  # car -> car
}


PROJECT_CLASS_NAMES = {
    0: "human",
    1: "car",
}


SPLIT_FOLDERS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
}


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_image_file(images_dir: Path, stem: str) -> Path | None:
    """
    Finds an image by filename stem using common image extensions.
    Example:
        stem = "0000001_00000_d_0000001"
        returns matching .jpg/.png/etc.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_image_size(image_path: Path) -> tuple[int, int]:
    """
    Returns image width and height.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def clamp_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    """
    Clamps bounding box coordinates so they stay inside the image.
    Returns x1, y1, x2, y2.
    """
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)

    return x1, y1, x2, y2


def convert_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int):
    """
    Converts absolute pixel box coordinates into YOLO normalized format:
        x_center, y_center, width, height
    All values are between 0 and 1.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    x_center = x1 + box_w / 2.0
    y_center = y1 + box_h / 2.0

    return (
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h,
    )


def parse_annotation_file(annotation_path: Path, image_width: int, image_height: int):
    """
    Parses one VisDrone annotation file and returns YOLO-format labels.

    Returns:
        labels: list of strings, each line formatted as:
                class_id x_center y_center width height
        stats:  Counter containing conversion statistics
    """
    labels = []
    stats = Counter()

    if not annotation_path.exists():
        stats["missing_annotation_file"] += 1
        return labels, stats

    with open(annotation_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split(",")

        if len(parts) < 6:
            stats["invalid_rows"] += 1
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])
            score = int(float(parts[4]))
            category_id = int(float(parts[5]))
        except ValueError:
            stats["invalid_rows"] += 1
            continue

        # score = 0 means ignored object/region in VisDrone
        if score == 0:
            stats["ignored_score_zero"] += 1
            continue

        # Keep only pedestrian, people, and car
        if category_id not in CLASS_MAPPING:
            stats[f"skipped_{VISDRONE_CLASS_NAMES.get(category_id, 'unknown')}"] += 1
            continue

        x1, y1, x2, y2 = clamp_box(x, y, w, h, image_width, image_height)

        box_w = x2 - x1
        box_h = y2 - y1

        # Skip invalid or extremely tiny boxes
        if box_w <= 1 or box_h <= 1:
            stats["invalid_or_tiny_boxes"] += 1
            continue

        yolo_x, yolo_y, yolo_w, yolo_h = convert_to_yolo(
            x1, y1, x2, y2, image_width, image_height
        )

        new_class_id = CLASS_MAPPING[category_id]

        labels.append(
            f"{new_class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}"
        )

        stats[f"kept_{PROJECT_CLASS_NAMES[new_class_id]}"] += 1

    return labels, stats


def copy_image(source_path: Path, destination_path: Path):
    """
    Copies image to YOLO dataset folder.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def convert_split(raw_dir: Path, output_dir: Path, split: str):
    """
    Converts one split: train or val.
    """
    split_folder = SPLIT_FOLDERS[split]

    input_images_dir = raw_dir / split_folder / "images"
    input_annotations_dir = raw_dir / split_folder / "annotations"

    output_images_dir = output_dir / "images" / split
    output_labels_dir = output_dir / "labels" / split

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    if not input_images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {input_images_dir}")

    if not input_annotations_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {input_annotations_dir}")

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_images_dir.glob(f"*{ext}"))

    image_files = sorted(image_files)

    split_stats = Counter()
    class_distribution = Counter()

    print(f"\nConverting split: {split}")
    print(f"Images found: {len(image_files)}")

    for image_path in tqdm(image_files, desc=f"Processing {split}"):
        image_width, image_height = get_image_size(image_path)

        annotation_path = input_annotations_dir / f"{image_path.stem}.txt"

        labels, stats = parse_annotation_file(
            annotation_path=annotation_path,
            image_width=image_width,
            image_height=image_height,
        )

        split_stats.update(stats)

        for label in labels:
            class_id = int(label.split()[0])
            class_distribution[PROJECT_CLASS_NAMES[class_id]] += 1

        # Copy image
        output_image_path = output_images_dir / image_path.name
        copy_image(image_path, output_image_path)

        # Write YOLO label file
        output_label_path = output_labels_dir / f"{image_path.stem}.txt"
        with open(output_label_path, "w", encoding="utf-8") as label_file:
            label_file.write("\n".join(labels))

    summary = {
        "split": split,
        "total_images": len(image_files),
        "class_distribution": dict(class_distribution),
        "conversion_stats": dict(split_stats),
    }

    return summary


def write_dataset_yaml(output_dir: Path):
    """
    Creates YOLO dataset YAML file.
    """
    yaml_path = output_dir / "visdrone.yaml"

    output_path = str(output_dir.resolve()).replace("\\", "/")

    yaml_content = f"""# Auto-generated YOLO dataset config for Antlings assessment

path: {output_path}
train: images/train
val: images/val

nc: 2
names:
  0: human
  1: car
"""

    with open(yaml_path, "w", encoding="utf-8") as file:
        file.write(yaml_content)

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone dataset into YOLO format for human and car detection."
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Path to raw VisDrone dataset folder containing VisDrone2019-DET-train and VisDrone2019-DET-val.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/visdrone_yolo",
        help="Output directory for converted YOLO dataset.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Dataset splits to convert.",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for split in args.splits:
        summary = convert_split(raw_dir, output_dir, split)
        all_summaries[split] = summary

    yaml_path = write_dataset_yaml(output_dir)

    summary_path = output_dir / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(all_summaries, file, indent=4)

    print("\nDataset conversion complete.")
    print(f"YOLO dataset saved to: {output_dir}")
    print(f"YOLO YAML file saved to: {yaml_path}")
    print(f"Conversion summary saved to: {summary_path}")

    print("\nClass names:")
    for class_id, class_name in PROJECT_CLASS_NAMES.items():
        print(f"{class_id}: {class_name}")


if __name__ == "__main__":
    main()