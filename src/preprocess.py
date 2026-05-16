
from pathlib import Path
import shutil
import random

def convert_visdrone_to_human_car(src_root, dst_root, train_limit=500, val_limit=120):
    """
    Convert VisDrone 10-class labels into 2 classes:
    0 = human  (original pedestrian + people)
    1 = car    (original car)
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    for split in ["train", "val"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def process_split(src_split_name, dst_split_name, max_images):
        src_img_dir = src_root / src_split_name / "images"
        src_lbl_dir = src_root / src_split_name / "labels"

        dst_img_dir = dst_root / "images" / dst_split_name
        dst_lbl_dir = dst_root / "labels" / dst_split_name

        label_files = list(src_lbl_dir.glob("*.txt"))
        random.seed(42)
        random.shuffle(label_files)

        copied = 0

        for label_path in label_files:
            if copied >= max_images:
                break

            img_path = src_img_dir / f"{label_path.stem}.jpg"
            if not img_path.exists():
                continue

            new_lines = []

            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                old_cls = int(float(parts[0]))

                if old_cls in [0, 1]:
                    new_cls = 0
                elif old_cls == 3:
                    new_cls = 1
                else:
                    continue

                new_lines.append(" ".join([str(new_cls)] + parts[1:5]))

            if not new_lines:
                continue

            shutil.copy2(img_path, dst_img_dir / img_path.name)

            with open(dst_lbl_dir / f"{label_path.stem}.txt", "w") as f:
                f.write("\\n".join(new_lines))

            copied += 1

        print(f"{dst_split_name} copied: {copied}")

    process_split("VisDrone2019-DET-train", "train", train_limit)
    process_split("VisDrone2019-DET-val", "val", val_limit)
