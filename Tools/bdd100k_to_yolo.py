from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "train": 3,
    "motor": 4,
    "bike": 5,
    "traffic light": 6,
    "traffic sign": 7,
}


def yolo_line(label: dict, width: int, height: int) -> str | None:
    category = label.get("category")
    if category not in CLASS_MAP or "box2d" not in label:
        return None

    box = label["box2d"]
    x1 = max(0.0, float(box["x1"]))
    y1 = max(0.0, float(box["y1"]))
    x2 = min(float(width), float(box["x2"]))
    y2 = min(float(height), float(box["y2"]))
    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{CLASS_MAP[category]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def convert_split(labels_json: Path, images_dir: Path, labels_out: Path) -> int:
    labels_out.mkdir(parents=True, exist_ok=True)
    with labels_json.open("r", encoding="utf-8") as file:
        frames = json.load(file)

    converted = 0
    for frame in frames:
        image_name = frame["name"]
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        width, height = read_image_size(image_path)
        lines = [
            line
            for label in frame.get("labels", [])
            if (line := yolo_line(label, width, height)) is not None
        ]
        label_path = labels_out / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        converted += 1
    return converted


def write_dataset_yaml(output_dir: Path, train_images: Path, val_images: Path) -> None:
    names = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda item: item[1])]
    yaml_text = [
        f"path: {output_dir.as_posix()}",
        f"train: {train_images.resolve().as_posix()}",
        f"val: {val_images.resolve().as_posix()}",
        "names:",
    ]
    yaml_text.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output_dir / "bdd100k.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BDD100K detection labels to YOLO format.")
    parser.add_argument("--bdd-root", required=True, help="BDD100K root folder")
    parser.add_argument("--output", default="data/bdd100k_yolo", help="Output folder for the YOLO data yaml")
    args = parser.parse_args()

    bdd_root = Path(args.bdd_root)
    output_dir = Path(args.output)
    labels_root = bdd_root / "labels" / "100k"

    train_json = bdd_root / "labels" / "det_20" / "det_train.json"
    val_json = bdd_root / "labels" / "det_20" / "det_val.json"
    train_images = bdd_root / "images" / "100k" / "train"
    val_images = bdd_root / "images" / "100k" / "val"

    required = [train_json, val_json, train_images, val_images]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing BDD100K paths:\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    (labels_root / "train").mkdir(parents=True, exist_ok=True)
    (labels_root / "val").mkdir(parents=True, exist_ok=True)

    train_count = convert_split(train_json, train_images, labels_root / "train")
    val_count = convert_split(val_json, val_images, labels_root / "val")
    write_dataset_yaml(output_dir, train_images, val_images)

    print(f"Converted {train_count} train images and {val_count} validation images.")
    print(f"YOLO data file: {output_dir / 'bdd100k.yaml'}")


if __name__ == "__main__":
    main()
