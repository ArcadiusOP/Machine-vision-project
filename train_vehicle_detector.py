from __future__ import annotations

import argparse
from pathlib import Path


def train(data_yaml: Path, output_name: str, epochs: int, image_size: int, batch: int) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install -r requirements.txt") from exc

    # This uses the YOLO architecture definition only. It does not load pretrained weights.
    model = YOLO("yolo11n.yaml")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        name=output_name,
        pretrained=False,
    )

    run_dir = Path("runs") / "detect" / output_name / "weights"
    best_model = run_dir / "best.pt"
    target = Path("models") / "vehicle_detector.pt"
    if best_model.exists():
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(best_model.read_bytes())
        print(f"Copied best detector to {target}")
    else:
        print(f"Training finished, but {best_model} was not found. Check the runs folder.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom vehicle detector from scratch.")
    parser.add_argument("--data", default="data/bdd100k_yolo/bdd100k.yaml", help="YOLO data yaml")
    parser.add_argument("--name", default="bdd100k_vehicle_detector", help="Ultralytics run name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    train(Path(args.data), args.name, args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()
