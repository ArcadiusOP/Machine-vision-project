from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2


MODEL_PATH = Path(__file__).resolve().parent / "models" / "vehicle_detector.pt"
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"
YOLO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_CLASSES = {"car", "bus", "truck", "train", "motor", "motorcycle", "bike", "bicycle"}


class VehicleDetector:
    def __init__(self, model_path: Path = MODEL_PATH, confidence: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.names = {}
        self.using_default_model = False
        self.load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            return

        model_source = str(self.model_path) if self.model_path.exists() else DEFAULT_YOLO_WEIGHTS
        self.using_default_model = not self.model_path.exists()
        self.model = YOLO(model_source)
        self.names = self.model.names

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        if self.model is None:
            return []

        results = self.model.predict(frame, imgsz=640, conf=self.confidence, verbose=False)
        boxes: List[Tuple[int, int, int, int]] = []
        if not results:
            return boxes

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = str(self.names.get(class_id, class_id)).lower()
            if class_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes

    def track(self, frame) -> List[dict]:
        if self.model is None:
            return []

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=YOLO_VEHICLE_CLASS_IDS,
            conf=self.confidence,
            verbose=False,
            imgsz=960,
        )
        tracked = []
        if not results:
            return tracked

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracked

        for box, track_id in zip(boxes, boxes.id.int().tolist()):
            class_id = int(box.cls[0])
            class_name = str(self.names.get(class_id, class_id)).lower()
            if class_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tracked.append(
                {
                    "track_id": int(track_id),
                    "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "confidence": float(box.conf[0]),
                    "class_name": class_name,
                }
            )
        return tracked
