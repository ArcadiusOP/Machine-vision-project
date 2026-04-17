from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "violation_model.joblib"

FEATURE_COLUMNS = [
    "speed_kmh",
    "speed_over_limit",
    "speed_ratio",
    "direction_dx",
    "direction_dy",
    "path_length",
    "displacement",
    "straightness",
    "turn_angle",
    "wrong_way_score",
    "seen_count",
]


def expected_vector(expected_direction: str) -> np.ndarray:
    vectors = {
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
        "down": np.array([0.0, 1.0]),
        "up": np.array([0.0, -1.0]),
    }
    return vectors.get(expected_direction, vectors["right"])


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    mag1 = float(np.linalg.norm(v1))
    mag2 = float(np.linalg.norm(v2))
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    value = float(np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0))
    return math.degrees(math.acos(value))


def track_feature_dict(track, speed_limit_kmh: float, expected_direction_name: str) -> Dict[str, float]:
    history = np.array(track.history, dtype=float)
    if len(history) < 2:
        dx = dy = path_length = displacement = turn_angle = 0.0
    else:
        deltas = np.diff(history, axis=0)
        path_length = float(np.sum(np.linalg.norm(deltas, axis=1)))
        net = history[-1] - history[0]
        dx, dy = float(net[0]), float(net[1])
        displacement = float(np.linalg.norm(net))
        early = history[min(8, len(history) - 1)] - history[0]
        recent_start = history[max(0, len(history) - 12)]
        recent = history[-1] - recent_start
        turn_angle = vector_angle(early, recent)

    straightness = displacement / path_length if path_length > 1e-6 else 0.0
    direction = np.array([dx, dy], dtype=float)
    expected = expected_vector(expected_direction_name)
    if np.linalg.norm(direction) < 1e-6:
        wrong_way_score = 0.0
    else:
        wrong_way_score = max(0.0, -float(np.dot(direction / np.linalg.norm(direction), expected)))

    return {
        "speed_kmh": float(track.speed_kmh),
        "speed_over_limit": float(track.speed_kmh - speed_limit_kmh),
        "speed_ratio": float(track.speed_kmh / max(1.0, speed_limit_kmh)),
        "direction_dx": dx,
        "direction_dy": dy,
        "path_length": path_length,
        "displacement": displacement,
        "straightness": straightness,
        "turn_angle": turn_angle,
        "wrong_way_score": wrong_way_score,
        "seen_count": float(track.seen_count),
    }


def derived_feature_value(row: Dict[str, float], column: str) -> float:
    if column in row:
        return float(row[column])
    speed = float(row.get("speed_kmh", 0.0))
    speed_over_limit = float(row.get("speed_over_limit", 0.0))
    speed_limit = max(1.0, speed - speed_over_limit)
    if column == "speed_ratio":
        return float(speed / speed_limit)
    return 0.0


def features_to_array(feature_rows: Iterable[Dict[str, float]]) -> np.ndarray:
    return np.array([[derived_feature_value(row, column) for column in FEATURE_COLUMNS] for row in feature_rows], dtype=float)


class ViolationModel:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.pipeline = None
        self.labels: List[str] = []
        self.load()

    @property
    def available(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        if not self.model_path.exists():
            return
        bundle = joblib.load(self.model_path)
        self.pipeline = bundle["pipeline"]
        self.labels = list(bundle["labels"])

    def predict(self, feature_row: Dict[str, float]) -> Optional[Dict[str, float]]:
        if self.pipeline is None:
            return None
        probabilities = self.pipeline.predict_proba(features_to_array([feature_row]))
        result = {}
        for label, class_probs in zip(self.labels, probabilities):
            classes = list(self.pipeline.named_steps["classifier"].estimators_[self.labels.index(label)].classes_)
            if 1 in classes:
                result[label] = float(class_probs[0][classes.index(1)])
            else:
                result[label] = 0.0
        return result
