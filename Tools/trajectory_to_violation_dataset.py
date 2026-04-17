from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from violation_model import FEATURE_COLUMNS, expected_vector, vector_angle
from train_violation_model import LABEL_COLUMNS


def as_float(row: dict, names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] != "":
            return float(row[name])
    return default


def as_int(row: dict, names: Iterable[str], default: int = 0) -> int:
    return int(round(as_float(row, names, default)))


def track_id(row: dict) -> int:
    return as_int(row, ["trackId", "id"])


def frame_id(row: dict) -> int:
    return as_int(row, ["frame", "frameId"])


def point(row: dict) -> Tuple[float, float]:
    return (
        as_float(row, ["xCenter", "x", "centerX"]),
        as_float(row, ["yCenter", "y", "centerY"]),
    )


def speed_kmh(row: dict) -> float:
    if "speed" in row and row["speed"] != "":
        return float(row["speed"]) * 3.6
    vx = as_float(row, ["xVelocity", "vx", "lonVelocity"])
    vy = as_float(row, ["yVelocity", "vy", "latVelocity"])
    return math.hypot(vx, vy) * 3.6


def path_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def build_feature_row(
    rows: List[dict],
    speed_limit_kmh: float,
    expected_direction_name: str,
    uturn_angle_threshold: float,
) -> Dict[str, float]:
    rows = sorted(rows, key=frame_id)
    points = [point(row) for row in rows]
    speeds = [speed_kmh(row) for row in rows]

    if len(points) < 2:
        dx = dy = displacement = turn_angle = 0.0
    else:
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        displacement = math.hypot(dx, dy)
        early = points[min(8, len(points) - 1)]
        recent_start = points[max(0, len(points) - 12)]
        initial_vector = (early[0] - points[0][0], early[1] - points[0][1])
        recent_vector = (points[-1][0] - recent_start[0], points[-1][1] - recent_start[1])
        turn_angle = vector_angle(initial_vector, recent_vector)

    total_path = path_length(points)
    straightness = displacement / total_path if total_path > 1e-6 else 0.0
    direction_norm = math.hypot(dx, dy)
    expected = expected_vector(expected_direction_name)
    if direction_norm < 1e-6:
        wrong_way_score = 0.0
    else:
        unit_direction = (dx / direction_norm, dy / direction_norm)
        wrong_way_score = max(0.0, -(unit_direction[0] * expected[0] + unit_direction[1] * expected[1]))

    mean_speed = sum(speeds) / max(1, len(speeds))
    return {
        "speed_kmh": mean_speed,
        "speed_over_limit": mean_speed - speed_limit_kmh,
        "speed_ratio": mean_speed / max(1.0, speed_limit_kmh),
        "direction_dx": dx,
        "direction_dy": dy,
        "path_length": total_path,
        "displacement": displacement,
        "straightness": straightness,
        "turn_angle": turn_angle,
        "wrong_way_score": wrong_way_score,
        "seen_count": float(len(rows)),
        "overspeeding": int(mean_speed > speed_limit_kmh),
        "wrong_way": int(wrong_way_score > 0.65 and displacement > 8.0),
        "illegal_uturn": int(turn_angle >= uturn_angle_threshold and straightness < 0.55 and total_path > 15.0),
    }


def convert(root: Path, output: Path, speed_limit_kmh: float, expected_direction_name: str, uturn_angle_threshold: float) -> None:
    track_files = sorted(root.rglob("*_tracks.csv"))
    if not track_files:
        raise FileNotFoundError(f"No *_tracks.csv files found under {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FEATURE_COLUMNS + LABEL_COLUMNS)
        writer.writeheader()

        for track_file in track_files:
            grouped: Dict[int, List[dict]] = {}
            with track_file.open("r", newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                for row in reader:
                    grouped.setdefault(track_id(row), []).append(row)

            for rows in grouped.values():
                if len(rows) < 12:
                    continue
                feature_row = build_feature_row(rows, speed_limit_kmh, expected_direction_name, uturn_angle_threshold)
                writer.writerow({key: round(float(feature_row[key]), 6) for key in FEATURE_COLUMNS + LABEL_COLUMNS})
                written += 1

    print(f"Wrote {written} labelled trajectory rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert trajectory datasets such as inD, rounD, or highD into violation model training rows."
    )
    parser.add_argument("--root", required=True, help="Dataset folder containing *_tracks.csv files")
    parser.add_argument("--output", default="data/violation_training_data_from_trajectories.csv")
    parser.add_argument("--speed-limit", type=float, default=50.0)
    parser.add_argument("--expected-direction", choices=["right", "left", "down", "up"], default="right")
    parser.add_argument("--uturn-angle-threshold", type=float, default=130.0)
    args = parser.parse_args()
    convert(Path(args.root), Path(args.output), args.speed_limit, args.expected_direction, args.uturn_angle_threshold)


if __name__ == "__main__":
    main()
