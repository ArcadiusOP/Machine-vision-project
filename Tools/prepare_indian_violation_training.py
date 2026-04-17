from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_violation_model import LABEL_COLUMNS
from violation_model import FEATURE_COLUMNS


def yes_no_to_float(value: str) -> float:
    value = (value or "").strip().lower()
    if value in {"yes", "positive", "true"}:
        return 1.0
    if value in {"no", "negative", "false", "not paid", "not conducted", "n/a"}:
        return 0.0
    return 0.0


def build_row(source: dict) -> dict:
    speed_limit = float(source.get("Speed_Limit", 0) or 0)
    recorded_speed = float(source.get("Recorded_Speed", 0) or 0)
    over_limit = recorded_speed - speed_limit
    violation_type = (source.get("Violation_Type") or "").strip().lower()

    row = {
        "speed_kmh": recorded_speed,
        "speed_over_limit": over_limit,
        "speed_ratio": recorded_speed / max(1.0, speed_limit),
        "direction_dx": 0.0,
        "direction_dy": 0.0,
        "path_length": max(24.0, recorded_speed * 2.1),
        "displacement": max(20.0, recorded_speed * 1.8),
        "straightness": 0.96,
        "turn_angle": 0.0,
        "wrong_way_score": 0.0,
        "seen_count": 24.0 + float(source.get("Previous_Violations", 0) or 0),
        "overspeeding": 1 if violation_type == "over-speeding" or over_limit > 0 else 0,
        "wrong_way": 0,
        "illegal_uturn": 0,
    }
    return row


def write_rows(writer: csv.DictWriter, input_path: Path) -> int:
    count = 0
    with input_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for source in reader:
            writer.writerow(build_row(source))
            count += 1
    return count


def merge(seed_path: Path, indian_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FEATURE_COLUMNS + LABEL_COLUMNS)
        writer.writeheader()

        with seed_path.open("r", newline="", encoding="utf-8") as seed_file:
            seed_reader = csv.DictReader(seed_file)
            for row in seed_reader:
                writer.writerow(row)
                total += 1

        total += write_rows(writer, indian_path)

    print(f"Wrote merged training dataset with {total} rows to {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    merge(
        project_root / "data" / "violation_training_seed.csv",
        project_root / "data" / "Indian_Traffic_Violations.csv",
        project_root / "data" / "violation_training_data.csv",
    )
