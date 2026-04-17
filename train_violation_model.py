from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from violation_model import FEATURE_COLUMNS, MODEL_PATH, features_to_array

LABEL_COLUMNS = ["overspeeding", "wrong_way", "illegal_uturn"]

def load_dataset(path: Path):   
    rows = []
    labels = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [column for column in FEATURE_COLUMNS + LABEL_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

        for row in reader:
            rows.append({column: float(row[column]) for column in FEATURE_COLUMNS})
            labels.append([int(float(row[column])) for column in LABEL_COLUMNS])

    if len(rows) < 20:
        raise ValueError("Add at least 20 labelled rows before training.")
    return features_to_array(rows), np.array(labels, dtype=int)

def train(dataset_path: Path, output_path: Path) -> None:
    x, y = load_dataset(dataset_path)
    pattern_labels = np.array(["".join(str(value) for value in row) for row in y])
    _, counts = np.unique(pattern_labels, return_counts=True)
    stratify = pattern_labels if np.min(counts) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.35,
        random_state=42,
        stratify=stratify,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=220,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                    )
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    print(classification_report(y_test, predictions, target_names=LABEL_COLUMNS, zero_division=0))
    output_path.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": LABEL_COLUMNS, "features": FEATURE_COLUMNS}, output_path)
    print(f"Saved trained model to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train illegal-driving classifier from labelled track features.")
    parser.add_argument("--dataset", default="data/violation_training_data.csv", help="CSV dataset path")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Model output path")
    args = parser.parse_args()
    train(Path(args.dataset), Path(args.output))

if __name__ == "__main__":
    main()
