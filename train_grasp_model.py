"""Train and evaluate an ML grasp-height selector.

Uses leave-one-geometry-out cross-validation.
Requires only NumPy and Python's standard library.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASET = Path(
    "grasp_dataset/parametric/trials_seed4210_n10.csv"
)

GRASP_OPTIONS = (0.2, 0.5, 0.8)
REGULARIZATION = 2.0
ITERATIONS = 2500
LEARNING_RATE = 0.1


def features(row):
    """Features available before attempting a grasp."""

    body_r = float(row["body_radius_m"])
    body_h = float(row["body_height_m"])
    neck_r = float(row["neck_radius_m"])
    neck_h = float(row["neck_height_m"])
    x = float(row["source_x_m"])
    y = float(row["source_y_m"])
    f = float(row["grasp_fraction_of_neck"])

    return np.array([
        body_r,
        body_h,
        neck_r,
        neck_h,
        x,
        y,
        f,
        f * f,
        f * (neck_r / body_r),
        f * neck_h,
    ], dtype=float)


def label(row):
    value = row["pour_success"].strip().lower()

    if value == "true":
        return 1.0

    if value == "false":
        return 0.0

    raise ValueError("Missing or invalid success label")


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


class GraspClassifier:
    def fit(self, rows):
        X = np.array([features(r) for r in rows])
        y = np.array([label(r) for r in rows])

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-10] = 1.0

        X = (X - self.mean) / self.std

        # Include an unregularized intercept.
        X = np.column_stack([np.ones(len(X)), X])

        weights = np.zeros(X.shape[1])
        penalty = np.ones(X.shape[1])
        penalty[0] = 0.0

        for _ in range(ITERATIONS):
            probabilities = sigmoid(X @ weights)

            gradient = (
                X.T @ (probabilities - y) / len(y)
                + REGULARIZATION
                * penalty * weights / len(y)
            )

            weights -= LEARNING_RATE * gradient

        self.weights = weights
        return self

    def predict_proba(self, rows):
        X = np.array([features(r) for r in rows])
        X = (X - self.mean) / self.std
        X = np.column_stack([np.ones(len(X)), X])

        return sigmoid(X @ self.weights)


def main():
    with DATASET.open(newline="") as file:
        rows = list(csv.DictReader(file))

    geometries = sorted(set(r["geometry_id"] for r in rows))

    scenarios = defaultdict(list)

    for row in rows:
        key = (row["geometry_id"], row["position_id"])
        scenarios[key].append(row)

    results = []

    print("\nLEAVE-ONE-GEOMETRY-OUT EVALUATION")
    print("=" * 75)

    for held_out in geometries:

        train = [
            r for r in rows
            if r["geometry_id"] != held_out
        ]

        model = GraspClassifier().fit(train)

        test_keys = sorted(
            key for key in scenarios
            if key[0] == held_out
        )

        for key in test_keys:
            candidates = sorted(
                scenarios[key],
                key=lambda r: float(
                    r["grasp_fraction_of_neck"]
                ),
            )

            if len(candidates) != len(GRASP_OPTIONS):
                raise ValueError(
                    f"Missing grasp candidates for {key}"
                )

            scores = model.predict_proba(candidates)
            ml_index = int(np.argmax(scores))

            actual = [int(label(r)) for r in candidates]
            fractions = [
                float(r["grasp_fraction_of_neck"])
                for r in candidates
            ]

            baseline_results = {}

            for f in GRASP_OPTIONS:
                idx = next(
                    i for i, v in enumerate(fractions)
                    if abs(v - f) < 1e-6
                )
                baseline_results[f] = actual[idx]

            result = {
                "geometry": held_out,
                "position": key[1],
                "ml_fraction": fractions[ml_index],
                "ml_success": actual[ml_index],
                "baseline_20": baseline_results[0.2],
                "baseline_50": baseline_results[0.5],
                "baseline_80": baseline_results[0.8],
                "oracle": max(actual),
            }

            results.append(result)

            print(
                f"{held_out} position {key[1]} | "
                f"ML chose {fractions[ml_index]:.0%} | "
                f"ML success: {actual[ml_index]} | "
                f"Best possible: {max(actual)}"
            )

    print("\n" + "=" * 75)
    print("POLICY COMPARISON")
    print("=" * 75)

    comparisons = {
        "ML selection": "ml_success",
        "Fixed 20%": "baseline_20",
        "Fixed 50%": "baseline_50",
        "Fixed 80%": "baseline_80",
        "Oracle (upper bound)": "oracle",
    }

    for name, column in comparisons.items():
        n = len(results)
        successes = sum(r[column] for r in results)

        print(
            f"{name:23s}: "
            f"{successes:2d}/{n} = {successes/n:.1%}"
        )

    output = Path(
        "grasp_dataset/parametric/ml_evaluation.csv"
    )

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved detailed results:", output)


if __name__ == "__main__":
    main()
