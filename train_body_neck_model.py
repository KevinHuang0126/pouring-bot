"""Train a geometry-aware grasp selector using NumPy only.

Evaluation: leave-one-vessel-geometry-out cross-validation.
Only width-feasible candidates are considered for selection.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DATASET = Path(
    "grasp_dataset/parametric/trials_body_neck_seed4210_n10.csv"
)

OUTPUT = Path(
    "grasp_dataset/parametric/body_neck_ml_evaluation.csv"
)

REGULARIZATION = 2.0
LEARNING_RATE = 0.1
ITERATIONS = 3000

OPTIONS = (
    ("body", 0.4),
    ("body", 0.75),
    ("neck", 0.2),
    ("neck", 0.5),
    ("neck", 0.8),
)


def is_true(value):
    return str(value).strip().lower() == "true"


def feasible(row):
    return is_true(row["width_feasible"])


def outcome(row):
    return int(is_true(row["pour_success"]))


def feature_vector(row):
    br = float(row["body_radius_m"])
    bh = float(row["body_height_m"])
    nr = float(row["neck_radius_m"])
    nh = float(row["neck_height_m"])
    mass = float(row["mass_kg"])
    x = float(row["source_x_m"])
    y = float(row["source_y_m"])
    h = float(row["grasp_height_m"])

    neck = float(row["grasp_region"] == "neck")
    fraction = float(row["grasp_fraction_of_region"])
    local_radius = float(row["local_radius_m"])
    total_h = bh + nh

    return np.array([
        br, bh, nr, nh,
        mass, x, y,
        h, h / total_h,
        neck, fraction,
        local_radius,
        nr / br,
        h * neck,
        h / max(local_radius, 1e-6),
        x * h,
        y * h,
    ], dtype=float)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


class LogisticGraspModel:

    def fit(self, rows):
        X = np.array([feature_vector(r) for r in rows])
        y = np.array([outcome(r) for r in rows])

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-10] = 1.0

        X = (X - self.mean) / self.std
        X = np.column_stack([np.ones(len(X)), X])

        w = np.zeros(X.shape[1])
        penalty = np.ones(len(w))
        penalty[0] = 0.0

        for _ in range(ITERATIONS):
            p = sigmoid(X @ w)
            gradient = (
                X.T @ (p - y) / len(y)
                + REGULARIZATION * penalty * w / len(y)
            )
            w -= LEARNING_RATE * gradient

        self.weights = w
        return self

    def predict(self, rows):
        X = np.array([feature_vector(r) for r in rows])
        X = (X - self.mean) / self.std
        X = np.column_stack([np.ones(len(X)), X])
        return sigmoid(X @ self.weights)


def candidate_name(row):
    region = row["grasp_region"]
    fraction = float(row["grasp_fraction_of_region"])
    return f"{region}_{round(fraction * 100)}"


def choose_fixed(candidates, region, fraction):
    for row in candidates:
        if (
            row["grasp_region"] == region
            and abs(
                float(row["grasp_fraction_of_region"]) - fraction
            ) < 1e-6
        ):
            return row

    # A fixed policy cannot execute an infeasible grasp.
    return None


def main():
    with DATASET.open(newline="") as f:
        all_rows = list(csv.DictReader(f))

    geometries = sorted({
        row["geometry_id"] for row in all_rows
    })

    scenarios = defaultdict(list)
    for row in all_rows:
        key = (row["geometry_id"], row["position_id"])
        scenarios[key].append(row)

    results = []

    print("\nFIVE-CANDIDATE GRASP SELECTION")
    print("Leave-one-geometry-out evaluation")
    print("=" * 75)

    for held_out in geometries:
        training = [
            row for row in all_rows
            if row["geometry_id"] != held_out
            and feasible(row)
            and row["failure_reason"] != "ik_tracking_failure"
        ]

        model = LogisticGraspModel().fit(training)

        keys = sorted(
            key for key in scenarios
            if key[0] == held_out
        )

        for key in keys:
            candidates = [
                row for row in scenarios[key]
                if feasible(row)
            ]

            if not candidates:
                raise RuntimeError(f"No feasible candidates: {key}")

            scores = model.predict(candidates)
            selected = candidates[int(np.argmax(scores))]

            baseline_20 = choose_fixed(
                candidates, "neck", 0.2
            )
            baseline_50 = choose_fixed(
                candidates, "neck", 0.5
            )
            baseline_body = choose_fixed(
                candidates, "body", 0.75
            )

            # Deterministic geometry heuristic:
            # prefer lower neck, then middle neck,
            # then upper body, then any feasible grasp.
            heuristic = next(
                (
                    choose_fixed(candidates, region, fraction)
                    for region, fraction in (
                        ("neck", 0.2),
                        ("neck", 0.5),
                        ("body", 0.75),
                        ("neck", 0.8),
                        ("body", 0.4),
                    )
                    if choose_fixed(candidates, region, fraction)
                    is not None
                ),
                None,
            )

            result = {
                "geometry_id": held_out,
                "position_id": key[1],
                "ml_choice": candidate_name(selected),
                "ml_success": outcome(selected),
                "fixed_neck_20": (
                    outcome(baseline_20) if baseline_20 else 0
                ),
                "fixed_neck_50": (
                    outcome(baseline_50) if baseline_50 else 0
                ),
                "fixed_body_75": (
                    outcome(baseline_body) if baseline_body else 0
                ),
                "geometry_heuristic": outcome(heuristic),
                "oracle": max(outcome(r) for r in candidates),
            }

            results.append(result)

            print(
                f"{held_out} position {key[1]} | "
                f"ML: {result['ml_choice']:10s} | "
                f"Success: {result['ml_success']} | "
                f"Oracle: {result['oracle']}"
            )

    print("\n" + "=" * 75)
    print("POLICY COMPARISON")
    print("=" * 75)

    policies = {
        "ML selection": "ml_success",
        "Fixed neck 20%": "fixed_neck_20",
        "Fixed neck 50%": "fixed_neck_50",
        "Fixed body 75%": "fixed_body_75",
        "Geometry heuristic": "geometry_heuristic",
        "Oracle (upper bound)": "oracle",
    }

    for name, column in policies.items():
        successes = sum(r[column] for r in results)
        total = len(results)
        print(
            f"{name:25s}: "
            f"{successes:2d}/{total} = "
            f"{successes / total:.1%}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
