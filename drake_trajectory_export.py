"""Export Drake rigid-body poses for replay in a fluid simulator."""

import csv
from pathlib import Path

import numpy as np


class TrajectoryRecorder:
    """Record time-stamped rigid-body poses in world coordinates."""

    HEADER = [
        "time_s",
        "body",
        "x_m", "y_m", "z_m",
        "qw", "qx", "qy", "qz",
    ]

    def __init__(self):
        self.rows = []

    def record(self, time_s, body, pose):
        """Record a Drake RigidTransform.

        Args:
            time_s: Simulation time in seconds.
            body: Unique rigid-body name.
            pose: Drake RigidTransform, pose X_WB mapping body coordinates into world coordinates.
        """
        position = np.asarray(pose.translation(), dtype=float)
        quat = pose.rotation().ToQuaternion()

        values = [
            float(time_s),
            str(body),
            *position.tolist(),
            float(quat.w()),
            float(quat.x()),
            float(quat.y()),
            float(quat.z()),
        ]

        if not np.all(np.isfinite(values[2:])):
            raise ValueError("Non-finite pose detected")

        self.rows.append(values)

    def save(self, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)
            writer.writerows(self.rows)

        return path


def load_trajectory(filename):
    """Load and validate a trajectory for downstream simulation."""
    from collections import defaultdict

    trajectories = defaultdict(list)

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            body = row["body"]
            values = np.array([
                float(row[k])
                for k in (
                    "x_m", "y_m", "z_m",
                    "qw", "qx", "qy", "qz",
                )
            ])

            if not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid pose for {body}")

            quaternion = values[3:]
            norm = np.linalg.norm(quaternion)

            if abs(norm - 1.0) > 1e-4:
                raise ValueError(
                    f"Invalid quaternion for {body}: norm={norm}"
                )

            trajectories[body].append(
                (float(row["time_s"]), values)
            )

    for body, samples in trajectories.items():
        times = np.array([sample[0] for sample in samples])

        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise ValueError(
                f"Non-increasing timestamps for {body}"
            )

    return dict(trajectories)
