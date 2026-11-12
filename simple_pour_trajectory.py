"""Static Drake / Meshcat demonstration of a simplified labware pouring model.

Run from the repository root (in an environment with Drake and NumPy):

    python simple_pour_trajectory.py

The trajectory computation is independent of the scene and works with a Drake
RigidTransform or a NumPy 4x4 homogeneous transform. Call compute_pour_trajectory
with the *current* vessel body pose on every robot-control update, then call
render_pour_trajectory to refresh the Meshcat overlay.

This is deliberately a visual/planning surrogate, not a fluid simulation:
- a heuristic fill-dependent tilt threshold gates release;
- a single low-rim point emits a ballistic line;
- launch speed and direction are approximate;
- no free-surface dynamics, splashing, wall contact, or volume depletion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

GRAVITY = 9.81                  # m/s^2
TABLE_TOP = 0.20               # same as visualize_station.py
TABLE_CENTER = (0.60, 0.00, 0.17)
PACKAGE_XML = Path(__file__).resolve().parent / "models" / "package.xml"


@dataclass(frozen=True)
class VesselSpec:
    """Mouth and centerline geometry in the vessel's SDF body frame (meters)."""

    name: str
    height: float                 # mouth z above bottom/body origin
    mouth_radius: float           # opening radius (not the vessel's widest radius)
    fill_display_radius: float    # only for the schematic fill indicator


# Read directly from the dimensions in models/*.sdf.
VESSELS = {
    "test_tube": VesselSpec("test_tube", 0.12, 0.012, 0.007),
    "erlenmeyer": VesselSpec("erlenmeyer", 0.16, 0.016, 0.012),
    "boiling_flask": VesselSpec("boiling_flask", 0.17, 0.018, 0.014),
}


@dataclass(frozen=True)
class PourResult:
    active: bool
    points_W: np.ndarray          # shape (N, 3), meters in the world frame
    mouth_W: np.ndarray           # release position at the lower rim, shape (3,)
    tilt_deg: float
    onset_deg: float
    launch_velocity_W: np.ndarray  # velocity including vessel motion, m/s


def _pose_components(X_WV):
    """Return rotation and translation of world-from-vessel pose.

    X_WV can be a pydrake.math.RigidTransform or a homogeneous 4x4 matrix.
    """
    if hasattr(X_WV, "rotation") and hasattr(X_WV, "translation"):
        return (np.asarray(X_WV.rotation().matrix(), dtype=float),
                np.asarray(X_WV.translation(), dtype=float))
    X = np.asarray(X_WV, dtype=float)
    if X.shape != (4, 4):
        raise ValueError("X_WV must be a Drake RigidTransform or 4x4 matrix")
    return X[:3, :3], X[:3, 3]


def compute_pour_trajectory(
    X_WV,
    vessel: VesselSpec,
    fill_fraction: float,
    *,
    linear_velocity_W=(0.0, 0.0, 0.0),
    angular_velocity_W=(0.0, 0.0, 0.0),
    surface_z: float = TABLE_TOP,
    sample_dt: float = 0.015,
    max_time: float = 1.5,
) -> PourResult:
    """Compute the instantaneous jet path emitted from the *lowest mouth rim*.

    The vessel-local +z axis points from the base toward the opening. The
    kinematic interface accepts both translational and angular vessel velocity,
    so the same call can eventually use a robot-held body pose/velocity.

    The heuristic onset angle (25-60 deg for fills of 100-0%) is intentionally
    not a fluid-static derivation. It makes the visible surrogate responsive to
    fill fraction, and should be replaced/calibrated for each vessel later.
    All positions, distances, and times use SI units.
    """
    if not (0.0 <= fill_fraction <= 1.0):
        raise ValueError("fill_fraction must lie in [0, 1]")
    if sample_dt <= 0 or max_time <= 0:
        raise ValueError("sample_dt and max_time must be positive")
    R, p = _pose_components(X_WV)
    v = np.asarray(linear_velocity_W, dtype=float).reshape(3)
    omega = np.asarray(angular_velocity_W, dtype=float).reshape(3)
    axis_W = R[:, 2]
    tilt_deg = float(np.degrees(np.arccos(np.clip(axis_W[2], -1.0, 1.0))))
    onset_deg = 25.0 + 35.0 * (1.0 - fill_fraction)

    # The lowest point on the (circular) mouth rim, chosen in vessel coordinates.
    gravity_in_vessel = R.T @ np.array([0.0, 0.0, -1.0])
    rim_xy = gravity_in_vessel[:2]
    rim_norm = np.linalg.norm(rim_xy)
    if rim_norm > 1e-9:
        rim_xy = rim_xy / rim_norm
    else:
        rim_xy = np.array([1.0, 0.0])  # only relevant when no flow
    mouth_V = np.array([vessel.mouth_radius * rim_xy[0],
                        vessel.mouth_radius * rim_xy[1], vessel.height])
    mouth_W = p + R @ mouth_V

    no_flow = (fill_fraction <= 0 or tilt_deg < onset_deg or
               mouth_W[2] <= surface_z)
    if no_flow:
        return PourResult(False, np.empty((0, 3)), mouth_W, tilt_deg,
                          onset_deg, np.zeros(3))

    # Project the tilted vessel axis into the horizontal world plane. The jet
    # exits broadly in that direction, with a small downward launch component.
    horizontal = np.array([axis_W[0], axis_W[1], 0.0])
    horizontal /= np.linalg.norm(horizontal)
    release_dir = horizontal + np.array([0.0, 0.0, -0.15])
    release_dir /= np.linalg.norm(release_dir)

    # Bernoulli-inspired velocity scale from nominal liquid head, attenuated
    # smoothly near onset. This is a modeling knob, NOT a validated flow rate.
    head_m = max(0.005, fill_fraction * vessel.height * 0.6)
    activation = np.clip((tilt_deg - onset_deg) / 35.0, 0.0, 1.0)
    speed = 0.75 * np.sqrt(2 * GRAVITY * head_m) * np.sqrt(activation)
    v_mouth = v + np.cross(omega, R @ mouth_V)
    launch_velocity_W = v_mouth + speed * release_dir

    # End at the first intersection with the horizontal surface (table top).
    # Ballistic equation: z(t) = z0 + vz*t - 0.5*g*t^2.
    dz = mouth_W[2] - surface_z
    t_hit = (launch_velocity_W[2] +
             np.sqrt(launch_velocity_W[2] ** 2 + 2 * GRAVITY * dz)) / GRAVITY
    duration = min(float(t_hit), float(max_time))
    times = np.arange(0.0, duration, sample_dt)
    times = np.append(times, duration)
    points = (mouth_W[None, :] + times[:, None] * launch_velocity_W[None, :]
              + 0.5 * times[:, None] ** 2 * np.array([0.0, 0.0, -GRAVITY]))
    return PourResult(True, points, mouth_W, tilt_deg, onset_deg,
                      launch_velocity_W)


def target_opening_error(
    result: PourResult,
    target_center_xy,
    target_opening_z: float,
    target_radius: float,
) -> tuple[bool, float]:
    """Check whether the descending jet enters a horizontal circular opening.

    Returns (hit, distance_from_center_m). A path that never crosses the opening
    plane downward returns (False, inf). Intended as a cheap planning score, not
    as collision detection or simulated liquid collection. For valid scoring,
    compute a trajectory with surface_z <= target_opening_z so it reaches the
    opening plane rather than stopping at the table prematurely.
    """
    if target_radius <= 0:
        raise ValueError("target_radius must be positive")
    if not result.active:
        return False, float("inf")
    pts = result.points_W
    for i in range(1, len(pts)):
        z0, z1 = pts[i - 1, 2], pts[i, 2]
        if z0 >= target_opening_z >= z1 and z1 < z0:
            alpha = (target_opening_z - z0) / (z1 - z0)
            xy = pts[i - 1, :2] + alpha * (pts[i, :2] - pts[i - 1, :2])
            miss_m = float(np.linalg.norm(xy - np.asarray(target_center_xy)))
            return miss_m <= target_radius, miss_m
    return False, float("inf")


def render_pour_trajectory(meshcat, path: str, result: PourResult) -> None:
    """Replace an existing world-frame Meshcat jet; delete it when inactive."""
    from pydrake.all import Rgba

    meshcat.Delete(path)
    if result.active:
        meshcat.SetLine(path, result.points_W.T, line_width=4.0,
                        rgba=Rgba(0.05, 0.50, 0.95, 1.0))


def build_demo():
    """Show all three existing source SDFs, statically tilted above the table."""
    from pydrake.all import (
        AddMultibodyPlantSceneGraph, Cylinder, DiagramBuilder, Meshcat,
        MeshcatVisualizer, Parser, Rgba, RigidTransform, RollPitchYaw, Simulator,
    )

    # name, translation (m), RPY degrees, filled fraction (0..1).
    # Values chosen to show different trajectories, not physical equilibrium.
    examples = [
        ("test_tube", (0.31, -0.23, 0.54), (0, 63, 0), 0.85),
        ("erlenmeyer", (0.46, 0.11, 0.55), (0, 54, 35), 0.65),
        ("boiling_flask", (0.68, -0.12, 0.59), (-52, 0, -25), 0.75),
    ]

    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(str(PACKAGE_XML))

    table = parser.AddModelsFromUrl("package://pouring_bot/table.sdf")[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", table),
                     RigidTransform(TABLE_CENTER))

    # Receiver shown as in visualize_station.py. It is a visual target only;
    # ballistic paths presently terminate on the table plane, not its opening.
    receiver = parser.AddModelsFromUrl("package://pouring_bot/beaker.sdf")[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", receiver),
                     RigidTransform([0.72, 0.14, TABLE_TOP]))

    source_instances = []
    for name, position, rpy_deg, fraction in examples:
        model = parser.AddModelsFromUrl(f"package://pouring_bot/{name}.sdf")[0]
        X_WV = RigidTransform(RollPitchYaw(np.deg2rad(rpy_deg)), position)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body", model), X_WV)
        source_instances.append((name, model, fraction))

    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    diagram.ForcedPublish(simulator.get_context())
    plant_context = plant.GetMyContextFromRoot(simulator.get_context())

    print("\nSimple pouring model: static source poses and schematic fill indicators")
    print("The robot is intentionally omitted from this isolated physics demo.")
    for name, model, fraction in source_instances:
        spec = VESSELS[name]
        # Query the actual Drake body pose; this same pattern works when the
        # body eventually moves with the robot instead of being welded.
        body = plant.GetBodyByName("body", model)
        X_WV = plant.EvalBodyPoseInWorld(plant_context, body)
        result = compute_pour_trajectory(X_WV, spec, fraction)
        render_pour_trajectory(meshcat, f"pouring/{name}/jet", result)

        # A *schematic* colored core illustrates the assigned fill amount. It
        # is not a liquid mesh, real free surface, or mass-conserving volume.
        fill_height = max(0.001, fraction * spec.height)
        fill_path = f"pouring/{name}/schematic_fill"
        meshcat.SetObject(fill_path,
                          Cylinder(spec.fill_display_radius, fill_height),
                          Rgba(0.12, 0.64, 0.96, 0.55))
        meshcat.SetTransform(
            fill_path, X_WV @ RigidTransform([0.0, 0.0, fill_height / 2.0]))
        print(f"  {name:14s} fill={fraction:.0%} tilt={result.tilt_deg:5.1f} deg "
              f"onset={result.onset_deg:4.1f} deg "
              f"pouring={result.active} points={len(result.points_W)}")

    print("\nReceiver (beaker) opening: z = 0.29 m, radius = 0.04 m.")
    print("Use target_opening_error(result, (0.72, 0.14), 0.29, 0.04) for planning.")
    print("\nOpen Meshcat:", meshcat.web_url())
    print("The blue jet overlays are static. No particles or real fluid are simulated.")
    input("Press Enter to close the visualization.\n")


if __name__ == "__main__":
    build_demo()
