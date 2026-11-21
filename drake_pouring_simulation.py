"""Drake-driven pouring experiment with ballistic liquid particles.

Drake handles the source vessel's revolute-joint kinematics.
Liquid particles are a simplified ballistic approximation, not an SPH fluid.
"""

from __future__ import annotations
from drake_trajectory_export import TrajectoryRecorder

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from pydrake.all import (
    Box,
    Cylinder,
    FixedOffsetFrame,
    Meshcat,
    MultibodyPlant,
    RevoluteJoint,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Rgba,
    SpatialInertia,
    Sphere,
    UnitInertia,
)


# ----------------------------- Scene parameters -----------------------------

GRAVITY = 9.81

SOURCE_INNER_RADIUS = 0.035
SOURCE_WALL_THICKNESS = 0.003
SOURCE_HEIGHT = 0.110

RECEIVER_INNER_RADIUS = 0.060
RECEIVER_WALL_THICKNESS = 0.004
RECEIVER_HEIGHT = 0.120

SOURCE_PIVOT_W = np.array([0.0, 0.0, 0.160])
RECEIVER_CENTER_W = np.array([0.110, 0.0, 0.0])

PARTICLE_SPACING = 0.008
PARTICLE_RADIUS = 0.0025

SIMULATION_DURATION = 3.0
TIME_STEP = 0.01
POUR_RAMP_DURATION = 1.5
FINAL_TILT_DEG = 105.0

OUTPUT_CSV = Path(__file__).resolve().parent / "drake_pour_particles.csv"


# ------------------------------- Drake plant --------------------------------

def build_source_plant():
    """Build a Drake MultibodyPlant with a source vessel on a revolute joint."""
    plant = MultibodyPlant(time_step=0.0)

    model = plant.AddModelInstance("pouring_demo")

    inertia = SpatialInertia(
        mass=0.5,
        p_PScm_E=np.array([0.0, 0.0, SOURCE_HEIGHT / 2]),
        G_SP_E=UnitInertia(0.01, 0.01, 0.01),
    )
    source_body = plant.AddRigidBody("source_vessel", model, inertia)

    # The joint axis is world/body Y. Positive rotation tips the vessel
    # opening toward +X.
    pivot_frame = plant.AddFrame(
        FixedOffsetFrame(
            "source_pivot",
            plant.world_frame(),
            RigidTransform(SOURCE_PIVOT_W),
        )
    )

    joint = plant.AddJoint(
        RevoluteJoint(
            "source_tilt",
            pivot_frame,
            source_body.body_frame(),
            [0.0, 1.0, 0.0],
        )
    )

    plant.Finalize()
    context = plant.CreateDefaultContext()

    return plant, context, source_body, joint


def source_pose_at_time(plant, context, source_body, joint, t):
    """Return source pose, tilt angle, and world angular velocity."""
    ramp_fraction = np.clip(t / POUR_RAMP_DURATION, 0.0, 1.0)

    smooth_fraction = ramp_fraction**2 * (3.0 - 2.0 * ramp_fraction)
    angle_max = np.deg2rad(FINAL_TILT_DEG)
    angle = angle_max * smooth_fraction

    # Derivative of the smoothstep trajectory.
    angular_speed = (
        angle_max
        * 6.0 * ramp_fraction * (1.0 - ramp_fraction)
        / POUR_RAMP_DURATION
    )

    joint.set_angle(context, angle)
    X_WB = plant.EvalBodyPoseInWorld(context, source_body)

    # Positive rotation is about world Y.
    angular_velocity_W = np.array([0.0, angular_speed, 0.0])

    return X_WB, angle, angular_velocity_W


# ------------------------------ Vessel visuals ------------------------------

def add_hollow_vessel_visual(
    meshcat,
    path,
    radius,
    height,
    wall_thickness,
    color,
    *,
    segments=20,
):
    """Draw a hollow vessel from a bottom and thin wall panels.

    The panels are visual geometry in Meshcat; they are not collision
    boundaries and do not physically constrain the ballistic particles.
    """
    bottom_thickness = 0.004

    meshcat.SetObject(
        f"{path}/bottom",
        Cylinder(radius, bottom_thickness),
        color,
    )
    meshcat.SetTransform(
        f"{path}/bottom",
        RigidTransform([0.0, 0.0, bottom_thickness / 2]),
    )

    # Approximate a cylindrical wall with thin rectangular panels.
    chord = 2.0 * radius * np.sin(np.pi / segments)
    panel_thickness = wall_thickness
    panel = Box(chord, panel_thickness, height)

    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Local X is tangent to the circular wall.
        yaw = angle + np.pi / 2.0
        X = RigidTransform(
            RollPitchYaw(0.0, 0.0, yaw),
            [x, y, height / 2],
        )

        meshcat.SetObject(f"{path}/wall_{i}", panel, color)
        meshcat.SetTransform(f"{path}/wall_{i}", X)


def update_source_visual(meshcat, X_WB):
    """Move the source vessel's Meshcat geometry with the Drake body pose."""
    # Each source component is expressed in the vessel's body frame.
    for component in ["bottom"] + [
        f"wall_{i}" for i in range(20)
    ]:
        path = f"pouring/source/{component}"
        # The component's local transform is stored below by this helper.
        X_BC = SOURCE_VISUAL_TRANSFORMS[component]
        meshcat.SetTransform(path, X_WB @ X_BC)


SOURCE_VISUAL_TRANSFORMS = {}


def add_source_visual(meshcat):
    """Create source panels and retain their transforms for animation."""
    radius = SOURCE_INNER_RADIUS + SOURCE_WALL_THICKNESS / 2
    segments = 20
    bottom_thickness = 0.004

    SOURCE_VISUAL_TRANSFORMS["bottom"] = RigidTransform(
        [0.0, 0.0, bottom_thickness / 2]
    )
    meshcat.SetObject(
        "pouring/source/bottom",
        Cylinder(radius, bottom_thickness),
        Rgba(0.35, 0.62, 0.83, 0.45),
    )

    chord = 2.0 * radius * np.sin(np.pi / segments)
    panel = Box(chord, SOURCE_WALL_THICKNESS, SOURCE_HEIGHT)

    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        X_BC = RigidTransform(
            RollPitchYaw(0.0, 0.0, angle + np.pi / 2),
            [x, y, SOURCE_HEIGHT / 2],
        )
        name = f"wall_{i}"
        SOURCE_VISUAL_TRANSFORMS[name] = X_BC
        meshcat.SetObject(
            f"pouring/source/{name}",
            panel,
            Rgba(0.35, 0.62, 0.83, 0.45),
        )


def add_receiver_visual(meshcat):
    add_hollow_vessel_visual(
        meshcat,
        "pouring/receiver",
        RECEIVER_INNER_RADIUS + RECEIVER_WALL_THICKNESS / 2,
        RECEIVER_HEIGHT,
        RECEIVER_WALL_THICKNESS,
        Rgba(0.30, 0.75, 0.48, 0.55),
    )

    # Translate receiver's local bottom-origin geometry into world coordinates.
    X_WR = RigidTransform(RECEIVER_CENTER_W)
    for component in ["bottom"] + [f"wall_{i}" for i in range(20)]:
        path = f"pouring/receiver/{component}"
        meshcat.SetTransform(path, X_WR @ RECEIVER_VISUAL_TRANSFORMS[component])


RECEIVER_VISUAL_TRANSFORMS = {}


def create_receiver_visual(meshcat):
    """Create the receiver and retain its local component transforms."""
    radius = RECEIVER_INNER_RADIUS + RECEIVER_WALL_THICKNESS / 2
    segments = 20
    bottom_thickness = 0.004

    RECEIVER_VISUAL_TRANSFORMS["bottom"] = RigidTransform(
        [0.0, 0.0, bottom_thickness / 2]
    )
    meshcat.SetObject(
        "pouring/receiver/bottom",
        Cylinder(radius, bottom_thickness),
        Rgba(0.30, 0.75, 0.48, 0.55),
    )

    chord = 2.0 * radius * np.sin(np.pi / segments)
    panel = Box(chord, RECEIVER_WALL_THICKNESS, RECEIVER_HEIGHT)

    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        X_RC = RigidTransform(
            RollPitchYaw(0.0, 0.0, angle + np.pi / 2),
            [x, y, RECEIVER_HEIGHT / 2],
        )
        name = f"wall_{i}"
        RECEIVER_VISUAL_TRANSFORMS[name] = X_RC
        meshcat.SetObject(
            f"pouring/receiver/{name}",
            panel,
            Rgba(0.30, 0.75, 0.48, 0.55),
        )

    X_WR = RigidTransform(RECEIVER_CENTER_W)
    for name, X_RC in RECEIVER_VISUAL_TRANSFORMS.items():
        meshcat.SetTransform(f"pouring/receiver/{name}", X_WR @ X_RC)


# ------------------------------ Liquid particles ----------------------------

def initialize_particles():
    """Create a regular grid of particles inside the source vessel."""
    positions = []

    spacing = PARTICLE_SPACING
    radial_limit = SOURCE_INNER_RADIUS - PARTICLE_RADIUS - 0.5 * spacing

    z_values = np.arange(
        2.0 * spacing,
        SOURCE_HEIGHT - 1.5 * spacing,
        spacing,
    )

    xy_values = np.arange(
        -radial_limit,
        radial_limit + 0.5 * spacing,
        spacing,
    )

    for z in z_values:
        for x in xy_values:
            for y in xy_values:
                if x * x + y * y <= radial_limit**2:
                    positions.append([x, y, z])

    return np.asarray(positions, dtype=float)


def lowest_rim_height(X_WB):
    """World height of the lowest point on the tilted vessel rim."""
    angles = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    rim_points_B = np.column_stack(
        [
            SOURCE_INNER_RADIUS * np.cos(angles),
            SOURCE_INNER_RADIUS * np.sin(angles),
            np.full_like(angles, SOURCE_HEIGHT),
        ]
    )

    rim_points_W = np.array([X_WB @ p for p in rim_points_B])
    return float(np.min(rim_points_W[:, 2]))


def update_particles(
    X_WB,
    angular_velocity_W,
    attached_B,
    flying_W,
    flying_velocity_W,
    collected,
    spilled,
    initial_count,
    dt,
):
    """Approximate draining from the low side, then propagate ballistic particles."""
    attached_B = np.asarray(attached_B, dtype=float).reshape((-1, 3))
    flying_W = np.asarray(flying_W, dtype=float).reshape((-1, 3))
    flying_velocity_W = np.asarray(
        flying_velocity_W, dtype=float
    ).reshape((-1, 3))

    attached_W = np.asarray(
        [X_WB @ p for p in attached_B], dtype=float
    ).reshape((-1, 3))

    if len(attached_B) and initial_count > 0:
        R_WB = X_WB.rotation().matrix()

        # Gravity expressed in vessel coordinates identifies the low side.
        gravity_B = R_WB.T @ np.array([0.0, 0.0, -1.0])
        low_direction_B = gravity_B[:2]
        low_direction_norm = np.linalg.norm(low_direction_B)

        if low_direction_norm > 1e-10:
            low_direction_B /= low_direction_norm

            # Approximate a horizontal free surface. Its height decreases
            # in proportion to the remaining particle count.
            initial_surface_height = SOURCE_HEIGHT - 2.0 * PARTICLE_SPACING
            surface_height = (
                initial_surface_height * len(attached_B) / initial_count
            )
            free_surface_z = SOURCE_PIVOT_W[2] + surface_height
            rim_z = lowest_rim_height(X_WB)

            # Only drain when the low rim intersects the estimated surface.
            overflow = rim_z <= free_surface_z

            # Restrict release to the low half of the vessel and particles
            # near the current estimated free surface.
            low_side_coordinate = attached_B[:, :2] @ low_direction_B
            near_low_side = low_side_coordinate > 0.0
            near_surface = attached_B[:, 2] >= (
                surface_height - 2.0 * PARTICLE_SPACING
            )

            eligible = np.flatnonzero(
                near_low_side & near_surface
            ) if overflow else np.empty(0, dtype=int)

            # A small artificial flow-rate limit prevents bulk release.
            max_release_per_step = 2

            if len(eligible) > max_release_per_step:
                # Prefer particles closest to the low side and upper surface.
                scores = (
                    low_side_coordinate[eligible]
                    + 0.25 * attached_B[eligible, 2]
                )
                selected = eligible[
                    np.argsort(scores)[-max_release_per_step:]
                ]
            else:
                selected = eligible

            release_mask = np.zeros(len(attached_B), dtype=bool)
            release_mask[selected] = True

            newly_released = attached_W[release_mask]

            if len(newly_released):
                # Released particles inherit the vessel's instantaneous
                # rigid-body velocity: v = omega x (p - pivot).
                radii_W = newly_released - SOURCE_PIVOT_W
                new_velocities_W = np.cross(
                    np.broadcast_to(
                        angular_velocity_W, radii_W.shape
                    ),
                    radii_W,
                )

                flying_W = np.vstack([flying_W, newly_released])
                flying_velocity_W = np.vstack(
                    [flying_velocity_W, new_velocities_W]
                )

            attached_B = attached_B[~release_mask]

    # Ballistic propagation under gravity.
    if len(flying_W):
        flying_velocity_W[:, 2] -= GRAVITY * dt
        flying_W += flying_velocity_W * dt

        receiver_xy = flying_W[:, :2] - RECEIVER_CENTER_W[:2]
        radial_distance = np.linalg.norm(receiver_xy, axis=1)

        # Existing simplified floor-based collection test.
        crossed_floor = flying_W[:, 2] <= 0.0
        inside_receiver = radial_distance <= (
            RECEIVER_INNER_RADIUS - PARTICLE_RADIUS
        )

        captured_now = crossed_floor & inside_receiver
        spilled_now = crossed_floor & ~inside_receiver

        collected += int(np.count_nonzero(captured_now))
        spilled += int(np.count_nonzero(spilled_now))

        keep = ~(captured_now | spilled_now)
        flying_W = flying_W[keep]
        flying_velocity_W = flying_velocity_W[keep]

    return (
        attached_B,
        flying_W,
        flying_velocity_W,
        collected,
        spilled,
    )


# -------------------------------- Simulation ---------------------------------

def run_simulation(*, visualize=True, duration=SIMULATION_DURATION, dt=TIME_STEP):
    plant, plant_context, source_body, joint = build_source_plant()

    attached_B = initialize_particles()
    initial_count = len(attached_B)

    flying_W = np.empty((0, 3))
    flying_velocity_W = np.empty((0, 3))
    collected = 0
    spilled = 0

    meshcat = Meshcat() if visualize else None

    if meshcat:
        meshcat.Delete()
        add_source_visual(meshcat)
        create_receiver_visual(meshcat)

        meshcat.SetObject(
            "pouring/ground",
            Box(1.0, 0.7, 0.02),
            Rgba(0.75, 0.75, 0.75, 1.0),
        )
        meshcat.SetTransform(
            "pouring/ground",
            RigidTransform([0.15, 0.0, -0.015]),
        )

        # Draw particle spheres once; update their transforms during the run.
        for i in range(initial_count):
            meshcat.SetObject(
                f"pouring/particles/{i}",
                Sphere(PARTICLE_RADIUS),
                Rgba(0.05, 0.35, 0.95, 0.9),
            )

        print("Meshcat URL:", meshcat.web_url())

    records = []
    steps = int(np.ceil(duration / dt))

    trajectory_recorder = TrajectoryRecorder()
    for step in range(steps + 1):
        t = min(step * dt, duration)

        X_WB, angle, angular_velocity_W = source_pose_at_time(
            plant, plant_context, source_body, joint, t
        )

        (
            attached_B,
            flying_W,
            flying_velocity_W,
            collected,
            spilled,
        ) = update_particles(
            X_WB,
            angular_velocity_W,
            attached_B,
            flying_W,
            flying_velocity_W,
            collected,
            spilled,
            initial_count,
            dt if step > 0 else 0.0,
        )
        trajectory_recorder.record(
            time_s=t,
            body="source",
            pose=X_WB,
        )
        attached_W = np.array([X_WB @ p for p in attached_B])
        all_positions = (
            np.vstack([
                np.asarray(attached_W).reshape((-1, 3)),
                np.asarray(flying_W).reshape((-1, 3)),
            ])
            if len(attached_W) or len(flying_W)
            else np.empty((0, 3))
        )

        if meshcat:
            update_source_visual(meshcat, X_WB)

            # Clear particle transforms from the previous frame.
            meshcat.Delete("pouring/particles")
            for i, p in enumerate(all_positions):
                meshcat.SetObject(
                    f"pouring/particles/{i}",
                    Sphere(PARTICLE_RADIUS),
                    Rgba(0.05, 0.35, 0.95, 0.9),
                )
                meshcat.SetTransform(
                    f"pouring/particles/{i}",
                    RigidTransform(p),
                )

            time.sleep(0.005)

        records.append(
            {
                "time_s": t,
                "tilt_deg": np.degrees(angle),
                "attached": len(attached_B),
                "airborne": len(flying_W),
                "collected": collected,
                "spilled": spilled,
            }
        )

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # Each particle represents approximately one grid cell of fluid.
    particle_volume = PARTICLE_SPACING**3
    collected_volume = collected * particle_volume
    spilled_volume = spilled * particle_volume

    print("\nSimulation complete")
    print(f"Initial particles: {initial_count}")
    print(f"Remaining in source: {len(attached_B)}")
    print(f"Collected particles: {collected}")
    print(f"Spilled particles: {spilled}")
    print(f"Still airborne: {len(flying_W)}")
    print(f"Estimated collected volume: {collected_volume * 1e6:.2f} mL")
    print(f"Estimated spilled volume: {spilled_volume * 1e6:.2f} mL")
    print(f"Particle history: {OUTPUT_CSV}")

    if meshcat:
        input("Press Enter to close the Meshcat simulation.\n")

    trajectory_path = trajectory_recorder.save(
        "drake_vessel_trajectory.csv"
    )

    print(f"Drake trajectory exported: {trajectory_path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--duration", type=float, default=SIMULATION_DURATION)
    parser.add_argument("--dt", type=float, default=TIME_STEP)
    args = parser.parse_args()

    if args.duration <= 0 or args.dt <= 0:
        parser.error("--duration and --dt must be positive")

    run_simulation(
        visualize=not args.no_vis,
        duration=args.duration,
        dt=args.dt,
    )


if __name__ == "__main__":
    main()
