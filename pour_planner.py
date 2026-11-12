
"""Plan and execute pouring motions for a grasped vessel."""

import numpy as np

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)

from grasp_planner import (
    GRASP_POINT_G,
    _solve_ik,
)

from simple_pour_trajectory import (
    VESSELS,
    GRAVITY,
    compute_pour_trajectory,
    render_pour_trajectory,
    target_opening_error,
)


# --------------------------------------------------
# POURING PARAMETERS
# --------------------------------------------------

BEAKER_CENTER_XY = np.array([0.72, 0.14])
BEAKER_OPENING_Z = 0.29
BEAKER_RADIUS = 0.04

# Height of the vessel mouth during pouring.
POUR_MOUTH_Z = 0.43

# Start with a partially filled vessel.
FILL_FRACTION = 0.75

# Maximum tilt from upright.
MAX_TILT_DEG = 110.0

TRANSPORT_STEPS = 20
TILT_STEPS = 30

TRANSPORT_DURATION = 4.0
TILT_DURATION = 5.0
POUR_HOLD_DURATION = 3.0
RETURN_DURATION = 4.0

CONTROL_DT = 0.05


# --------------------------------------------------
# VESSEL AND GRIPPER GEOMETRY
# --------------------------------------------------

def get_gripper_pose(plant, plant_context):

    wsg = plant.GetModelInstanceByName("wsg")

    gripper_frame = plant.GetFrameByName(
        "body", wsg
    )

    return plant.CalcRelativeTransform(
        plant_context,
        plant.world_frame(),
        gripper_frame,
    )


def get_vessel_pose(
    plant,
    plant_context,
    vessel_body,
):

    return plant.EvalBodyPoseInWorld(
        plant_context,
        vessel_body,
    )


def vessel_pose_from_mouth(
    mouth_position,
    rotation,
    vessel_spec,
):
    """Construct a vessel pose from its mouth center."""

    mouth_V = np.array([
        0.0,
        0.0,
        vessel_spec.height,
    ])

    body_position = (
        np.asarray(mouth_position)
        - rotation.matrix() @ mouth_V
    )

    return RigidTransform(
        rotation,
        body_position,
    )


# --------------------------------------------------
# AIM THE LIQUID STREAM
# --------------------------------------------------

def compute_aimed_vessel_pose(
    rotation,
    vessel_spec,
    mouth_z,
    fill_fraction,
):
    """
    Position the vessel so its predicted ballistic
    stream intersects the beaker opening.

    This assumes a stationary vessel at each
    planning waypoint.
    """

    mouth_position = np.array([
        BEAKER_CENTER_XY[0],
        BEAKER_CENTER_XY[1],
        mouth_z,
    ])

    X_WV = vessel_pose_from_mouth(
        mouth_position,
        rotation,
        vessel_spec,
    )

    result = compute_pour_trajectory(
        X_WV,
        vessel_spec,
        fill_fraction,
    )

    if result.active:

        p = result.mouth_W
        v = result.launch_velocity_W

        dz = p[2] - BEAKER_OPENING_Z

        if dz > 0.0:

            # Time until the jet reaches the
            # horizontal beaker opening plane.
            t_hit = (
                v[2]
                + np.sqrt(
                    v[2] ** 2
                    + 2.0 * GRAVITY * dz
                )
            ) / GRAVITY

            predicted_hit_xy = (
                p[:2] + v[:2] * t_hit
            )

            correction_xy = (
                BEAKER_CENTER_XY
                - predicted_hit_xy
            )

            # Shift the source mouth to compensate
            # for the horizontal jet displacement.
            mouth_position[:2] += correction_xy

            X_WV = vessel_pose_from_mouth(
                mouth_position,
                rotation,
                vessel_spec,
            )

    return X_WV


# --------------------------------------------------
# INVERSE KINEMATICS
# --------------------------------------------------

def solve_vessel_pose_ik(
    plant,
    plant_context,
    X_WV_target,
    X_GV,
    seed_positions,
):
    """
    Convert a desired vessel pose into a desired
    gripper pose, then solve robot IK.
    """

    X_WG_target = (
        X_WV_target @ X_GV.inverse()
    )

    grasp_target = X_WG_target.multiply(
        GRASP_POINT_G
    )

    return _solve_ik(
        plant=plant,
        plant_context=plant_context,
        target_position=grasp_target,
        target_rotation=X_WG_target.rotation(),
        orientation_tolerance_deg=3.0,
        seed_positions=seed_positions,
        attempts=12,
        perturbation=0.10,
    )


def plan_pour_motion(
    plant,
    plant_context,
    vessel_body,
    vessel_name,
):
    """
    Plan transport and tilt trajectories without
    modifying the live simulation state.
    """

    vessel_spec = VESSELS[vessel_name]

    X_WG_start = get_gripper_pose(
        plant,
        plant_context,
    )

    X_WV_start = get_vessel_pose(
        plant,
        plant_context,
        vessel_body,
    )

    # Measure the actual grasp transform.
    X_GV = (
        X_WG_start.inverse() @ X_WV_start
    )

    R_start = X_WV_start.rotation()

    mouth_V = np.array([
        0.0,
        0.0,
        vessel_spec.height,
    ])

    mouth_start = X_WV_start.multiply(
        mouth_V
    )

    # Avoid lowering the mouth during transport.
    mouth_z = max(
        POUR_MOUTH_Z,
        float(mouth_start[2]),
    )

    # Keep the vessel approximately upright
    # while moving toward the receiver.
    mouth_goal = np.array([
        BEAKER_CENTER_XY[0],
        BEAKER_CENTER_XY[1],
        mouth_z,
    ])

    iiwa = plant.GetModelInstanceByName(
        "iiwa"
    )

    q_start = plant.GetPositions(
        plant_context,
        iiwa,
    ).copy()

    seed_full = plant.GetPositions(
        plant_context
    ).copy()

    transport_waypoints = []
    tilt_waypoints = []

    # ------------------------------------------
    # PLAN TRANSPORT
    # ------------------------------------------

    print("\nPlanning transport to beaker...")

    for step in range(
        1, TRANSPORT_STEPS + 1
    ):

        s = step / TRANSPORT_STEPS

        mouth_target = (
            (1.0 - s) * mouth_start
            + s * mouth_goal
        )

        X_WV_target = vessel_pose_from_mouth(
            mouth_target,
            R_start,
            vessel_spec,
        )

        q_next, next_seed = solve_vessel_pose_ik(
            plant,
            plant_context,
            X_WV_target,
            X_GV,
            seed_full,
        )

        if q_next is None:

            print(
                f"Transport IK failed at step "
                f"{step}/{TRANSPORT_STEPS}."
            )

            return None

        transport_waypoints.append(
            q_next.copy()
        )

        seed_full = next_seed.copy()

    print("Transport IK succeeded.")

    # ------------------------------------------
    # PLAN TILT
    # ------------------------------------------

    print("\nPlanning pouring tilt...")

    for step in range(
        1, TILT_STEPS + 1
    ):

        theta_deg = (
            MAX_TILT_DEG
            * step / TILT_STEPS
        )

        theta_rad = np.deg2rad(
            theta_deg
        )

        # Rotate around the world y-axis.
        # This tips the vessel toward +x.
        R_tilt = (
            RotationMatrix.MakeYRotation(
                theta_rad
            ) @ R_start
        )

        X_WV_target = compute_aimed_vessel_pose(
            R_tilt,
            vessel_spec,
            mouth_z,
            FILL_FRACTION,
        )

        q_next, next_seed = solve_vessel_pose_ik(
            plant,
            plant_context,
            X_WV_target,
            X_GV,
            seed_full,
        )

        if q_next is None:

            print(
                f"Pour IK failed at "
                f"{theta_deg:.1f} degrees."
            )

            return None

        tilt_waypoints.append(
            q_next.copy()
        )

        seed_full = next_seed.copy()

        if step % 5 == 0:

            print(
                f"Planned tilt: "
                f"{theta_deg:.1f} degrees"
            )

    print("Pour IK succeeded.")

    return {
        "q_start": q_start,
        "X_GV": X_GV,
        "transport": transport_waypoints,
        "tilt": tilt_waypoints,
    }


# --------------------------------------------------
# LIVE POUR VISUALIZATION
# --------------------------------------------------

def update_liquid_stream(
    meshcat,
    plant,
    plant_context,
    vessel_body,
    vessel_name,
    fill_fraction,
):

    X_WV = get_vessel_pose(
        plant,
        plant_context,
        vessel_body,
    )

    V_WV = plant.EvalBodySpatialVelocityInWorld(
        plant_context,
        vessel_body,
    )

    result = compute_pour_trajectory(
        X_WV,
        VESSELS[vessel_name],
        fill_fraction,
        linear_velocity_W=V_WV.translational(),
        angular_velocity_W=V_WV.rotational(),
    )

    render_pour_trajectory(
        meshcat,
        "pouring/live_stream",
        result,
    )

    return result


# --------------------------------------------------
# EXECUTE JOINT WAYPOINTS
# --------------------------------------------------

def execute_waypoints(
    simulator,
    station,
    station_context,
    plant,
    plant_context,
    meshcat,
    vessel_body,
    vessel_name,
    X_GV,
    waypoints,
    duration,
):

    if not waypoints:
        return True

    iiwa = plant.GetModelInstanceByName(
        "iiwa"
    )

    q_previous = plant.GetPositions(
        plant_context,
        iiwa,
    ).copy()

    segment_duration = (
        duration / len(waypoints)
    )

    steps_per_segment = max(
        1,
        int(np.ceil(
            segment_duration / CONTROL_DT
        )),
    )

    for q_next in waypoints:

        start_time = (
            simulator.get_context().get_time()
        )

        for i in range(
            1, steps_per_segment + 1
        ):

            s = i / steps_per_segment

            q_command = (
                q_previous
                + s * (q_next - q_previous)
            )

            station.GetInputPort(
                "iiwa.position"
            ).FixValue(
                station_context,
                q_command,
            )

            simulator.AdvanceTo(
                start_time
                + s * segment_duration
            )

            # Check that the vessel still follows
            # the gripper rather than slipping.
            X_WG = get_gripper_pose(
                plant,
                plant_context,
            )

            X_WV = get_vessel_pose(
                plant,
                plant_context,
                vessel_body,
            )

            X_GV_actual = (
                X_WG.inverse() @ X_WV
            )

            grasp_error = np.linalg.norm(
                X_GV_actual.translation()
                - X_GV.translation()
            )

            R_error = (
                X_GV.rotation().matrix().T
                @ X_GV_actual.rotation().matrix()
            )

            angular_error = np.arccos(
                np.clip(
                    (np.trace(R_error) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )

            if (
                grasp_error > 0.025
                or angular_error > np.deg2rad(20)
            ):

                print(
                    "\nWARNING: Vessel slipped "
                    "relative to the gripper."
                )

                # Stop commanding further motion.
                q_actual = plant.GetPositions(
                    plant_context, iiwa
                ).copy()

                station.GetInputPort(
                    "iiwa.position"
                ).FixValue(
                    station_context, q_actual
                )

                meshcat.Delete(
                    "pouring/live_stream"
                )

                return False

            update_liquid_stream(
                meshcat,
                plant,
                plant_context,
                vessel_body,
                vessel_name,
                FILL_FRACTION,
            )

        q_previous = q_next.copy()

    return True


# --------------------------------------------------
# COMPLETE POURING SEQUENCE
# --------------------------------------------------

def execute_pour(
    simulator,
    station,
    station_context,
    plant,
    plant_context,
    meshcat,
    vessel_body,
    vessel_name,
):

    print("\nPHASE 6: Planning pour...")

    plan = plan_pour_motion(
        plant,
        plant_context,
        vessel_body,
        vessel_name,
    )

    if plan is None:

        print("Pour skipped: IK planning failed.")

        return False

    # Keep the gripper closed throughout pouring.
    station.GetInputPort(
        "wsg.position"
    ).FixValue(
        station_context,
        [0.0],
    )

    def run_motion(waypoints, duration):

        return execute_waypoints(
            simulator=simulator,
            station=station,
            station_context=station_context,
            plant=plant,
            plant_context=plant_context,
            meshcat=meshcat,
            vessel_body=vessel_body,
            vessel_name=vessel_name,
            X_GV=plan["X_GV"],
            waypoints=waypoints,
            duration=duration,
        )

    # ------------------------------------------
    # TRANSPORT
    # ------------------------------------------

    print("\nPHASE 7: Moving above beaker...")

    if not run_motion(
        plan["transport"],
        TRANSPORT_DURATION,
    ):

        return False

    # ------------------------------------------
    # TILT
    # ------------------------------------------

    print("\nPHASE 8: Tilting vessel...")

    if not run_motion(
        plan["tilt"],
        TILT_DURATION,
    ):

        return False

    # ------------------------------------------
    # HOLD POURING POSE
    # ------------------------------------------

    print("\nPHASE 9: Pouring...")

    if not run_motion(
        [plan["tilt"][-1]],
        POUR_HOLD_DURATION,
    ):

        return False

    result = update_liquid_stream(
        meshcat,
        plant,
        plant_context,
        vessel_body,
        vessel_name,
        FILL_FRACTION,
    )

    hit, miss_distance = target_opening_error(
        result,
        BEAKER_CENTER_XY,
        BEAKER_OPENING_Z,
        BEAKER_RADIUS,
    )

    print("\nPOUR RESULTS")

    print(
        f"Vessel tilt: "
        f"{result.tilt_deg:.1f} degrees"
    )

    print(
        f"Liquid stream active: "
        f"{result.active}"
    )

    print(
        f"Stream enters beaker: {hit}"
    )

    print(
        f"Distance from opening center: "
        f"{miss_distance:.4f} m"
    )

    # ------------------------------------------
    # RETURN UPRIGHT
    # ------------------------------------------

    print("\nPHASE 10: Returning upright...")

    return_waypoints = (
        list(reversed(plan["tilt"][:-1]))
        + [plan["transport"][-1]]
    )

    if not run_motion(
        return_waypoints,
        RETURN_DURATION,
    ):

        return False

    meshcat.Delete(
        "pouring/live_stream"
    )

    print("\nPouring sequence complete!")

    return True
