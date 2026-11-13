
"""Jacobian-based pouring controller.

Supports:
- Test tube
- Erlenmeyer flask
- Boiling flask

Automatically adjusts pouring clearance based
on vessel geometry.

Uses differential inverse kinematics for:
1. Raising vessel for clearance.
2. Transporting to receiving beaker.
3. Tilting and aiming liquid stream.
4. Holding pouring pose.
5. Returning upright.

No SciPy dependency.
"""

import numpy as np

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    JacobianWrtVariable,
)

from simple_pour_trajectory import (
    VESSELS,
    GRAVITY,
    compute_pour_trajectory,
    render_pour_trajectory,
    target_opening_error,
)


# ==================================================
# RECEIVING CONTAINER GEOMETRY
# ==================================================

BEAKER_CENTER_XY = np.array([
    0.72,
    0.14,
])

BEAKER_OPENING_Z = 0.29
BEAKER_RADIUS = 0.04


# ==================================================
# POURING PARAMETERS
# ==================================================

# Minimum mouth height used for the test tube.
POUR_MOUTH_Z = 0.46

# Clearance between source vessel bottom
# and receiving container rim during transport.
VESSEL_CLEARANCE = 0.04

# Initial fill fraction.
FILL_FRACTION = 0.75

# Default maximum tilt.
MAX_TILT_DEG = 110.0

# Geometry-specific maximum tilt settings.
#
# Start with the same angle for all geometries.
# These can be tuned independently later.
VESSEL_TILT_ANGLES = {
    "test_tube": 110.0,
    "erlenmeyer": 110.0,
    "boiling_flask": 110.0,
}

# Geometry-specific tilt durations.
#
# Larger vessels are tilted more slowly to
# reduce the likelihood of slipping.
VESSEL_TILT_DURATIONS = {
    "test_tube": 8.0,
    "erlenmeyer": 10.0,
    "boiling_flask": 10.0,
}


# ==================================================
# MOTION DURATIONS
# ==================================================

CLEARANCE_DURATION = 3.0
TRANSPORT_DURATION = 6.0
TILT_DURATION = 8.0
POUR_HOLD_DURATION = 3.0
RETURN_DURATION = 8.0

CONTROL_DT = 0.02

SETTLE_DURATION = 2.0


# ==================================================
# DIFFERENTIAL IK PARAMETERS
# ==================================================

POSITION_GAIN = 5.0
ORIENTATION_GAIN = 4.0

DAMPING = 0.035

JOINT_SPEED_LIMIT = 1.0
LINEAR_SPEED_LIMIT = 0.15
ANGULAR_SPEED_LIMIT = 0.65

POSITION_TOLERANCE = 0.025

ORIENTATION_TOLERANCE = (
    np.deg2rad(8.0)
)


# ==================================================
# GRASP MONITORING PARAMETERS
# ==================================================

GRASP_POSITION_TOLERANCE = 0.025

GRASP_ORIENTATION_TOLERANCE = (
    np.deg2rad(20.0)
)


# ==================================================
# GENERAL HELPERS
# ==================================================

def smoothstep(s):
    """Smooth interpolation from zero to one."""

    s = np.clip(
        s,
        0.0,
        1.0,
    )

    return (
        3.0 * s**2
        - 2.0 * s**3
    )


def limit_norm(
    vector,
    max_norm,
):
    """Limit magnitude while preserving direction."""

    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm > max_norm:

        return (
            vector * max_norm / norm
        )

    return vector


def rotation_error_world(
    R_desired,
    R_actual,
):
    """
    Calculate orientation error as a world-frame
    axis-angle rotation vector.

    Uses Drake instead of SciPy.
    """

    R_error = RotationMatrix(
        R_desired.matrix()
        @ R_actual.matrix().T
    )

    angle_axis = (
        R_error.ToAngleAxis()
    )

    return (
        angle_axis.angle()
        * angle_axis.axis()
    )


def get_arm_indices(
    plant,
    iiwa,
):
    """Find iiwa position and velocity indices."""

    position_indices = []
    velocity_indices = []

    for joint_index in plant.GetJointIndices(
        iiwa
    ):

        joint = plant.get_joint(
            joint_index
        )

        position_indices.extend(
            range(
                joint.position_start(),
                joint.position_start()
                + joint.num_positions(),
            )
        )

        velocity_indices.extend(
            range(
                joint.velocity_start(),
                joint.velocity_start()
                + joint.num_velocities(),
            )
        )

    return (
        np.array(
            position_indices,
            dtype=int,
        ),
        np.array(
            velocity_indices,
            dtype=int,
        ),
    )


# ==================================================
# VESSEL AND GRIPPER GEOMETRY
# ==================================================

def get_gripper_pose(
    plant,
    plant_context,
):
    """Return actual world-frame gripper pose."""

    wsg = plant.GetModelInstanceByName(
        "wsg"
    )

    gripper_frame = (
        plant.GetFrameByName(
            "body",
            wsg,
        )
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
    """Return actual world-frame vessel pose."""

    return plant.EvalBodyPoseInWorld(
        plant_context,
        vessel_body,
    )


def vessel_pose_from_mouth(
    mouth_position,
    rotation,
    vessel_spec,
):
    """
    Calculate vessel body pose from desired
    mouth center and orientation.
    """

    # Vessel mouth center in the vessel frame.
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


def calculate_mouth_height(
    vessel_spec,
    mouth_start,
):
    """
    Calculate geometry-dependent mouth height.

    Ensures the nominal upright vessel bottom
    clears the receiving container rim.
    """

    minimum_mouth_z = (
        BEAKER_OPENING_Z
        + vessel_spec.height
        + VESSEL_CLEARANCE
    )

    mouth_z = max(
        POUR_MOUTH_Z,
        float(mouth_start[2]),
        minimum_mouth_z,
    )

    return mouth_z


# ==================================================
# LIQUID AIMING
# ==================================================

def compute_aimed_vessel_pose(
    rotation,
    vessel_spec,
    mouth_z,
    fill_fraction,
):
    """
    Calculate a vessel pose whose predicted
    ballistic stream intersects the beaker.

    The initial calculation assumes the vessel
    is stationary.
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

    # No aiming adjustment is required
    # before liquid release begins.
    if result.active:

        p = result.mouth_W

        v = result.launch_velocity_W

        dz = (
            p[2]
            - BEAKER_OPENING_Z
        )

        if dz > 0.0:

            # Ballistic time to beaker opening.
            t_hit = (
                v[2]
                + np.sqrt(
                    v[2]**2
                    + 2.0 * GRAVITY * dz
                )
            ) / GRAVITY

            # Predicted horizontal landing point.
            predicted_hit_xy = (
                p[:2]
                + v[:2] * t_hit
            )

            # Shift mouth position to compensate.
            correction_xy = (
                BEAKER_CENTER_XY
                - predicted_hit_xy
            )

            mouth_position[:2] += (
                correction_xy
            )

            X_WV = vessel_pose_from_mouth(
                mouth_position,
                rotation,
                vessel_spec,
            )

    return X_WV


# ==================================================
# LIVE LIQUID STREAM
# ==================================================

def update_liquid_stream(
    meshcat,
    plant,
    plant_context,
    vessel_body,
    vessel_name,
    fill_fraction,
):
    """
    Update the ballistic liquid trajectory using
    actual vessel pose and spatial velocity.
    """

    X_WV = get_vessel_pose(
        plant,
        plant_context,
        vessel_body,
    )

    V_WV = (
        plant.EvalBodySpatialVelocityInWorld(
            plant_context,
            vessel_body,
        )
    )

    result = compute_pour_trajectory(
        X_WV,
        VESSELS[vessel_name],
        fill_fraction,
        linear_velocity_W=(
            V_WV.translational()
        ),
        angular_velocity_W=(
            V_WV.rotational()
        ),
    )

    render_pour_trajectory(
        meshcat,
        "pouring/live_stream",
        result,
    )

    return result


# ==================================================
# GRASP STABILITY
# ==================================================

def check_grasp(
    plant,
    plant_context,
    vessel_body,
    X_GV_reference,
):
    """
    Check whether the vessel maintains its
    original pose relative to the gripper.
    """

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
        X_WG.inverse()
        @ X_WV
    )

    position_error = np.linalg.norm(
        X_GV_actual.translation()
        - X_GV_reference.translation()
    )

    orientation_error = np.linalg.norm(
        rotation_error_world(
            X_GV_reference.rotation(),
            X_GV_actual.rotation(),
        )
    )

    grasp_ok = (
        position_error
        < GRASP_POSITION_TOLERANCE
        and orientation_error
        < GRASP_ORIENTATION_TOLERANCE
    )

    return (
        grasp_ok,
        position_error,
        orientation_error,
    )


# ==================================================
# DIFFERENTIAL IK CARTESIAN CONTROLLER
# ==================================================

def command_cartesian_motion(
    simulator,
    station,
    station_context,
    plant,
    plant_context,
    meshcat,
    vessel_body,
    vessel_name,
    X_GV,
    pose_function,
    duration,
    phase_name,
    settle_duration=SETTLE_DURATION,
):
    """
    Follow a desired vessel pose trajectory using
    differential inverse kinematics.

    pose_function(s) returns the desired vessel
    pose for normalized time s in [0, 1].

    Desired vessel pose is converted into the
    corresponding desired gripper pose.
    """

    iiwa = plant.GetModelInstanceByName(
        "iiwa"
    )

    wsg = plant.GetModelInstanceByName(
        "wsg"
    )

    gripper_frame = (
        plant.GetFrameByName(
            "body",
            wsg,
        )
    )

    world = plant.world_frame()

    # Arm indices within full plant.
    (
        position_indices,
        velocity_indices,
    ) = get_arm_indices(
        plant,
        iiwa,
    )

    joint_lower = (
        plant.GetPositionLowerLimits()[
            position_indices
        ]
    )

    joint_upper = (
        plant.GetPositionUpperLimits()[
            position_indices
        ]
    )

    position_port = (
        station.GetInputPort(
            "iiwa.position"
        )
    )

    # Keep gripper closed.
    station.GetInputPort(
        "wsg.position"
    ).FixValue(
        station_context,
        [0.0],
    )

    # Control loop timing.
    steps = max(
        1,
        int(
            np.ceil(
                duration / CONTROL_DT
            )
        ),
    )

    dt = duration / steps

    settle_steps = int(
        np.ceil(
            settle_duration / dt
        )
    )

    start_time = (
        simulator.get_context().get_time()
    )

    X_VG = X_GV.inverse()

    def desired_gripper_pose(s):

        X_WV_desired = (
            pose_function(s)
        )

        return (
            X_WV_desired
            @ X_VG
        )

    print(f"\n{phase_name}")

    X_WG_previous = (
        desired_gripper_pose(0.0)
    )

    last_reported = -1

    # ----------------------------------------------
    # CONTROL LOOP
    # ----------------------------------------------

    for i in range(
        1,
        steps + settle_steps + 1,
    ):

        s = min(
            i / steps,
            1.0,
        )

        X_WG_desired = (
            desired_gripper_pose(s)
        )

        X_WG_actual = get_gripper_pose(
            plant,
            plant_context,
        )

        # ------------------------------------------
        # LINEAR VELOCITY COMMAND
        # ------------------------------------------

        position_error = (
            X_WG_desired.translation()
            - X_WG_actual.translation()
        )

        linear_feedforward = (
            X_WG_desired.translation()
            - X_WG_previous.translation()
        ) / dt

        v_desired = (
            linear_feedforward
            + POSITION_GAIN
            * position_error
        )

        v_desired = limit_norm(
            v_desired,
            LINEAR_SPEED_LIMIT,
        )

        # ------------------------------------------
        # ANGULAR VELOCITY COMMAND
        # ------------------------------------------

        orientation_error = (
            rotation_error_world(
                X_WG_desired.rotation(),
                X_WG_actual.rotation(),
            )
        )

        angular_feedforward = (
            rotation_error_world(
                X_WG_desired.rotation(),
                X_WG_previous.rotation(),
            )
            / dt
        )

        omega_desired = (
            angular_feedforward
            + ORIENTATION_GAIN
            * orientation_error
        )

        omega_desired = limit_norm(
            omega_desired,
            ANGULAR_SPEED_LIMIT,
        )

        # Drake spatial velocity convention:
        # [angular; translational].
        V_desired = np.concatenate([
            omega_desired,
            v_desired,
        ])

        # ------------------------------------------
        # SPATIAL JACOBIAN
        # ------------------------------------------

        J_full = (
            plant.CalcJacobianSpatialVelocity(
                plant_context,
                JacobianWrtVariable.kV,
                gripper_frame,
                [0.0, 0.0, 0.0],
                world,
                world,
            )
        )

        J = J_full[
            :,
            velocity_indices,
        ]

        # ------------------------------------------
        # DAMPED PSEUDOINVERSE
        # ------------------------------------------

        # qdot = J.T (J J.T + lambda^2 I)^-1 V

        qdot = J.T @ np.linalg.solve(
            J @ J.T
            + DAMPING**2 * np.eye(6),
            V_desired,
        )

        max_speed = np.max(
            np.abs(qdot)
        )

        if max_speed > JOINT_SPEED_LIMIT:

            qdot *= (
                JOINT_SPEED_LIMIT
                / max_speed
            )

        # ------------------------------------------
        # JOINT POSITION COMMAND
        # ------------------------------------------

        q_actual = plant.GetPositions(
            plant_context,
            iiwa,
        ).copy()

        q_command = (
            q_actual
            + dt * qdot
        )

        # Enforce position limits.
        q_command = np.clip(
            q_command,
            joint_lower + 1e-4,
            joint_upper - 1e-4,
        )

        position_port.FixValue(
            station_context,
            q_command,
        )

        simulator.AdvanceTo(
            start_time + i * dt
        )

        # ------------------------------------------
        # GRASP STABILITY CHECK
        # ------------------------------------------

        (
            grasp_ok,
            grasp_position_error,
            grasp_orientation_error,
        ) = check_grasp(
            plant,
            plant_context,
            vessel_body,
            X_GV,
        )

        if not grasp_ok:

            print(
                "\nWARNING: Vessel slipped "
                "relative to gripper."
            )

            print(
                "Vessel:",
                vessel_name,
            )

            print(
                "Grasp position error:",
                round(
                    grasp_position_error,
                    4,
                ),
                "m",
            )

            print(
                "Grasp orientation error:",
                round(
                    np.rad2deg(
                        grasp_orientation_error
                    ),
                    2,
                ),
                "deg",
            )

            # Stop motion at actual configuration.
            position_port.FixValue(
                station_context,
                plant.GetPositions(
                    plant_context,
                    iiwa,
                ).copy(),
            )

            meshcat.Delete(
                "pouring/live_stream"
            )

            return False

        # ------------------------------------------
        # LIQUID STREAM VISUALIZATION
        # ------------------------------------------

        result = update_liquid_stream(
            meshcat,
            plant,
            plant_context,
            vessel_body,
            vessel_name,
            FILL_FRACTION,
        )

        progress = int(
            s * 4
        )

        if progress > last_reported:

            last_reported = progress

            print(
                f"  Progress: {100 * s:.0f}%"
                f" | Tilt: "
                f"{result.tilt_deg:.1f} deg"
                f" | Flow: {result.active}"
            )

        X_WG_previous = (
            X_WG_desired
        )

    # ----------------------------------------------
    # FINAL TRACKING ERROR
    # ----------------------------------------------

    X_WG_final = get_gripper_pose(
        plant,
        plant_context,
    )

    X_WG_goal = (
        desired_gripper_pose(1.0)
    )

    final_position_error = np.linalg.norm(
        X_WG_goal.translation()
        - X_WG_final.translation()
    )

    final_orientation_error = (
        np.linalg.norm(
            rotation_error_world(
                X_WG_goal.rotation(),
                X_WG_final.rotation(),
            )
        )
    )

    print(
        "Final position error:",
        round(
            final_position_error,
            4,
        ),
        "m",
    )

    print(
        "Final orientation error:",
        round(
            np.rad2deg(
                final_orientation_error
            ),
            2,
        ),
        "deg",
    )

    tracking_ok = (
        final_position_error
        < POSITION_TOLERANCE
        and final_orientation_error
        < ORIENTATION_TOLERANCE
    )

    if not tracking_ok:

        print(
            "WARNING: Desired Cartesian "
            "pose was not reached."
        )

        # Print joint margins for debugging.
        q_actual = plant.GetPositions(
            plant_context,
            iiwa,
        ).copy()

        print(
            "Current arm configuration:",
            np.round(
                q_actual,
                3,
            ),
        )

        print(
            "Lower joint limit margins:",
            np.round(
                q_actual - joint_lower,
                3,
            ),
        )

        print(
            "Upper joint limit margins:",
            np.round(
                joint_upper - q_actual,
                3,
            ),
        )

        position_port.FixValue(
            station_context,
            q_actual,
        )

        meshcat.Delete(
            "pouring/live_stream"
        )

        return False

    print(
        "Motion completed successfully."
    )

    return True


# ==================================================
# COMPLETE POURING PIPELINE
# ==================================================

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
    """
    Execute transport, tilt, pour, and recovery
    after successful pickup.
    """

    # ----------------------------------------------
    # GET SELECTED VESSEL GEOMETRY
    # ----------------------------------------------

    vessel_spec = VESSELS[
        vessel_name
    ]

    max_tilt_deg = (
        VESSEL_TILT_ANGLES.get(
            vessel_name,
            MAX_TILT_DEG,
        )
    )

    tilt_duration = (
        VESSEL_TILT_DURATIONS.get(
            vessel_name,
            TILT_DURATION,
        )
    )

    return_duration = max(
        RETURN_DURATION,
        tilt_duration,
    )

    # ----------------------------------------------
    # MEASURE ACTUAL GRASP TRANSFORM
    # ----------------------------------------------

    X_WG_start = get_gripper_pose(
        plant,
        plant_context,
    )

    X_WV_start = get_vessel_pose(
        plant,
        plant_context,
        vessel_body,
    )

    # Measure actual vessel pose relative
    # to the gripper after successful lift.
    X_GV = (
        X_WG_start.inverse()
        @ X_WV_start
    )

    R_start = (
        X_WV_start.rotation()
    )

    mouth_V = np.array([
        0.0,
        0.0,
        vessel_spec.height,
    ])

    mouth_start = (
        X_WV_start.multiply(
            mouth_V
        )
    )

    # ----------------------------------------------
    # GEOMETRY-DEPENDENT CLEARANCE
    # ----------------------------------------------

    mouth_z = calculate_mouth_height(
        vessel_spec,
        mouth_start,
    )

    print("\n" + "=" * 55)
    print("JACOBIAN-BASED POURING")
    print("=" * 55)

    print(
        "Vessel:",
        vessel_name,
    )

    print(
        f"Vessel height: "
        f"{vessel_spec.height:.3f} m"
    )

    print(
        f"Mouth radius: "
        f"{vessel_spec.mouth_radius:.3f} m"
    )

    print(
        "Initial mouth position:",
        np.round(
            mouth_start,
            4,
        ),
    )

    print(
        f"Desired mouth height: "
        f"{mouth_z:.3f} m"
    )

    print(
        f"Commanded tilt rotation: "
        f"{max_tilt_deg:.1f} degrees"
    )

    print(
        f"Tilt duration: "
        f"{tilt_duration:.1f} s"
    )

    print(
        f"Fill fraction: "
        f"{FILL_FRACTION:.0%}"
    )

    # ----------------------------------------------
    # MOTION EXECUTION HELPER
    # ----------------------------------------------

    def run_motion(
        pose_function,
        duration,
        phase_name,
        settle_duration=SETTLE_DURATION,
    ):

        return command_cartesian_motion(
            simulator=simulator,
            station=station,
            station_context=station_context,
            plant=plant,
            plant_context=plant_context,
            meshcat=meshcat,
            vessel_body=vessel_body,
            vessel_name=vessel_name,
            X_GV=X_GV,
            pose_function=pose_function,
            duration=duration,
            phase_name=phase_name,
            settle_duration=settle_duration,
        )

    # ==============================================
    # PHASE 6: RAISE FOR CLEARANCE
    # ==============================================

    mouth_raised = (
        mouth_start.copy()
    )

    mouth_raised[2] = mouth_z

    def clearance_pose(s):

        alpha = smoothstep(s)

        mouth_target = (
            (1.0 - alpha)
            * mouth_start
            + alpha
            * mouth_raised
        )

        return vessel_pose_from_mouth(
            mouth_target,
            R_start,
            vessel_spec,
        )

    if not run_motion(
        clearance_pose,
        CLEARANCE_DURATION,
        "PHASE 6: Raising vessel "
        "for clearance...",
    ):

        return False

    # ==============================================
    # PHASE 7: TRANSPORT ABOVE BEAKER
    # ==============================================

    mouth_goal = np.array([
        BEAKER_CENTER_XY[0],
        BEAKER_CENTER_XY[1],
        mouth_z,
    ])

    def transport_pose(s):

        alpha = smoothstep(s)

        mouth_target = (
            (1.0 - alpha)
            * mouth_raised
            + alpha
            * mouth_goal
        )

        return vessel_pose_from_mouth(
            mouth_target,
            R_start,
            vessel_spec,
        )

    if not run_motion(
        transport_pose,
        TRANSPORT_DURATION,
        "PHASE 7: Transporting "
        "to beaker...",
    ):

        return False

    # ==============================================
    # PHASE 8: TILT AND AIM
    # ==============================================

    def tilt_pose(s):

        alpha = smoothstep(s)

        theta = np.deg2rad(
            max_tilt_deg * alpha
        )

        # Preserve the successful test tube
        # tilt direction toward positive world X.
        R_tilt = (
            RotationMatrix.MakeYRotation(
                theta
            )
            @ R_start
        )

        # Aim using selected vessel geometry.
        return compute_aimed_vessel_pose(
            R_tilt,
            vessel_spec,
            mouth_z,
            FILL_FRACTION,
        )

    if not run_motion(
        tilt_pose,
        tilt_duration,
        "PHASE 8: Tilting vessel...",
    ):

        return False

    # ==============================================
    # PHASE 9: HOLD POURING POSE
    # ==============================================

    X_WV_pour = tilt_pose(
        1.0
    )

    def hold_pose(s):

        return X_WV_pour

    if not run_motion(
        hold_pose,
        POUR_HOLD_DURATION,
        "PHASE 9: Holding pouring pose...",
        settle_duration=0.0,
    ):

        return False

    # ==============================================
    # EVALUATE LIQUID TRAJECTORY
    # ==============================================

    result = update_liquid_stream(
        meshcat,
        plant,
        plant_context,
        vessel_body,
        vessel_name,
        FILL_FRACTION,
    )

    hit, miss_distance = (
        target_opening_error(
            result,
            BEAKER_CENTER_XY,
            BEAKER_OPENING_Z,
            BEAKER_RADIUS,
        )
    )

    print("\n" + "=" * 55)
    print("POUR RESULTS")
    print("=" * 55)

    print(
        "Vessel geometry:",
        vessel_name,
    )

    print(
        f"Actual vessel tilt: "
        f"{result.tilt_deg:.1f} degrees"
    )

    print(
        f"Liquid stream active: "
        f"{result.active}"
    )

    print(
        f"Stream enters beaker: "
        f"{hit}"
    )

    print(
        f"Opening center error: "
        f"{miss_distance:.4f} m"
    )

    print(
        "Actual vessel mouth:",
        np.round(
            result.mouth_W,
            4,
        ),
    )

    print("=" * 55)

    # ==============================================
    # PHASE 10: RETURN UPRIGHT
    # ==============================================

    def return_pose(s):

        return tilt_pose(
            1.0 - s
        )

    if not run_motion(
        return_pose,
        return_duration,
        "PHASE 10: Returning upright...",
    ):

        return False

    meshcat.Delete(
        "pouring/live_stream"
    )

    print(
        "\nPouring sequence complete!"
    )

    if hit:

        print(
            "SUCCESS: Predicted stream entered "
            "the receiving beaker."
        )

    else:

        print(
            "WARNING: Motion completed, but "
            "the predicted stream missed "
            "the receiving beaker."
        )

    return True
