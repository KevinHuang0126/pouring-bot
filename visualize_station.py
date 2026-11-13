
"""KUKA iiwa randomized grasping and pouring workstation.

Complete pipeline:
1. Select source vessel geometry.
2. Select source position.
3. Plan and execute grasp.
4. Close gripper with geometry-dependent force limit.
5. Lift using Wyatt's Jacobian controller.
6. Transport and pour using differential IK.
7. Return vessel upright.

Run:
    python visualize_station.py
"""

from pathlib import Path

import numpy as np

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)

from pydrake.all import (
    DiagramBuilder,
    JacobianWrtVariable,
    Meshcat,
    RigidTransform,
    Simulator,
)

from grasp_planner import (
    GRASP_POINT_G,
    solve_pregrasp_ik,
)

from pour_planner import execute_pour


# ==================================================
# RANDOMIZATION SETTINGS
# ==================================================

# Reproduce the Erlenmeyer flask slipping scenario.
#
# Set RANDOM_SEED = None when ready to return
# to fully randomized testing.

RANDOM_SEED = 252190880282444974816335130055296677874

RANDOMIZE_POSITION = True
RANDOMIZE_VESSEL = True

# Used when randomization is disabled.
FIXED_VESSEL = "test_tube"

FIXED_SOURCE_X = 0.42
FIXED_SOURCE_Y = -0.16

# Restricted workspace for initial testing.
SOURCE_X_RANGE = (0.35, 0.60)
SOURCE_Y_RANGE = (-0.23, 0.10)

MIN_BEAKER_CLEARANCE = 0.20


# ==================================================
# WORKSTATION GEOMETRY
# ==================================================

TABLE_CENTER = (0.60, 0.00, 0.17)
TABLE_TOP = 0.20

HOLDER_TOP = TABLE_TOP + 0.02

BEAKER_X = 0.72
BEAKER_Y = 0.14

PACKAGE_XML = (
    Path(__file__).resolve().parent
    / "models"
    / "package.xml"
)

VESSELS = (
    "test_tube",
    "erlenmeyer",
    "boiling_flask",
)

# Grasp heights above each vessel's body origin.
GRASP_HEIGHTS = {
    "test_tube": 0.07,
    "erlenmeyer": 0.12,
    "boiling_flask": 0.13,
}


# ==================================================
# GEOMETRY-DEPENDENT GRIP FORCE LIMITS
# ==================================================

# These are controller force limits in Newtons.
#
# Larger vessels are assigned a higher force limit
# because of their larger gravitational moment
# during tilting.

GRIP_FORCE_LIMITS = {
    "test_tube": 40.0,
    "erlenmeyer": 60.0,
    "boiling_flask": 60.0,
}


# ==================================================
# MOTION PARAMETERS
# ==================================================

APPROACH_DURATION = 5.0
CLOSE_DURATION = 2.0

LIFT_DISTANCE = 0.10
LIFT_DURATION = 3.0
LIFT_CONTROL_HZ = 50


# ==================================================
# BUILD DRAKE SCENARIO
# ==================================================

def scenario_text(
    vessel_name: str,
    source_x: float,
    source_y: float,
) -> str:

    if vessel_name not in VESSELS:

        raise ValueError(
            f"Unknown source vessel: {vessel_name}"
        )

    return f"""
directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [0]

- add_weld:
    parent: world
    child: iiwa::iiwa_link_0

- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90] }}

- add_model:
    name: table
    file: package://pouring_bot/table.sdf

- add_weld:
    parent: world
    child: table::body
    X_PC:
        translation: [{TABLE_CENTER[0]:.2f}, {TABLE_CENTER[1]:.2f}, {TABLE_CENTER[2]:.2f}]

- add_model:
    name: source_holder
    file: package://pouring_bot/source_holder.sdf

- add_weld:
    parent: world
    child: source_holder::body
    X_PC:
        translation: [{source_x:.3f}, {source_y:.3f}, {TABLE_TOP + 0.01:.2f}]

- add_model:
    name: {vessel_name}
    file: package://pouring_bot/{vessel_name}.sdf

- add_model:
    name: beaker
    file: package://pouring_bot/beaker.sdf

- add_weld:
    parent: world
    child: beaker::body
    X_PC:
        translation: [{BEAKER_X:.2f}, {BEAKER_Y:.2f}, {TABLE_TOP:.2f}]

model_drivers:
    iiwa: !IiwaDriver
        hand_model_name: wsg

    wsg: !SchunkWsgDriver {{}}
"""


# ==================================================
# RANDOM SOURCE POSITION
# ==================================================

def choose_source_position(rng):
    """
    Sample a source position while maintaining
    a minimum separation from the receiving beaker.
    """

    for _ in range(100):

        x = rng.uniform(*SOURCE_X_RANGE)
        y = rng.uniform(*SOURCE_Y_RANGE)

        distance = np.hypot(
            x - BEAKER_X,
            y - BEAKER_Y,
        )

        if distance >= MIN_BEAKER_CLEARANCE:

            return float(x), float(y)

    raise RuntimeError(
        "Could not sample a source position."
    )


# ==================================================
# GRIPPER FORCE HELPERS
# ==================================================

def set_gripper_force_limit(
    station,
    station_context,
    vessel_name,
):
    """
    Set geometry-dependent gripper force limit.

    Check for the required input port first so
    unsupported Drake versions do not crash.
    """

    force_limit = GRIP_FORCE_LIMITS[
        vessel_name
    ]

    input_ports = {
        station.get_input_port(i).get_name()
        for i in range(
            station.num_input_ports()
        )
    }

    if "wsg.force_limit" in input_ports:

        station.GetInputPort(
            "wsg.force_limit"
        ).FixValue(
            station_context,
            [force_limit],
        )

        print(
            f"Commanded grip force limit: "
            f"{force_limit:.1f} N"
        )

        return True

    else:

        print(
            "\nWARNING: Station does not expose "
            "wsg.force_limit."
        )

        print(
            "Using the default gripper "
            "force limit instead."
        )

        print(
            "Available WSG input ports:"
        )

        for port_name in sorted(
            input_ports
        ):

            if "wsg" in port_name:

                print(
                    "  ",
                    port_name,
                )

        return False


def print_gripper_force(
    station,
    station_context,
    label="",
):
    """
    Print measured gripper force if the
    station provides the relevant output.
    """

    output_ports = {
        station.get_output_port(i).get_name()
        for i in range(
            station.num_output_ports()
        )
    }

    if "wsg.force_measured" in output_ports:

        force = np.asarray(
            station.GetOutputPort(
                "wsg.force_measured"
            ).Eval(
                station_context
            )
        ).reshape(-1)

        print(
            f"{label} measured gripper force:",
            np.round(force, 3),
            "N",
        )

        return force

    else:

        print(
            f"{label} measured gripper force "
            "not available."
        )

        return None


def print_wsg_output_ports(station):
    """Display available gripper output signals."""

    print("\nAvailable WSG output ports:")

    for i in range(
        station.num_output_ports()
    ):

        port = station.get_output_port(i)

        name = port.get_name()

        if "wsg" in name:

            print(
                "  ",
                name,
            )


# ==================================================
# JOINT SPACE APPROACH
# ==================================================

def smoothstep(s):

    return 3.0 * s**2 - 2.0 * s**3


def command_joint_motion(
    simulator,
    station,
    station_context,
    q_start,
    q_goal,
    start_time,
    duration,
    steps,
):
    """Command smooth joint-space motion."""

    for i in range(1, steps + 1):

        s = i / steps

        alpha = smoothstep(s)

        q_command = (
            q_start
            + alpha * (q_goal - q_start)
        )

        station.GetInputPort(
            "iiwa.position"
        ).FixValue(
            station_context,
            q_command,
        )

        simulator.AdvanceTo(
            start_time + duration * s
        )


# ==================================================
# WYATT'S JACOBIAN-BASED LIFT
# ==================================================

def command_cartesian_lift(
    simulator,
    station,
    station_context,
    plant,
    plant_context,
    iiwa,
    gripper_frame,
    p_WG_start,
    start_time,
):
    """
    Lift in world Z using the gripper Jacobian.

    Uses differential IK and position feedback
    to track a smooth Cartesian lift profile.
    """

    world = plant.world_frame()

    first_joint = plant.GetJointByName(
        "iiwa_joint_1"
    )

    last_joint = plant.GetJointByName(
        "iiwa_joint_7"
    )

    arm_columns = slice(
        first_joint.velocity_start(),
        last_joint.velocity_start() + 1,
    )

    steps = int(
        LIFT_DURATION * LIFT_CONTROL_HZ
    )

    dt = LIFT_DURATION / steps

    qdot_limit = 1.0

    linear_speed_limit = 0.15

    for i in range(1, steps + 1):

        s = (i - 0.5) / steps

        alpha = smoothstep(s)

        alpha_dot = (
            6.0 * s * (1.0 - s)
            / LIFT_DURATION
        )

        # Desired gripper position.
        p_WG_desired = (
            p_WG_start
            + np.array([
                0.0,
                0.0,
                LIFT_DISTANCE * alpha,
            ])
        )

        # Actual gripper pose.
        X_WG_actual = (
            plant.CalcRelativeTransform(
                plant_context,
                world,
                gripper_frame,
            )
        )

        position_error = (
            p_WG_desired
            - X_WG_actual.translation()
        )

        # Feedforward plus feedback.
        v_WG_linear = (
            np.array([
                0.0,
                0.0,
                LIFT_DISTANCE * alpha_dot,
            ])
            + 2.0 * position_error
        )

        speed = np.linalg.norm(
            v_WG_linear
        )

        if speed > linear_speed_limit:

            v_WG_linear *= (
                linear_speed_limit / speed
            )

        # Drake spatial velocity:
        # [angular velocity; linear velocity].
        V_WG = np.concatenate([
            np.zeros(3),
            v_WG_linear,
        ])

        # Calculate spatial Jacobian.
        J_G = plant.CalcJacobianSpatialVelocity(
            plant_context,
            JacobianWrtVariable.kV,
            gripper_frame,
            [0.0, 0.0, 0.0],
            world,
            world,
        )

        # Wyatt's original pseudoinverse controller.
        qdot = (
            np.linalg.pinv(
                J_G[:, arm_columns]
            )
            @ V_WG
        )

        qdot = np.clip(
            qdot,
            -qdot_limit,
            qdot_limit,
        )

        q_actual = plant.GetPositions(
            plant_context,
            iiwa,
        ).copy()

        q_command = (
            q_actual + dt * qdot
        )

        station.GetInputPort(
            "iiwa.position"
        ).FixValue(
            station_context,
            q_command,
        )

        simulator.AdvanceTo(
            start_time + i * dt
        )


# ==================================================
# MAIN SIMULATION
# ==================================================

def main():

    # ----------------------------------------------
    # GENERATE REPRODUCIBLE SCENARIO
    # ----------------------------------------------

    if RANDOM_SEED is None:

        seed = int(
            np.random.SeedSequence().entropy
        )

    else:

        seed = RANDOM_SEED

    rng = np.random.default_rng(seed)

    # ----------------------------------------------
    # SELECT VESSEL GEOMETRY
    # ----------------------------------------------

    if RANDOMIZE_VESSEL:

        vessel_name = str(
            rng.choice(VESSELS)
        )

    else:

        vessel_name = FIXED_VESSEL

    # ----------------------------------------------
    # SELECT SOURCE POSITION
    # ----------------------------------------------

    if RANDOMIZE_POSITION:

        source_x, source_y = (
            choose_source_position(rng)
        )

    else:

        source_x = FIXED_SOURCE_X
        source_y = FIXED_SOURCE_Y

    grip_force_limit = (
        GRIP_FORCE_LIMITS[vessel_name]
    )

    print("\n" + "=" * 55)
    print("POURING SCENARIO")
    print("=" * 55)

    print(
        "Random seed:",
        seed,
    )

    print(
        "Randomize position:",
        RANDOMIZE_POSITION,
    )

    print(
        "Randomize vessel:",
        RANDOMIZE_VESSEL,
    )

    print(
        "Selected vessel:",
        vessel_name,
    )

    print(
        "Source position:",
        np.round(
            [
                source_x,
                source_y,
                HOLDER_TOP,
            ],
            3,
        ),
    )

    print(
        f"Requested grip force limit: "
        f"{grip_force_limit:.1f} N"
    )

    print("=" * 55)

    # ----------------------------------------------
    # BUILD SIMULATION
    # ----------------------------------------------

    meshcat = Meshcat()

    builder = DiagramBuilder()

    scenario = LoadScenario(
        data=scenario_text(
            vessel_name,
            source_x,
            source_y,
        )
    )

    station = builder.AddSystem(
        MakeHardwareStation(
            scenario,
            meshcat=meshcat,
            package_xmls=[str(PACKAGE_XML)],
        )
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)

    context = (
        simulator.get_mutable_context()
    )

    station_context = (
        station.GetMyContextFromRoot(
            context
        )
    )

    plant = station.GetSubsystemByName(
        "plant"
    )

    plant_context = (
        plant.GetMyContextFromRoot(
            context
        )
    )

    iiwa = plant.GetModelInstanceByName(
        "iiwa"
    )

    wsg = plant.GetModelInstanceByName(
        "wsg"
    )

    vessel_instance = (
        plant.GetModelInstanceByName(
            vessel_name
        )
    )

    vessel_body = plant.GetBodyByName(
        "body",
        vessel_instance,
    )

    # ----------------------------------------------
    # INITIALIZE FREE VESSEL
    # ----------------------------------------------

    X_WV = RigidTransform([
        source_x,
        source_y,
        HOLDER_TOP,
    ])

    plant.SetFreeBodyPose(
        plant_context,
        vessel_body,
        X_WV,
    )

    print(
        "\nIs vessel floating?",
        vessel_body.is_floating_base_body(),
    )

    q_home = plant.GetPositions(
        plant_context,
        iiwa,
    ).copy()

    # ----------------------------------------------
    # PLAN GEOMETRY-DEPENDENT GRASP
    # ----------------------------------------------

    grasp_target = np.array([
        source_x,
        source_y,
        HOLDER_TOP
        + GRASP_HEIGHTS[vessel_name],
    ])

    print(
        "Grasp target:",
        np.round(
            grasp_target,
            4,
        ),
    )

    q_grasp = solve_pregrasp_ik(
        plant,
        plant_context,
        grasp_target,
    )

    # ----------------------------------------------
    # INITIALIZE CONTROLLER INPUTS
    # ----------------------------------------------

    station.GetInputPort(
        "iiwa.position"
    ).FixValue(
        station_context,
        q_home,
    )

    station.GetInputPort(
        "iiwa.torque"
    ).FixValue(
        station_context,
        np.zeros(7),
    )

    station.GetInputPort(
        "wsg.position"
    ).FixValue(
        station_context,
        [0.08],
    )

    # Set the geometry-dependent force limit.
    #
    # This remains fixed throughout the entire
    # grasp, lift, transport, and pouring sequence.

    force_limit_applied = (
        set_gripper_force_limit(
            station,
            station_context,
            vessel_name,
        )
    )

    # Initialize open gripper.
    plant.SetPositions(
        plant_context,
        wsg,
        np.array([
            -0.04,
            0.04,
        ]),
    )

    simulator.set_target_realtime_rate(
        1.0
    )

    simulator.Initialize()

    diagram.ForcedPublish(
        context
    )

    print(
        "\nMeshcat:",
        meshcat.web_url(),
    )

    # Print available WSG output ports for
    # force measurement and debugging.
    print_wsg_output_ports(
        station
    )

    pickup_success = False
    pour_success = False

    # ----------------------------------------------
    # CHECK GRASP IK
    # ----------------------------------------------

    if q_grasp is None:

        print("\nGrasp IK failed.")
        print("No motion attempted.")

    else:

        # ==========================================
        # PHASE 1: APPROACH
        # ==========================================

        print(
            "\nPHASE 1: Approaching vessel..."
        )

        command_joint_motion(
            simulator=simulator,
            station=station,
            station_context=station_context,
            q_start=q_home,
            q_goal=q_grasp,
            start_time=0.0,
            duration=APPROACH_DURATION,
            steps=50,
        )

        q_actual = plant.GetPositions(
            plant_context,
            iiwa,
        ).copy()

        print("Approach complete.")

        print(
            "Joint tracking error:",
            round(
                np.linalg.norm(
                    q_actual - q_grasp
                ),
                4,
            ),
        )

        # ==========================================
        # PHASE 2: CLOSE GRIPPER
        # ==========================================

        print(
            "\nPHASE 2: Closing gripper..."
        )

        station.GetInputPort(
            "iiwa.position"
        ).FixValue(
            station_context,
            q_grasp,
        )

        # Command the gripper fully closed.
        station.GetInputPort(
            "wsg.position"
        ).FixValue(
            station_context,
            [0.0],
        )

        if force_limit_applied:

            print(
                f"Closing with force limit: "
                f"{grip_force_limit:.1f} N"
            )

        else:

            print(
                "Closing with default force limit."
            )

        close_end = (
            APPROACH_DURATION
            + CLOSE_DURATION
        )

        simulator.AdvanceTo(
            close_end
        )

        # ------------------------------------------
        # GRIPPER POSITION AND FORCE
        # ------------------------------------------

        finger_positions = (
            plant.GetPositions(
                plant_context,
                wsg,
            ).copy()
        )

        print(
            "Finger joint positions:",
            np.round(
                finger_positions,
                4,
            ),
        )

        print_gripper_force(
            station,
            station_context,
            label="After closure",
        )

        # ------------------------------------------
        # MEASURE GRASP GEOMETRY
        # ------------------------------------------

        gripper_frame = (
            plant.GetFrameByName(
                "body",
                wsg,
            )
        )

        X_WG = plant.CalcRelativeTransform(
            plant_context,
            plant.world_frame(),
            gripper_frame,
        )

        grasp_point = X_WG.multiply(
            GRASP_POINT_G
        )

        print(
            "Actual grasp point:",
            np.round(
                grasp_point,
                4,
            ),
        )

        # Record the vessel's position
        # before executing the lift.
        vessel_xyz_before = (
            plant.EvalBodyPoseInWorld(
                plant_context,
                vessel_body,
            ).translation().copy()
        )

        print(
            "Vessel position before lift:",
            np.round(
                vessel_xyz_before,
                4,
            ),
        )

        # ==========================================
        # PHASE 3: WYATT'S CARTESIAN LIFT
        # ==========================================

        print(
            "\nPHASE 3: Lifting with "
            "the gripper Jacobian..."
        )

        command_cartesian_lift(
            simulator=simulator,
            station=station,
            station_context=station_context,
            plant=plant,
            plant_context=plant_context,
            iiwa=iiwa,
            gripper_frame=gripper_frame,
            p_WG_start=(
                X_WG.translation().copy()
            ),
            start_time=close_end,
        )

        # ==========================================
        # PHASE 4: EVALUATE PICKUP
        # ==========================================

        vessel_xyz_after = (
            plant.EvalBodyPoseInWorld(
                plant_context,
                vessel_body,
            ).translation().copy()
        )

        height_change = (
            vessel_xyz_after[2]
            - vessel_xyz_before[2]
        )

        print("\nPICKUP RESULTS")

        print(
            "Before lift:",
            np.round(
                vessel_xyz_before,
                4,
            ),
        )

        print(
            "After lift:",
            np.round(
                vessel_xyz_after,
                4,
            ),
        )

        print(
            f"Vertical displacement: "
            f"{height_change:.3f} m"
        )

        # Measure force again after lifting.
        print_gripper_force(
            station,
            station_context,
            label="After lift",
        )

        # Print finger positions after lifting
        # to help identify any opening or slipping.
        finger_positions_after = (
            plant.GetPositions(
                plant_context,
                wsg,
            ).copy()
        )

        print(
            "Finger positions after lift:",
            np.round(
                finger_positions_after,
                4,
            ),
        )

        # ==========================================
        # PHASE 5: START INTEGRATED POUR
        # ==========================================

        if height_change > 0.06:

            pickup_success = True

            print(
                "Vessel rose substantially."
            )

            print(
                "Starting integrated "
                "pouring sequence..."
            )

            pour_success = execute_pour(
                simulator=simulator,
                station=station,
                station_context=station_context,
                plant=plant,
                plant_context=plant_context,
                meshcat=meshcat,
                vessel_body=vessel_body,
                vessel_name=vessel_name,
            )

            # Report grip force after the
            # pouring sequence finishes or stops.
            print_gripper_force(
                station,
                station_context,
                label="After pouring sequence",
            )

            final_finger_positions = (
                plant.GetPositions(
                    plant_context,
                    wsg,
                ).copy()
            )

            print(
                "Final finger positions:",
                np.round(
                    final_finger_positions,
                    4,
                ),
            )

        else:

            print(
                "Pickup not confirmed."
            )

            print(
                "Pouring skipped."
            )

    # ==============================================
    # FINAL SUMMARY
    # ==============================================

    diagram.ForcedPublish(
        context
    )

    print("\n" + "=" * 55)
    print("SIMULATION SUMMARY")
    print("=" * 55)

    print(
        "Random seed:",
        seed,
    )

    print(
        "Source vessel:",
        vessel_name,
    )

    print(
        f"Source position: "
        f"({source_x:.3f}, "
        f"{source_y:.3f}, "
        f"{HOLDER_TOP:.3f})"
    )

    print(
        f"Requested grip force limit: "
        f"{grip_force_limit:.1f} N"
    )

    print(
        "Custom force limit applied:",
        force_limit_applied,
    )

    print(
        "Pickup success:",
        pickup_success,
    )

    print(
        "Pouring motion completed:",
        pour_success,
    )

    print(
        "Meshcat:",
        meshcat.web_url(),
    )

    print("=" * 55)

    input(
        "\nPress Enter to close."
    )


if __name__ == "__main__":

    main()

