
"""KUKA iiwa camera-guided grasping and pouring workstation.

Complete pipeline:
1. Generate randomized source vessel and position.
2. Observe source vessel with a fixed overhead RGB-D camera.
3. Estimate source vessel x and y coordinates.
4. Plan grasp using the camera-estimated position.
5. Close gripper.
6. Lift using Wyatt's Jacobian controller.
7. Transport and pour using differential IK.
8. Return vessel upright.

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

from perception import (
    camera_config_yaml,
    detect_vessel_position,
)


# ==================================================
# RANDOMIZATION SETTINGS
# ==================================================

# Set to None for a different scenario every run.
# Set to an integer to reproduce a scenario.
RANDOM_SEED = 42

# Start with a fixed test tube while debugging
# camera localization.
RANDOMIZE_POSITION = False
RANDOMIZE_VESSEL = False

FIXED_VESSEL = "test_tube"

FIXED_SOURCE_X = 0.42
FIXED_SOURCE_Y = -0.16

# Randomized source workspace.
SOURCE_X_RANGE = (0.35, 0.60)
SOURCE_Y_RANGE = (-0.23, 0.10)

MIN_BEAKER_CLEARANCE = 0.20

USE_CAMERA_LOCALIZATION = True
PERCEPTION_ONLY = False

# ==================================================
# PERCEPTION SETTINGS
# ==================================================

# True:
# Use the camera to estimate source x and y.
#
# False:
# Use the generated ground-truth position.
USE_CAMERA_LOCALIZATION = True

# True:
# Detect the vessel, show the estimated grasp
# location, and stop before moving the robot.
#
# False:
# Execute the entire grasp-and-pour sequence.
PERCEPTION_ONLY = False


# ==================================================
# WORKSTATION GEOMETRY
# ==================================================

TABLE_CENTER = (0.60, 0.00, 0.17)
TABLE_TOP = 0.20

HOLDER_TOP = TABLE_TOP + 0.02

# The receiving container remains fixed.
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

# Known grasp heights for each geometry.
#
# These use the lower flask grasp positions
# discussed previously to reduce the moment arm
# between the grasp and the center of mass.

GRASP_HEIGHTS = {
    "test_tube": 0.07,
    "erlenmeyer": 0.10,
    "boiling_flask": 0.11,
}


# ==================================================
# GEOMETRY-DEPENDENT GRIP FORCE LIMITS
# ==================================================

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
    """
    Construct the workstation.

    The source holder is positioned randomly.

    The source vessel remains a free body so
    the robot can physically grasp and lift it.

    The receiving beaker is welded in place.
    """

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
    Sample an initial source position.

    Enforce minimum separation between the
    source vessel and the receiving container.
    """

    for _ in range(100):

        x = rng.uniform(
            *SOURCE_X_RANGE
        )

        y = rng.uniform(
            *SOURCE_Y_RANGE
        )

        distance = np.hypot(
            x - BEAKER_X,
            y - BEAKER_Y,
        )

        if distance >= MIN_BEAKER_CLEARANCE:

            return float(x), float(y)

    raise RuntimeError(
        "Could not sample a valid "
        "source position."
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
    Set the geometry-dependent force limit,
    if the station exposes that input port.
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

    print(
        "\nWARNING: Station does not expose "
        "wsg.force_limit."
    )

    print(
        "Using default gripper force limit."
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
    Print measured gripper force when
    the corresponding output is available.
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

    print(
        f"{label} measured gripper force "
        "not available."
    )

    return None


def print_wsg_output_ports(station):
    """Print available gripper outputs."""

    print(
        "\nAvailable WSG output ports:"
    )

    for i in range(
        station.num_output_ports()
    ):

        name = (
            station.get_output_port(i)
            .get_name()
        )

        if "wsg" in name:

            print(
                "  ",
                name,
            )


def print_camera_output_ports(station):
    """Print available camera outputs for debugging."""

    print(
        "\nAvailable camera output ports:"
    )

    for i in range(
        station.num_output_ports()
    ):

        name = (
            station.get_output_port(i)
            .get_name()
        )

        if any(
            keyword in name.lower()
            for keyword in (
                "camera",
                "overhead",
                "rgb",
                "depth",
                "label",
                "image",
            )
        ):

            print(
                "  ",
                name,
            )


# ==================================================
# JOINT SPACE APPROACH
# ==================================================

def smoothstep(s):
    """Smooth interpolation from zero to one."""

    return (
        3.0 * s**2
        - 2.0 * s**3
    )


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

    for i in range(
        1,
        steps + 1,
    ):

        s = i / steps

        alpha = smoothstep(s)

        q_command = (
            q_start
            + alpha
            * (q_goal - q_start)
        )

        station.GetInputPort(
            "iiwa.position"
        ).FixValue(
            station_context,
            q_command,
        )

        simulator.AdvanceTo(
            start_time
            + duration * s
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

    Uses the existing differential IK controller
    with feedforward velocity and position
    feedback to track a smooth vertical motion.
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
        LIFT_DURATION
        * LIFT_CONTROL_HZ
    )

    dt = LIFT_DURATION / steps

    qdot_limit = 1.0

    linear_speed_limit = 0.15

    for i in range(
        1,
        steps + 1,
    ):

        s = (
            (i - 0.5)
            / steps
        )

        alpha = smoothstep(s)

        alpha_dot = (
            6.0
            * s
            * (1.0 - s)
            / LIFT_DURATION
        )

        # Desired Cartesian gripper position.
        p_WG_desired = (
            p_WG_start
            + np.array([
                0.0,
                0.0,
                LIFT_DISTANCE * alpha,
            ])
        )

        # Actual Cartesian gripper position.
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

        # Feedforward velocity plus feedback.
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
                linear_speed_limit
                / speed
            )

        # Spatial velocity convention:
        # angular velocity, then linear velocity.
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

        # Keep only the iiwa columns.
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
            q_actual
            + dt * qdot
        )

        station.GetInputPort(
            "iiwa.position"
        ).FixValue(
            station_context,
            q_command,
        )

        simulator.AdvanceTo(
            start_time
            + i * dt
        )


# ==================================================
# INITIAL CAMERA LOCALIZATION
# ==================================================

def localize_source_vessel(
    station,
    station_context,
    plant,
    plant_context,
    vessel_body,
    vessel_name,
    meshcat,
    source_x,
    source_y,
):
    """
    Determine the initial grasp location.

    In camera mode, the estimated x/y comes
    exclusively from camera observations.

    Ground-truth vessel coordinates are used
    only to calculate localization error.

    In baseline mode, the simulator's generated
    x/y coordinates are used directly.
    """

    # Known z coordinate based on geometry.
    grasp_z = (
        HOLDER_TOP
        + GRASP_HEIGHTS[vessel_name]
    )

    # ----------------------------------------------
    # CAMERA-BASED LOCALIZATION
    # ----------------------------------------------

    if USE_CAMERA_LOCALIZATION:

        print(
            "\nUsing camera-based "
            "source localization..."
        )

        estimated_x, estimated_y = (
            detect_vessel_position(
                station=station,
                station_context=station_context,
                plant=plant,
                vessel_body=vessel_body,
                meshcat=meshcat,
                grasp_z=grasp_z,
            )
        )

    else:

        print(
            "\nUsing ground-truth "
            "source localization."
        )

        estimated_x = source_x

        estimated_y = source_y

    estimated_xy = np.array([
        estimated_x,
        estimated_y,
    ])

    if not np.all(
        np.isfinite(estimated_xy)
    ):

        raise RuntimeError(
            "Localization returned "
            "invalid coordinates."
        )

    # ----------------------------------------------
    # EVALUATE LOCALIZATION ACCURACY
    # ----------------------------------------------

    # IMPORTANT:
    # The true position is read only AFTER
    # the camera estimate has been calculated.
    #
    # It is never passed into the perception
    # estimator or used to correct its output.

    true_position = (
        plant.EvalBodyPoseInWorld(
            plant_context,
            vessel_body,
        ).translation().copy()
    )

    true_xy = true_position[:2]

    localization_error = (
        np.linalg.norm(
            estimated_xy
            - true_xy
        )
    )

    print("\n" + "=" * 55)
    print("LOCALIZATION RESULTS")
    print("=" * 55)

    print(
        "Vessel geometry:",
        vessel_name,
    )

    print(
        "Ground-truth source x/y:",
        np.round(
            true_xy,
            5,
        ),
    )

    print(
        "Estimated source x/y:",
        np.round(
            estimated_xy,
            5,
        ),
    )

    print(
        f"Localization error: "
        f"{1000.0 * localization_error:.2f} mm"
    )

    # ----------------------------------------------
    # GENERATE GRASP TARGET
    # ----------------------------------------------

    grasp_target = np.array([
        estimated_x,
        estimated_y,
        grasp_z,
    ])

    print(
        "Known grasp height:",
        round(grasp_z, 5),
        "m",
    )

    print(
        "Estimated grasp target:",
        np.round(
            grasp_target,
            5,
        ),
    )

    print("=" * 55)

    return (
        grasp_target,
        localization_error,
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
            np.random.SeedSequence()
            .entropy
        )

    else:

        seed = RANDOM_SEED

    rng = np.random.default_rng(
        seed
    )

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

    # This is simulation ground truth.
    #
    # It is used to place the object in
    # the environment, but is not provided
    # to the camera-based grasp planner.

    if RANDOMIZE_POSITION:

        source_x, source_y = (
            choose_source_position(rng)
        )

    else:

        source_x = FIXED_SOURCE_X

        source_y = FIXED_SOURCE_Y

    grip_force_limit = (
        GRIP_FORCE_LIMITS[
            vessel_name
        ]
    )

    print("\n" + "=" * 55)
    print("CAMERA-GUIDED POURING SCENARIO")
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
        "Generated source position "
        "(ground truth):",
        np.round([
            source_x,
            source_y,
            HOLDER_TOP,
        ], 3),
    )

    print(
        "Fixed receiver position:",
        np.round([
            BEAKER_X,
            BEAKER_Y,
            TABLE_TOP,
        ], 3),
    )

    print(
        "Use camera localization:",
        USE_CAMERA_LOCALIZATION,
    )

    print(
        "Perception-only mode:",
        PERCEPTION_ONLY,
    )

    print(
        f"Requested grip force limit: "
        f"{grip_force_limit:.1f} N"
    )

    print("=" * 55)

    # ----------------------------------------------
    # BUILD WORKSTATION AND CAMERA
    # ----------------------------------------------

    meshcat = Meshcat()

    builder = DiagramBuilder()

    # Existing workstation configuration.
    scenario_data = scenario_text(
        vessel_name,
        source_x,
        source_y,
    )

    # Append the fixed overhead RGB-D camera.
    if USE_CAMERA_LOCALIZATION:

        scenario_data += (
            camera_config_yaml()
        )

    scenario = LoadScenario(
        data=scenario_data
    )

    station = builder.AddSystem(
        MakeHardwareStation(
            scenario,
            meshcat=meshcat,
            package_xmls=[
                str(PACKAGE_XML)
            ],
        )
    )

    diagram = builder.Build()

    simulator = Simulator(
        diagram
    )

    context = (
        simulator.get_mutable_context()
    )

    station_context = (
        station.GetMyContextFromRoot(
            context
        )
    )

    plant = (
        station.GetSubsystemByName(
            "plant"
        )
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
    # INITIALIZE FREE SOURCE VESSEL
    # ----------------------------------------------

    # Place the source vessel in the scene.
    #
    # The robot will later estimate this
    # position independently from the camera.

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
    # INITIALIZE CONTROLLER INPUTS
    # ----------------------------------------------

    # Hold the arm at its initial position.
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

    # Start with the gripper open.
    station.GetInputPort(
        "wsg.position"
    ).FixValue(
        station_context,
        [0.08],
    )

    # Geometry-dependent gripping force.
    force_limit_applied = (
        set_gripper_force_limit(
            station,
            station_context,
            vessel_name,
        )
    )

    plant.SetPositions(
        plant_context,
        wsg,
        np.array([
            -0.04,
            0.04,
        ]),
    )

    # ----------------------------------------------
    # INITIALIZE SIMULATION
    # ----------------------------------------------

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

    print_wsg_output_ports(
        station
    )

    if USE_CAMERA_LOCALIZATION:

        print_camera_output_ports(
            station
        )

    # ==============================================
    # PHASE 0: CAMERA-BASED SOURCE LOCALIZATION
    # ==============================================

    print(
        "\nPHASE 0: Localizing "
        "source vessel..."
    )

    try:

        grasp_target, localization_error = (
            localize_source_vessel(
                station=station,
                station_context=station_context,
                plant=plant,
                plant_context=plant_context,
                vessel_body=vessel_body,
                vessel_name=vessel_name,
                meshcat=meshcat,
                source_x=source_x,
                source_y=source_y,
            )
        )

    except (
        RuntimeError,
        ValueError,
    ) as error:

        print(
            "\nLOCALIZATION FAILED:"
        )

        print(
            error
        )

        print(
            "Robot motion skipped."
        )

        if USE_CAMERA_LOCALIZATION:

            print_camera_output_ports(
                station
            )

        input(
            "\nPress Enter to close."
        )

        return

    # ----------------------------------------------
    # PERCEPTION-ONLY TEST MODE
    # ----------------------------------------------

    if PERCEPTION_ONLY:

        print("\n" + "=" * 55)
        print("PERCEPTION-ONLY TEST COMPLETE")
        print("=" * 55)

        print(
            "Selected vessel:",
            vessel_name,
        )

        print(
            "Random seed:",
            seed,
        )

        print(
            f"Localization error: "
            f"{1000.0 * localization_error:.2f} mm"
        )

        if USE_CAMERA_LOCALIZATION:

            print(
                "\nRed marker: "
                "estimated grasp location."
            )

            print(
                "Blue marker: "
                "estimated vessel top center."
            )

        print(
            "\nRobot motion is disabled "
            "in perception-only mode."
        )

        print(
            "Set PERCEPTION_ONLY = False "
            "to execute the full pipeline."
        )

        print(
            "\nMeshcat:",
            meshcat.web_url(),
        )

        input(
            "\nPress Enter to close."
        )

        return

    # ==============================================
    # PLAN GRASP USING ESTIMATED LOCATION
    # ==============================================

    print(
        "\nPlanning grasp using "
        "estimated vessel position..."
    )

    q_grasp = solve_pregrasp_ik(
        plant,
        plant_context,
        grasp_target,
    )

    pickup_success = False

    pour_success = False

    # ----------------------------------------------
    # CHECK GRASP IK
    # ----------------------------------------------

    if q_grasp is None:

        print(
            "\nGrasp IK failed."
        )

        print(
            "No motion attempted."
        )

    else:

        # ==========================================
        # PHASE 1: APPROACH
        # ==========================================

        print(
            "\nPHASE 1: Approaching vessel..."
        )

        # Use current simulation time so
        # the approach works even if camera
        # initialization timing changes later.

        approach_start = (
            simulator.get_context()
            .get_time()
        )

        q_approach_start = (
            plant.GetPositions(
                plant_context,
                iiwa,
            ).copy()
        )

        command_joint_motion(
            simulator=simulator,
            station=station,
            station_context=station_context,
            q_start=q_approach_start,
            q_goal=q_grasp,
            start_time=approach_start,
            duration=APPROACH_DURATION,
            steps=50,
        )

        q_actual = plant.GetPositions(
            plant_context,
            iiwa,
        ).copy()

        print(
            "Approach complete."
        )

        print(
            "Joint tracking error:",
            round(
                np.linalg.norm(
                    q_actual
                    - q_grasp
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
                "Closing with default "
                "force limit."
            )

        close_end = (
            simulator.get_context()
            .get_time()
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
        # MEASURE ACTUAL GRIPPER POSE
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

        print(
            "Desired grasp point:",
            np.round(
                grasp_target,
                4,
            ),
        )

        # Record the actual vessel position
        # before lift for pickup evaluation.

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

        print(
            "\nPICKUP RESULTS"
        )

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

        print_gripper_force(
            station,
            station_context,
            label="After lift",
        )

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

            # The existing pouring planner is
            # unchanged. It measures the grasp
            # transform after pickup and executes
            # transport, tilt, pour, and recovery.

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
        f"Generated source position: "
        f"({source_x:.3f}, "
        f"{source_y:.3f}, "
        f"{HOLDER_TOP:.3f})"
    )

    print(
        "Camera localization enabled:",
        USE_CAMERA_LOCALIZATION,
    )

    print(
        "Estimated source x/y:",
        np.round(
            grasp_target[:2],
            5,
        ),
    )

    print(
        f"Localization error: "
        f"{1000.0 * localization_error:.2f} mm"
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
