
"""KUKA iiwa randomized pouring workstation.

Run:
    python visualize_station.py

Current test:
1. Randomly position a test tube.
2. Solve IK for a side grasp.
3. Move the robot to the grasp pose.
4. Close the gripper.
5. Plan and execute a 10 cm lift.
6. Measure whether the test tube rises.
"""

from pathlib import Path

import numpy as np

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
)

from pydrake.all import (
    DiagramBuilder,
    Meshcat,
    RigidTransform,
    Simulator,
)

from grasp_planner import (
    GRASP_POINT_G,
    solve_pregrasp_ik,
    solve_lift_ik,
)


RANDOM_SEED = None

TABLE_CENTER = (0.60, 0.00, 0.17)
TABLE_TOP = 0.20
HOLDER_TOP = TABLE_TOP + 0.02

SOURCE_X_RANGE = (0.30, 0.80)
SOURCE_Y_RANGE = (-0.25, 0.25)

MIN_BEAKER_CLEARANCE = 0.20

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

GRASP_HEIGHTS = {
    "test_tube": 0.07,
    "erlenmeyer": 0.12,
    "boiling_flask": 0.13,
}

APPROACH_DURATION = 5.0
CLOSE_DURATION = 2.0

LIFT_DISTANCE = 0.10
LIFT_STEPS = 20
LIFT_DURATION = 3.0


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


def choose_source_position(rng):
    """Sample a source position away from the beaker."""

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


def smoothstep(s):
    """Cubic interpolation from 0 to 1."""
    return 3 * s**2 - 2 * s**3


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
    """Command a smooth joint-space motion."""

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


def main():

    rng = np.random.default_rng(RANDOM_SEED)

    # Force test tube while debugging the pickup.
    # Restore random selection later.
    vessel_name = "test_tube"

    # To restore randomized vessel types:
    # vessel_name = str(rng.choice(VESSELS))

    source_x, source_y = choose_source_position(rng)

    print("\nSelected vessel:", vessel_name)
    print(
        "Source position:",
        np.round(
            [source_x, source_y, HOLDER_TOP],
            3,
        ),
    )

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

    context = simulator.get_mutable_context()

    station_context = station.GetMyContextFromRoot(
        context
    )

    plant = station.GetSubsystemByName("plant")

    plant_context = plant.GetMyContextFromRoot(
        context
    )

    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg = plant.GetModelInstanceByName("wsg")

    vessel_instance = plant.GetModelInstanceByName(
        vessel_name
    )

    vessel_body = plant.GetBodyByName(
        "body",
        vessel_instance,
    )

    # Initialize the free vessel on the holder.
    X_WV = RigidTransform(
        [source_x, source_y, HOLDER_TOP]
    )

    plant.SetFreeBodyPose(
        plant_context,
        vessel_body,
        X_WV,
    )

    print(
        "Is vessel floating?",
        vessel_body.is_floating_base_body(),
    )

    # Remember the robot's initial joint positions.
    q_home = plant.GetPositions(
        plant_context,
        iiwa,
    ).copy()

    # Define the desired grasp-center position.
    grasp_target = np.array([
        source_x,
        source_y,
        HOLDER_TOP + GRASP_HEIGHTS[vessel_name],
    ])

    print(
        "Grasp target:",
        np.round(grasp_target, 4),
    )

    # PLAN GRASP
    q_grasp = solve_pregrasp_ik(
        plant,
        plant_context,
        grasp_target,
    )

    # Controller inputs.
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

    # Initialize the gripper in an open configuration.
    plant.SetPositions(
        plant_context,
        wsg,
        np.array([-0.04, 0.04]),
    )

    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    diagram.ForcedPublish(context)

    if q_grasp is None:
        print("\nGrasp IK failed.")
        print("No motion attempted.")

    else:

        # ------------------------------
        # PHASE 1: APPROACH
        # ------------------------------
        print("\nPHASE 1: Approaching vessel...")

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
                np.linalg.norm(q_actual - q_grasp),
                4,
            ),
        )

        # ------------------------------
        # PHASE 2: CLOSE GRIPPER
        # ------------------------------
        print("\nPHASE 2: Closing gripper...")

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

        close_end = (
            APPROACH_DURATION + CLOSE_DURATION
        )

        simulator.AdvanceTo(close_end)

        finger_positions = plant.GetPositions(
            plant_context,
            wsg,
        ).copy()

        print(
            "Finger joint positions:",
            np.round(finger_positions, 4),
        )

        # Grasp-center location after closure.
        gripper_frame = plant.GetFrameByName(
            "body",
            wsg,
        )

        X_WG = plant.CalcRelativeTransform(
            plant_context,
            plant.world_frame(),
            gripper_frame,
        )

        grasp_point = X_WG.multiply(
            GRASP_POINT_G
        )

        R_lift = X_WG.rotation()

        print(
            "Actual grasp point:",
            np.round(grasp_point, 4),
        )

        # Record the vessel's pose after closure.
        vessel_xyz_before = (
            plant.EvalBodyPoseInWorld(
                plant_context,
                vessel_body,
            ).translation().copy()
        )

        print(
            "Vessel position before lift:",
            np.round(vessel_xyz_before, 4),
        )

        # ------------------------------
        # PHASE 3: PLAN INCREMENTAL LIFT
        # ------------------------------
        print("\nPHASE 3: Planning lift...")
        print("\nJoint limit margins:")
        q_all = plant.GetPositions(plant_context)
        lower = plant.GetPositionLowerLimits()
        upper = plant.GetPositionUpperLimits()

        for joint_index in plant.GetJointIndices(iiwa):
            joint = plant.get_joint(joint_index)
            idx = joint.position_start()

            print(
                f"{joint.name()}: "
                f"q={q_all[idx]:.4f}, "
                f"lower margin={q_all[idx] - lower[idx]:.4f}, "
                f"upper margin={upper[idx] - q_all[idx]:.4f}"
            )
        # The simulated robot and vessel stay unchanged
        # throughout this planning phase.
        seed_full = plant.GetPositions(
            plant_context
        ).copy()

        q_start = plant.GetPositions(
            plant_context,
            iiwa,
        ).copy()

        lift_waypoints = []
        lift_feasible = True

        for step in range(1, LIFT_STEPS + 1):

            dz = (
                step
                * LIFT_DISTANCE
                / LIFT_STEPS
            )

            lift_target = (
                grasp_point
                + np.array([0.0, 0.0, dz])
            )

            q_next, next_seed = solve_lift_ik(
                plant=plant,
                plant_context=plant_context,
                target_position=lift_target,
                R_target=R_lift,
                seed_positions=seed_full,
            )

            if q_next is None:
                print(
                    f"Lift IK failed at step "
                    f"{step}/{LIFT_STEPS}."
                )
                lift_feasible = False
                break

            lift_waypoints.append(q_next.copy())

            # Use the previous solution to initialize
            # the next IK solve, without changing
            # the real simulation state.
            seed_full = next_seed.copy()

            if step % 5 == 0:
                print(
                    f"Planned lift: "
                    f"{dz * 100:.1f} cm"
                )

        # ------------------------------
        # PHASE 4: EXECUTE LIFT
        # ------------------------------
        if lift_feasible:

            print(
                "\nAll lift IK targets succeeded!"
            )
            print("\nPHASE 4: Lifting vessel...")

            current_time = close_end
            q_previous = q_start.copy()

            segment_duration = (
                LIFT_DURATION / LIFT_STEPS
            )

            for q_next in lift_waypoints:

                command_joint_motion(
                    simulator=simulator,
                    station=station,
                    station_context=station_context,
                    q_start=q_previous,
                    q_goal=q_next,
                    start_time=current_time,
                    duration=segment_duration,
                    steps=5,
                )

                current_time += segment_duration
                q_previous = q_next.copy()

            # ------------------------------
            # PHASE 5: EVALUATE PICKUP
            # ------------------------------
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
                np.round(vessel_xyz_before, 4),
            )
            print(
                "After lift:",
                np.round(vessel_xyz_after, 4),
            )
            print(
                f"Vertical displacement: "
                f"{height_change:.3f} m"
            )

            if height_change > 0.06:
                print(
                    "Vessel rose substantially."
                )
                print(
                    "Verify it is still held "
                    "between the fingers."
                )
            else:
                print(
                    "Pickup not confirmed."
                )

        else:
            print(
                "\nLift skipped because IK "
                "planning failed."
            )

    diagram.ForcedPublish(context)

    print("\nPouring workstation ready.")
    print(f"Source vessel: {vessel_name}")
    print(
        f"Source position: "
        f"({source_x:.3f}, {source_y:.3f}, "
        f"{HOLDER_TOP:.3f})"
    )
    print("Meshcat:", meshcat.web_url())

    input("\nPress Enter to close.\n")


if __name__ == "__main__":
    main()
