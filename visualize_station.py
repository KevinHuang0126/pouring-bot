"""KUKA iiwa pouring workstation.

The arm, gripper, table, holder, source vessel, and beaker are all bodies in
the plant. Labware is welded in place so the scene holds still. Run:

    python visualize_station.py
"""

from pathlib import Path

import numpy as np

from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import DiagramBuilder, Meshcat, Simulator
from grasp_planner import solve_pregrasp_ik

# None picks a new source vessel on every run. An integer repeats that vessel.
RANDOM_SEED = None

# Table slab is 0.80 m by 0.70 m by 0.06 m. Its near edge is at x = 0.20,
# clear of the iiwa base collision box, which extends to about x = 0.11.
TABLE_CENTER = (0.60, 0.00, 0.17)
TABLE_TOP = 0.20

HOLDER_TOP = TABLE_TOP + 0.02

# Initial sampling region within the tabletop (meters)
SOURCE_X_RANGE = (0.30, 0.80)
SOURCE_Y_RANGE = (-0.25, 0.25)

MIN_BEAKER_CLEARANCE = 0.20

BEAKER_X = 0.72
BEAKER_Y = 0.14

PACKAGE_XML = Path(__file__).resolve().parent / "models" / "package.xml"

VESSELS = ("test_tube", "erlenmeyer", "boiling_flask")


def scenario_text(vessel_name: str, source_x: float, source_y: float) -> str:
    if vessel_name not in VESSELS:
        raise ValueError(f"Unknown source vessel: {vessel_name}")

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
- add_weld:
    parent: world
    child: {vessel_name}::body
    X_PC:
        translation: [{source_x:.3f}, {source_y:.3f}, {HOLDER_TOP:.2f}]
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


def choose_vessel(seed):
    rng = np.random.default_rng(seed)
    return str(rng.choice(VESSELS))
def choose_source_position(rng):
    """Generate a random tabletop position for the source vessel."""

    for _ in range(100):
        x = rng.uniform(*SOURCE_X_RANGE)
        y = rng.uniform(*SOURCE_Y_RANGE)

        distance = np.hypot(x - BEAKER_X, y - BEAKER_Y)

        if distance >= MIN_BEAKER_CLEARANCE:
            return float(x), float(y)

    raise RuntimeError("No valid source position found")

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    vessel_name = str(rng.choice(VESSELS))
    source_x, source_y = choose_source_position(rng)

    meshcat = Meshcat()

    builder = DiagramBuilder()
    scenario = LoadScenario(
        data=scenario_text(vessel_name, source_x, source_y)
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
    station_context = station.GetMyContextFromRoot(context)

    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    iiwa = plant.GetModelInstanceByName("iiwa")
    q_home = plant.GetPositions(plant_context, iiwa)

    GRASP_HEIGHTS = {
        "test_tube": 0.07,
        "erlenmeyer": 0.12,
        "boiling_flask": 0.13,
    }

    grasp_target = np.array([
        source_x,
        source_y,
        HOLDER_TOP + GRASP_HEIGHTS[vessel_name],
    ])

    print("Grasp target:", grasp_target)

    q_ik = solve_pregrasp_ik(
        plant,
        plant_context,
        grasp_target,
    )

    q_target = q_ik if q_ik is not None else q_home

    station.GetInputPort("iiwa.position").FixValue(
        station_context, q_target
    )
    station.GetInputPort("iiwa.torque").FixValue(station_context, np.zeros(7))
    station.GetInputPort("wsg.position").FixValue(station_context, [0.08])

    for name in ("table", "source_holder", vessel_name, "beaker"):
        plant.GetModelInstanceByName(name)

    simulator.Initialize()

    # Display the IK configuration.
    plant.SetPositions(plant_context, iiwa, q_target)

    # Display the Schunk gripper with an 8 cm opening.
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetPositions(
        plant_context,
        wsg,
        np.array([-0.04, 0.04]),
    )

    diagram.ForcedPublish(context)
    gripper_frame = plant.GetFrameByName("body", wsg)

    X_WG = plant.CalcRelativeTransform(
        plant_context,
        plant.world_frame(),
        gripper_frame,
    )

    print("Gripper body position:", X_WG.translation())
    print("Gripper body rotation:")
    print(X_WG.rotation().matrix())

    print("\nSchunk WSG body positions relative to gripper:")

    for body_index in plant.GetBodyIndices(wsg):
        body = plant.get_body(body_index)

        X_GB = plant.CalcRelativeTransform(
            plant_context,
            gripper_frame,
            body.body_frame(),
        )

        print(
            body.name(),
            np.round(X_GB.translation(), 4),
        )

        # Show the computed configuration immediately.
        # This is a kinematic preview, not simulated arm motion.
        plant.SetPositions(plant_context, iiwa, q_target)

        diagram.ForcedPublish(context)
        diagram.ForcedPublish(context)

    print("\nPouring workstation ready.")
    print(f"Source vessel: {vessel_name}")
    print(f"Source position: ({source_x:.3f}, {source_y:.3f}, {HOLDER_TOP:.3f})")
    print(f"Beaker position: ({BEAKER_X}, {BEAKER_Y}, {TABLE_TOP})")
    print("Labware is welded into the plant and has collision geometry.")
    print("Meshcat:", meshcat.web_url())
    input("\nPress Enter to close.\n")


if __name__ == "__main__":
    main()
