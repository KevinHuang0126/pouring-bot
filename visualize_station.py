
"""Static KUKA iiwa pouring workstation with random source flask."""

import numpy as np

from manipulation.station import LoadScenario, MakeHardwareStation
from pydrake.all import (
    Box,
    Cylinder,
    DiagramBuilder,
    Meshcat,
    Rgba,
    RigidTransform,
    Simulator,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

# None = random vessel each run.
# Set to an integer, e.g. 42, for repeatable results.
RANDOM_SEED = None

TABLE_Z = 0.20

SOURCE_X = 0.35
SOURCE_Y = -0.20

BEAKER_X = 0.65
BEAKER_Y = 0.20

GLASS_COLOR = Rgba(0.65, 0.90, 0.98, 0.55)

SCENARIO = """
directives:
- add_directives:
    file: package://manipulation/iiwa_and_wsg.dmd.yaml
model_drivers:
    iiwa: !IiwaDriver
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""


# ==========================================================
# GEOMETRY HELPERS
# ==========================================================

def add_box(meshcat, path, dimensions, position, color):
    meshcat.SetObject(path, Box(*dimensions), Rgba(*color))
    meshcat.SetTransform(path, RigidTransform(position))


def add_cylinder(meshcat, path, radius, height, position, color):
    meshcat.SetObject(path, Cylinder(radius, height), Rgba(*color))
    meshcat.SetTransform(path, RigidTransform(position))


def add_revolved_vessel(meshcat, path, profile, position):
    """
    Create a hollow rotationally symmetric glass vessel.

    profile is a sequence of (radius, height) points
    tracing the outer bottom, outer wall, rim,
    inner wall, and inner bottom.
    """

    sections = 96
    vertices = []
    faces = []

    for radius, z in profile:
        for i in range(sections):
            theta = 2.0 * np.pi * i / sections
            vertices.append([
                radius * np.cos(theta),
                radius * np.sin(theta),
                z,
            ])

    for ring in range(len(profile) - 1):
        for i in range(sections):
            j = (i + 1) % sections

            a = ring * sections + i
            b = ring * sections + j
            c = (ring + 1) * sections + j
            d = (ring + 1) * sections + i

            faces.extend([
                [a, b, c],
                [a, c, d],
            ])

    vertices = np.asarray(vertices, dtype=float).T
    faces = np.asarray(faces, dtype=np.int32).T

    meshcat.SetTriangleMesh(
        path,
        vertices,
        faces,
        rgba=GLASS_COLOR,
    )

    meshcat.SetTransform(path, RigidTransform(position))


# ==========================================================
# SOURCE VESSEL MODELS
# ==========================================================
def add_test_tube(meshcat, x, y, z):
    """Original upright cylindrical test tube."""

    radius = 0.012
    height = 0.12
    thickness = 0.002

    profile = [
        (0.0, 0.0),
        (radius, 0.0),
        (radius, height),
        (radius - thickness, height),
        (radius - thickness, thickness),
        (0.0, thickness),
    ]

    add_revolved_vessel(
        meshcat,
        "/lab/source/test_tube",
        profile,
        [x, y, z],
    )

def add_erlenmeyer(meshcat, x, y, z):
    """Flat-bottom conical Erlenmeyer flask."""

    # Outer wall: flat base, conical body, narrow neck.
    # Inner wall follows the outside with a glass thickness.
    profile = [
        (0.000, 0.000),
        (0.045, 0.000),
        (0.045, 0.010),
        (0.043, 0.025),
        (0.040, 0.040),
        (0.032, 0.070),
        (0.022, 0.105),
        (0.016, 0.120),
        (0.016, 0.158),
        (0.019, 0.160),  # Lip
        (0.013, 0.160),  # Opening
        (0.013, 0.120),
        (0.019, 0.103),
        (0.029, 0.068),
        (0.037, 0.038),
        (0.041, 0.012),
        (0.041, 0.004),
        (0.000, 0.004),
    ]

    add_revolved_vessel(
        meshcat,
        "/lab/source/erlenmeyer",
        profile,
        [x, y, z],
    )


def add_boiling_flask(meshcat, x, y, z):
    """Round-bottom boiling flask with spherical body."""

    radius = 0.045
    center_z = 0.063

    # Spherical body from bottom toward shoulder.
    # Neck starts above the rounded body.
    angles = np.linspace(-np.pi / 2, 1.10, 36)

    outer = [
        (radius * np.cos(a),
         center_z + radius * np.sin(a))
        for a in angles
    ]

    inner_radius = radius - 0.003
    inner = [
        (inner_radius * np.cos(a),
         center_z + inner_radius * np.sin(a))
        for a in angles[::-1]
    ]

    profile = (
        outer
        + [
            (0.020, 0.109),
            (0.020, 0.170),
            (0.023, 0.173),  # Glass lip
            (0.017, 0.173),  # Opening
            (0.017, 0.109),
        ]
        + inner
        + [(0.0, center_z - inner_radius)]
    )

    add_revolved_vessel(
        meshcat,
        "/lab/source/boiling_flask",
        profile,
        [x, y, z],
    )


# ==========================================================
# RECEIVING BEAKER
# ==========================================================

def add_beaker(meshcat, x, y, z):
    """Straight-sided, hollow, open-top beaker."""

    radius = 0.04
    height = 0.09
    thickness = 0.003

    profile = [
        (0.0, 0.0),
        (radius, 0.0),
        (radius, height),
        (radius - thickness, height),
        (radius - thickness, thickness),
        (0.0, thickness),
    ]

    add_revolved_vessel(
        meshcat,
        "/lab/beaker",
        profile,
        [x, y, z],
    )


# ==========================================================
# WORKSTATION
# ==========================================================

def draw_labware(meshcat):
    """Construct a static workstation with one random source."""

    meshcat.Delete("/lab")

    # Worktable
    add_box(
        meshcat,
        "/lab/table",
        [1.2, 0.8, 0.06],
        [0.55, 0.0, 0.17],
        [0.65, 0.68, 0.72, 1.0],
    )

    # Holder base / source vessel platform
    add_box(
        meshcat,
        "/lab/source_holder/base",
        [0.16, 0.14, 0.02],
        [SOURCE_X, SOURCE_Y, TABLE_Z + 0.01],
        [0.15, 0.40, 0.72, 1.0],
    )

    # Randomly select one source vessel
    rng = np.random.default_rng(RANDOM_SEED)

    vessel_type = rng.choice([
        "test_tube","erlenmeyer",
        "boiling",
    ])

    if vessel_type == "test_tube":
    # Original test tube with its upright holder
        add_cylinder(
            meshcat,
            "/lab/source_holder/test_tube_holder",
            0.022,
            0.05,
            [SOURCE_X, SOURCE_Y, TABLE_Z + 0.045],
            [0.25, 0.45, 0.75, 1.0],
        )

        add_test_tube(
            meshcat,
            SOURCE_X,
            SOURCE_Y,
            
            TABLE_Z + 0.02,
        )

    elif vessel_type == "erlenmeyer":
        add_erlenmeyer(
            meshcat,
            SOURCE_X,
            SOURCE_Y,
            TABLE_Z + 0.02,
        )

    elif vessel_type == "boiling":
        add_boiling_flask(
            meshcat,
            SOURCE_X,
            SOURCE_Y,
            TABLE_Z + 0.02,
        )

    # Receiving beaker remains fixed
    add_beaker(
        meshcat,
        BEAKER_X,
        BEAKER_Y,
        TABLE_Z,
    )

    return vessel_type


# ==========================================================
# MAIN
# ==========================================================

def main():
    meshcat = Meshcat()

    builder = DiagramBuilder()
    scenario = LoadScenario(data=SCENARIO)

    station = builder.AddSystem(
        MakeHardwareStation(scenario, meshcat=meshcat)
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    station_context = station.GetMyContextFromRoot(context)

    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    iiwa = plant.GetModelInstanceByName("iiwa")
    q_home = plant.GetPositions(plant_context, iiwa)

    # Robot stays at its initial position
    station.GetInputPort("iiwa.position").FixValue(
        station_context,
        q_home,
    )

    station.GetInputPort("iiwa.torque").FixValue(
        station_context,
        np.zeros(7),
    )

    station.GetInputPort("wsg.position").FixValue(
        station_context,
        [0.08],
    )

    simulator.Initialize()
    diagram.ForcedPublish(context)

    vessel_type = draw_labware(meshcat)

    print("\nStatic pouring workstation ready!")
    print(f"Source vessel: {vessel_type}")
    print(f"Source position: ({SOURCE_X}, {SOURCE_Y})")
    print(f"Beaker position: ({BEAKER_X}, {BEAKER_Y})")
    print("Meshcat:", meshcat.web_url())

    input("\nPress Enter to close.\n")


if __name__ == "__main__":
    main()
