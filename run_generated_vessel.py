"""Run the existing Drake grasp-and-pour simulation on a generated vessel."""
import pour_planner
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import simple_pour_trajectory as simple_pour
import visualize_station as station


ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "models" / "generated"


def load_geometry(name):
    path = GENERATED_DIR / f"{name}.sdf"

    if not path.is_file():
        raise FileNotFoundError(f"Generated vessel not found: {path}")

    root = ET.parse(path).getroot()

    def read_cylinder(part):
        element = root.find(
            f".//collision[@name='{part}']/geometry/cylinder"
        )

        if element is None:
            raise ValueError(f"Missing {part} cylinder in {path}")

        radius = float(element.findtext("radius"))
        height = float(element.findtext("length"))

        return radius, height

    body_radius, body_height = read_cylinder("base")
    neck_radius, neck_height = read_cylinder("neck")

    return {
        "body_radius": body_radius,
        "body_height": body_height,
        "neck_radius": neck_radius,
        "neck_height": neck_height,
        "total_height": body_height + neck_height,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vessel", default="vessel_001")
    parser.add_argument("--grasp-height", type=float, default=None)
    parser.add_argument("--source-x", type=float, default=0.508)
    parser.add_argument("--source-y", type=float, default=-0.047)

    args = parser.parse_args()

    name = args.vessel

    geometry = load_geometry(name)

    # Default grasp: middle of the narrower neck.
    grasp_height = args.grasp_height

    if grasp_height is None:
        grasp_height = (
            geometry["body_height"]
            + 0.5 * geometry["neck_height"]
        )

    if not 0.0 < grasp_height < geometry["total_height"]:
        raise ValueError("Grasp height must lie within vessel height")

    # Register the vessel with the ballistic pouring model.
    # pour_planner imports this same VESSELS dictionary.
    simple_pour.VESSELS[name] = simple_pour.VesselSpec(
        name=name,
        height=geometry["total_height"],
        mouth_radius=geometry["neck_radius"],
        fill_display_radius=0.75 * geometry["neck_radius"],
    )

    # Register the vessel with the existing robot workstation.
    station.VESSELS = tuple(
        dict.fromkeys((*station.VESSELS, name))
    )

    station.GRASP_HEIGHTS[name] = grasp_height
    station.GRIP_FORCE_LIMITS[name] = 60.0
    # Match the standardized pouring baseline.
    pour_planner.VESSEL_TILT_ANGLES[name] = 95.0
    pour_planner.VESSEL_TILT_DURATIONS[name] = 10.0

    # Use a reproducible starting position.
    station.RANDOMIZE_VESSEL = False
    station.RANDOMIZE_POSITION = False
    station.FIXED_VESSEL = name

    station.FIXED_SOURCE_X = args.source_x
    station.FIXED_SOURCE_Y = args.source_y

    # Redirect only the generated vessel's model path.
    # All other scenario components remain unchanged.
    original_scenario_text = station.scenario_text

    def generated_scenario_text(vessel_name, source_x, source_y):
        text = original_scenario_text(
            vessel_name, source_x, source_y
        )

        if vessel_name == name:
            text = text.replace(
                f"package://pouring_bot/{name}.sdf",
                f"package://pouring_bot/generated/{name}.sdf",
            )

        return text

    station.scenario_text = generated_scenario_text

    print("\n" + "=" * 55)
    print("GENERATED VESSEL EXPERIMENT")
    print("=" * 55)
    print("Vessel:", name)
    print("Body radius:", geometry["body_radius"], "m")
    print("Body height:", geometry["body_height"], "m")
    print("Neck radius:", geometry["neck_radius"], "m")
    print("Neck height:", geometry["neck_height"], "m")
    print("Total height:", geometry["total_height"], "m")
    print("Grasp height:", grasp_height, "m")
    print("=" * 55)

    # Run the original grasp, lift, and 95-degree pour pipeline.
    station.main()


if __name__ == "__main__":
    main()
