"""Build matching hollow station visuals and export its measured SPH inputs."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from drake_trajectory_export import TrajectoryRecorder
from prepare_sph_pouring import PouringConfig, VesselProfile, cup_mesh, profile_mesh


SOURCE_VESSELS = ("test_tube", "erlenmeyer", "boiling_flask")


def flask_profile(name):
    if name == "erlenmeyer":
        return VesselProfile(
            outer=((0, .040), (.006, .040), (.080, .016), (.160, .016)),
            inner=((.004, .037), (.006, .037), (.080, .013), (.160, .013)),
        )
    if name == "boiling_flask":
        def bulb(radius, base, neck):
            shoulder = .045 + np.sqrt(radius**2 - neck**2)
            heights = np.linspace(base, shoulder, 13)
            rings = [(z, np.sqrt(radius**2 - (z - .045)**2)) for z in heights[:-1]]
            return (*rings, (shoulder, neck), (.170, neck))
        outer = ((0, .009), *bulb(.045, .002, .018))
        inner = bulb(.042, .005, .015)
        return VesselProfile(outer=outer, inner=inner)
    raise ValueError(f"No flask profile for {name}")


def hollow_model(models_dir, output, name, wall=None, bottom=None, *, profile=None):
    """Keep the station collision proxies and replace its visual with a hollow mesh."""
    tree = ET.parse(models_dir / f"{name}.sdf")
    link = tree.find("./model/link")
    visual = link.find("visual")
    mesh_path = output / f"{name}.obj"
    if profile is None:
        dimensions = cylindrical_mesh(link, mesh_path, wall, bottom)
    else:
        dimensions = None
        profile_mesh(mesh_path, profile)
        for extra in link.findall("visual")[1:]:
            link.remove(extra)
    visual.find("pose").text = "0 0 0 0 0 0"
    geometry = visual.find("geometry")
    geometry.clear()
    ET.SubElement(ET.SubElement(geometry, "mesh"), "uri").text = mesh_path.as_uri()
    path = output / f"{name}.sdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path.as_uri(), dimensions


def cylindrical_mesh(link, mesh_path, wall, bottom):
    visual = link.find("visual")
    collision = link.find("collision")
    cylinder = collision.find("./geometry/cylinder")
    radius = float(cylinder.findtext("radius"))
    height = float(cylinder.findtext("length"))
    for element in (collision, visual):
        pose = np.fromstring(element.findtext("pose"), sep=" ")
        if not np.allclose(pose, [0, 0, height / 2, 0, 0, 0]):
            raise ValueError(f"{mesh_path.stem} must have its body origin at the bottom center")

    dimensions = (radius - wall, wall, height, bottom)
    cup_mesh(mesh_path, *dimensions[:3], bottom=bottom)
    return dimensions


class StationSphExport:
    """Capture the whole dry-station run from the initial upright source pose."""

    def __init__(self, output, models_dir, vessel_name="test_tube"):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise ValueError("Station export directory must be empty")
        self.source_profile = None if vessel_name == "test_tube" else flask_profile(vessel_name)
        self.vessel_name = vessel_name
        self.source_model, self.source_cup = hollow_model(
            models_dir, self.output, vessel_name, 0.0015, 0.003, profile=self.source_profile,
        )
        self.receiver_model, self.receiver_cup = hollow_model(
            models_dir, self.output, "beaker", 0.003, 0.004,
        )
        self.recorder = TrajectoryRecorder()

    def start(self, simulator, plant, plant_context, source_body, receiver_body):
        self.simulator = simulator
        self.plant = plant
        self.plant_context = plant_context
        self.source_body = source_body
        self.start_time = simulator.get_context().get_time()
        source = plant.EvalBodyPoseInWorld(plant_context, source_body)
        receiver = plant.EvalBodyPoseInWorld(plant_context, receiver_body)
        for name, pose in (("source", source), ("receiver", receiver)):
            if not np.allclose(pose.rotation().matrix(), np.eye(3), atol=1e-6, rtol=0):
                raise ValueError(f"SPH requires an identity initial {name} rotation")
        self.configure(receiver.translation())
        self.record_pose()

    def configure(self, receiver_origin):
        fill = {
            "test_tube": ((-.007, -.007, .009), (.007, .007, .095)),
            "erlenmeyer": ((-.014, -.014, .010), (.014, .014, .045)),
            "boiling_flask": ((-.015, -.015, .020), (.015, .015, .065)),
        }
        lo, hi = fill[self.vessel_name]
        cup = {"source_cup": self.source_cup} if self.source_cup is not None else {}
        self.config = PouringConfig(
            **cup, source_profile=self.source_profile, receiver_cup=self.receiver_cup,
            receiver_origin=receiver_origin, fill_start=lo, fill_end=hi,
            particle_radius=.0015, max_time_step=.001,
        )

    def record_pose(self):
        time = self.simulator.get_context().get_time() - self.start_time
        if self.recorder.rows and time <= self.recorder.rows[-1][0]:
            raise ValueError("Station pose timestamps must be strictly increasing")
        pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.source_body)
        self.recorder.record(time, "source", pose)

    def save(self, *, motion_completed=None):
        trajectory = self.recorder.save(self.output / "source_trajectory.csv")
        (self.output / "cups.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        if motion_completed is not None:
            status = {"motion_completed": motion_completed}
            (self.output / "station_motion.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"Exported {len(self.recorder.rows)} source poses to {trajectory}")


def main():
    """Export an upright start, mouth-pivot tilt and hold for an outflow control."""
    from pydrake.all import RigidTransform, RotationMatrix

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--controlled", type=Path, required=True, help="New export directory")
    parser.add_argument("--vessel", choices=SOURCE_VESSELS, default="test_tube")
    args = parser.parse_args()
    export = StationSphExport(args.controlled, Path(__file__).with_name("models"), args.vessel)
    export.configure((.72, .14, .20))
    height = (export.source_profile.outer[-1][0]
              if export.source_profile else export.source_cup[2])
    mouth = np.array([.64, .14, .47])
    for time in np.linspace(0, 3, 151):
        u = np.clip((time - .25) / 1.5, 0, 1)
        rotation = RotationMatrix.MakeYRotation(np.deg2rad(120) * u**2 * (3 - 2*u))
        origin = mouth - rotation.matrix() @ [0., 0., height]
        export.recorder.record(time, "source", RigidTransform(rotation, origin))
    export.save()


if __name__ == "__main__":
    main()
