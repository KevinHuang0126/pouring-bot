"""Build matching hollow station visuals and export its measured SPH inputs."""

from dataclasses import asdict
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from drake_trajectory_export import TrajectoryRecorder
from prepare_sph_pouring import PouringConfig, cup_mesh


def hollow_model(models_dir, output, name, wall, bottom):
    """Keep the station cylinder collision proxy and give its visual a cavity."""
    tree = ET.parse(models_dir / f"{name}.sdf")
    link = tree.find("./model/link")
    visual = link.find("visual")
    collision = link.find("collision")
    cylinder = collision.find("./geometry/cylinder")
    radius = float(cylinder.findtext("radius"))
    height = float(cylinder.findtext("length"))
    for element in (collision, visual):
        pose = np.fromstring(element.findtext("pose"), sep=" ")
        if not np.allclose(pose, [0, 0, height / 2, 0, 0, 0]):
            raise ValueError(f"{name} must have its body origin at the bottom center")

    dimensions = (radius - wall, wall, height, bottom)
    mesh_path = output / f"{name}.obj"
    cup_mesh(mesh_path, *dimensions[:3], bottom=bottom)
    visual.find("pose").text = "0 0 0 0 0 0"
    geometry = visual.find("geometry")
    geometry.clear()
    ET.SubElement(ET.SubElement(geometry, "mesh"), "uri").text = mesh_path.as_uri()
    path = output / f"{name}.sdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path.as_uri(), dimensions


class StationSphExport:
    """Capture the whole dry-station run from the initial upright source pose."""

    def __init__(self, output, models_dir):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise ValueError("Station export directory must be empty")
        self.source_model, self.source_cup = hollow_model(
            models_dir, self.output, "test_tube", 0.0015, 0.003,
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
        self.config = PouringConfig(
            source_cup=self.source_cup,
            receiver_cup=self.receiver_cup,
            receiver_origin=receiver.translation(),
            fill_start=(-0.007, -0.007, 0.009),
            fill_end=(0.007, 0.007, 0.095),
            particle_radius=0.0015,
            max_time_step=0.001,
        )
        self.record_pose()

    def record_pose(self):
        time = self.simulator.get_context().get_time() - self.start_time
        if self.recorder.rows and time <= self.recorder.rows[-1][0]:
            raise ValueError("Station pose timestamps must be strictly increasing")
        pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.source_body)
        self.recorder.record(time, "source", pose)

    def save(self):
        trajectory = self.recorder.save(self.output / "source_trajectory.csv")
        (self.output / "cups.json").write_text(json.dumps(asdict(self.config), indent=2) + "\n")
        print(f"Exported {len(self.recorder.rows)} measured source poses to {trajectory}")
