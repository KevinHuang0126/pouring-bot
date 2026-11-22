import json
from pathlib import Path

import numpy as np
import pytest
from pydrake.all import RigidTransform, RotationMatrix

from drake_trajectory_export import load_trajectory
from prepare_sph_pouring import PouringConfig, prepare_scene
from station_sph_export import StationSphExport


def test_station_samples_measured_poses_and_reuses_mesh_dimensions(tmp_path):
    class Simulation:
        time = 5.0

        def get_context(self):
            return self

        def get_time(self):
            return self.time

        def EvalBodyPoseInWorld(self, context, body):
            return poses[body]

    root = Path(__file__).resolve().parents[1]
    simulation = Simulation()
    poses = {"tube": RigidTransform([0.42, -0.16, 0.22]),
             "beaker": RigidTransform([0.72, 0.14, 0.20])}
    output = tmp_path / "station"
    export = StationSphExport(output, root / "models")
    export.start(simulation, simulation, None, "tube", "beaker")
    with pytest.raises(ValueError, match="strictly increasing"):
        export.record_pose()
    simulation.time = 5.125
    poses["tube"] = RigidTransform(RotationMatrix.MakeYRotation(0.1), [0.43, -0.16, 0.24])
    export.record_pose()
    export.save()

    trajectory = output / "source_trajectory.csv"
    samples = load_trajectory(trajectory)["source"]
    assert [stamp for stamp, _ in samples] == [0.0, 0.125]
    np.testing.assert_allclose(samples[0][1], [0.42, -0.16, 0.22, 1, 0, 0, 0])
    np.testing.assert_allclose(samples[1][1][:3], [0.43, -0.16, 0.24])
    assert samples[1][1][5] == pytest.approx(np.sin(0.05))

    config = PouringConfig(**json.loads((output / "cups.json").read_text()))
    scene_path = prepare_scene(trajectory, tmp_path / "sph",
                               root / "DoubleDamBreak_export.json", settings=config)
    scene = json.loads(scene_path.read_text())
    source, receiver = scene["RigidBodies"]
    assert source["translation"] == [0.42, -0.16, 0.22]
    assert receiver["translation"] == [0.72, 0.14, 0.20]
    assert scene["Configuration"]["particleRadius"] == config.particle_radius
    assert scene["Configuration"]["cflMaxTimeStepSize"] == config.max_time_step
    for body, name in ((source, "test_tube"), (receiver, "beaker")):
        assert Path(body["geometryFile"]).read_bytes() == (output / f"{name}.obj").read_bytes()
