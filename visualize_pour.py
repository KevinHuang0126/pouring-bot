"""Play a Drake source trajectory and SPlisHSPlasH VTK frames in Meshcat."""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from drake_trajectory_export import PoseInterpolation, load_trajectory


def read_vtk_points(path):
    """Read the big-endian POINTS block of SPlisHSPlasH's binary legacy VTK."""
    with path.open("rb") as stream:
        headers = [stream.readline() for _ in range(5)]
        match = re.fullmatch(rb"POINTS (\d+) (float|double)\s*", headers[-1])
        if headers[2].strip() != b"BINARY" or match is None:
            raise ValueError(f"Expected a binary SPlisHSPlasH VTK file: {path}")
        size = int(match[1])
        scalar = ">f4" if match[2] == b"float" else ">f8"
        data = np.fromfile(stream, dtype=scalar, count=3 * size)
    if data.size != 3 * size or not np.isfinite(data).all():
        raise ValueError(f"Invalid particle positions in {path}")
    return data.astype(np.float32).reshape(-1, 3).T


def load_frames(output, fps):
    timestamp_file = output / "fluid_frames.csv"
    if timestamp_file.is_file():
        with timestamp_file.open(newline="") as stream:
            rows = [
                (int(row["frame"]), float(row["time_s"]), float(row["source_pose_time_s"]))
                for row in csv.DictReader(stream)
            ]
    else:
        # Earlier exports only stored a sequential index and the scene FPS.
        print("Using nominal VTK frame times for this older export.", flush=True)
        indices = sorted(int(p.stem.rsplit("_", 1)[-1])
                         for p in (output / "vtk").glob("ParticleData_Fluid_*.vtk"))
        rows = [(i, (i - 1) / fps, (i - 1) / fps) for i in indices]
    if not rows:
        raise ValueError(f"No fluid VTK frames in {output / 'vtk'}")
    return [
        (t, pose_time, read_vtk_points(output / "vtk" / f"ParticleData_Fluid_{i}.vtk"))
        for i, t, pose_time in rows
    ]


def source_poses(trajectory, pose_times):
    samples = load_trajectory(trajectory).get("source", [])
    if len(samples) < 2:
        raise ValueError("Expected at least two source pose samples")
    times, values = zip(*samples)
    values = np.asarray(values)
    interpolation = PoseInterpolation(times, values[:, :3], values[:, [4, 5, 6, 3]])
    xyz, rotations = interpolation.pose(pose_times)
    return list(zip(rotations, xyz))


def add_robot(meshcat):
    from pydrake.all import (
        AddMultibodyPlantSceneGraph, DiagramBuilder, MeshcatVisualizer,
        Parser, RigidTransform, RollPitchYaw,
    )

    builder = DiagramBuilder()
    plant, graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    # The same iiwa7 station model used by shasha/integrated-pouring and
    # wyatt/randomized-grasping-lift. Place its base relative to the SPH scene.
    arm, = parser.AddModels(
        url="package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"
    )
    gripper, = parser.AddModels(url="package://drake_models/wsg_50_description/sdf/schunk_wsg_50.sdf")
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("iiwa_link_0", arm),
        RigidTransform([-0.382, -0.382, -0.46]),
    )
    wrist = plant.GetFrameByName("iiwa_link_7", arm)
    plant.WeldFrames(wrist, plant.GetFrameByName("body", gripper),
                     RigidTransform(RollPitchYaw(np.pi / 2, 0, np.pi / 2), [0, 0, 0.09]))
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    fingers = [plant.GetJointByName(f"{side}_finger_sliding_joint", gripper).position_start()
               for side in ("left", "right")]
    return plant, wrist, diagram, context, fingers


def robot_positions(plant, wrist, context, fingers, poses, grip_height, grip_radius):
    from pydrake.all import InverseKinematics, RotationMatrix, Solve

    plant_context = plant.GetMyMutableContextFromRoot(context)
    previous = plant.GetPositions(plant_context).copy()
    previous[:7] = [0, 0.2, 0, -1.4, 0, 1.3, 0]
    opening = min(grip_radius, 0.055)
    previous[fingers] = [-opening, opening]
    azimuth_rotation = Rotation.from_euler("z", np.pi / 4).as_matrix()
    tool_rotation = azimuth_rotation @ Rotation.from_euler("y", np.pi / 2).as_matrix()
    wrist_offset = azimuth_rotation @ [-0.155, 0, grip_height]
    solutions = []
    for rotation, origin in poses:
        target = origin + rotation @ wrist_offset
        ik = InverseKinematics(plant, plant_context)
        ik.AddPositionConstraint(wrist, np.zeros(3), plant.world_frame(),
                                 target - 0.001, target + 0.001)
        ik.AddOrientationConstraint(plant.world_frame(), RotationMatrix(rotation @ tool_rotation),
                                    wrist, RotationMatrix(), 0.025)
        program = ik.prog()
        program.AddBoundingBoxConstraint(previous[fingers], previous[fingers], ik.q()[fingers])
        program.AddQuadraticErrorCost(np.eye(len(previous)), previous, ik.q())
        program.SetInitialGuess(ik.q(), previous)
        result = Solve(program)
        if not result.is_success():
            raise RuntimeError("The source trajectory exceeds this iiwa playback setup's reach")
        previous = result.GetSolution(ik.q())
        solutions.append(previous)
    return plant_context, solutions


def play(trajectory, output, scene_path, *, no_robot=False, once=False):
    from pydrake.all import (
        Meshcat, PointCloud, ReadObjToTriangleSurfaceMesh, Rgba, RigidTransform, RotationMatrix,
    )

    scene = json.loads(scene_path.read_text())
    frames = load_frames(output, scene["Configuration"]["dataExportFPS"])
    if (output / "spill_report.json").is_file():
        report = json.loads((output / "spill_report.json").read_text())
        hold_source = report.get("source_held_fixed", False)
    else:
        hold_source = False
    poses = source_poses(trajectory, [0 if hold_source else frame[1] for frame in frames])
    meshcat = Meshcat()
    if no_robot:
        target = (poses[0][1] + np.asarray(scene["RigidBodies"][1]["translation"])) / 2
        target[2] += 0.09
        meshcat.SetCameraPose(target + [0.40, -0.60, 0.28], target)
    else:
        meshcat.SetCameraPose([0.75, -1.10, 0.65], [-0.17, -0.12, 0.12])
    point_size = 3 * scene["Configuration"].get("particleRadius", 0.003)
    cup_meshes = []
    for name, body, color in zip(
        ("source", "receiver"), scene["RigidBodies"],
        (Rgba(0.95, 0.55, 0.18, 0.65), Rgba(0.65, 0.72, 0.77, 0.40)),
    ):
        mesh_path = scene_path.parent / body["geometryFile"]
        if not mesh_path.is_file():
            mesh_path = scene_path.parent / mesh_path.name
        mesh = ReadObjToTriangleSurfaceMesh(mesh_path, np.asarray(body["scale"], dtype=float))
        cup_meshes.append(mesh)
        meshcat.SetObject(f"/cups/{name}", mesh, color)
    receiver = scene["RigidBodies"][1]
    axis = np.asarray(receiver["rotationAxis"])
    angle = receiver["rotationAngle"]
    receiver_rotation = (Rotation.from_rotvec(angle * axis / np.linalg.norm(axis)).as_matrix()
                         if angle else np.eye(3))
    meshcat.SetTransform("/cups/receiver", RigidTransform(
        RotationMatrix(receiver_rotation), receiver["translation"],
    ))

    robot = None
    if not no_robot:
        try:
            plant, wrist, diagram, context, fingers = add_robot(meshcat)
            vertices = np.array([cup_meshes[0].vertex(i) for i in range(cup_meshes[0].num_vertices())])
            radius = np.linalg.norm(vertices[:, :2], axis=1).max()
            height = vertices[:, 2].max()
            plant_context, joints = robot_positions(
                plant, wrist, context, fingers, poses, height / 2, radius,
            )
            robot = plant, diagram, context, plant_context, joints
        except RuntimeError as exc:
            print(f"Robot playback unavailable: {exc}. Showing cups and fluid.", flush=True)

    print(f"Meshcat viewer: {meshcat.web_url()} ({len(frames)} fluid frames, "
          f"robot {'enabled' if robot else 'disabled'})", flush=True)
    while True:
        start = time.monotonic()
        for i, ((stamp, pose_time, xyz), (rotation, origin)) in enumerate(zip(frames, poses)):
            time.sleep(max(0, start + stamp - frames[0][0] - time.monotonic()))
            meshcat.SetTransform("/cups/source", RigidTransform(RotationMatrix(rotation), origin))
            cloud = PointCloud(xyz.shape[1])
            cloud.mutable_xyzs()[:] = xyz
            meshcat.SetObject("/fluid", cloud, point_size=point_size, rgba=Rgba(0.07, 0.48, 0.95))
            if robot:
                plant, diagram, context, plant_context, joints = robot
                context.SetTime(pose_time)
                plant.SetPositions(plant_context, joints[i])
                diagram.ForcedPublish(context)
        if once:
            print(f"Played {len(frames)} frames.", flush=True)
            return
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, help="Defaults to the trajectory copied into the scene")
    parser.add_argument("--fluid-output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="Scene JSON for older exports")
    parser.add_argument("--no-robot", action="store_true", help="Show cups and fluid only")
    parser.add_argument("--once", action="store_true", help="Play once, then close the server")
    args = parser.parse_args()
    scene = args.scene or args.fluid_output / "scene" / "scene_animated.json"
    try:
        trajectory = args.trajectory or scene.parent / "source_trajectory.csv"
        play(trajectory, args.fluid_output, scene, no_robot=args.no_robot, once=args.once)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
