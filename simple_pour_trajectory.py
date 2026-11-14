"""Ballistic pouring surrogate with fill-dependent exit speed (SI units)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

GRAVITY = 9.81
TABLE_TOP = 0.20
TABLE_CENTER = (0.60, 0.00, 0.17)
PACKAGE_XML = Path(__file__).resolve().parent / 'models' / 'package.xml'

@dataclass(frozen=True)
class VesselSpec:
    name: str
    height: float
    mouth_radius: float
    fill_display_radius: float

VESSELS = {
    'test_tube': VesselSpec('test_tube', .12, .012, .007),
    'erlenmeyer': VesselSpec('erlenmeyer', .16, .016, .012),
    'boiling_flask': VesselSpec('boiling_flask', .17, .018, .014),
}

@dataclass(frozen=True)
class PourResult:
    active: bool
    points_W: np.ndarray
    mouth_W: np.ndarray
    tilt_deg: float
    onset_deg: float
    launch_velocity_W: np.ndarray
    exit_speed_m_s: float = 0.0


def _pose_components(X_WV):
    if hasattr(X_WV, 'rotation') and hasattr(X_WV, 'translation'):
        return np.asarray(X_WV.rotation().matrix(), dtype=float), np.asarray(X_WV.translation(), dtype=float)
    X = np.asarray(X_WV, dtype=float)
    if X.shape != (4, 4):
        raise ValueError('X_WV must be a Drake RigidTransform or 4x4 matrix')
    return X[:3, :3], X[:3, 3]


def compute_pour_trajectory(X_WV, vessel, fill_fraction, *,
                            linear_velocity_W=(0., 0., 0.),
                            angular_velocity_W=(0., 0., 0.),
                            surface_z=TABLE_TOP, sample_dt=.015, max_time=1.5):
    """Return instantaneous jet; speed is relative to the moving outlet."""
    if not 0 <= fill_fraction <= 1:
        raise ValueError('fill_fraction must lie in [0, 1]')
    if sample_dt <= 0 or max_time <= 0:
        raise ValueError('sample_dt and max_time must be positive')
    R, p = _pose_components(X_WV)
    v = np.asarray(linear_velocity_W, dtype=float).reshape(3)
    omega = np.asarray(angular_velocity_W, dtype=float).reshape(3)
    axis_W = R[:, 2]
    tilt_deg = float(np.degrees(np.arccos(np.clip(axis_W[2], -1., 1.))))
    onset_deg = 25. + 35. * (1. - fill_fraction)
    rim_xy = (R.T @ np.array([0., 0., -1.]))[:2]
    rim_norm = np.linalg.norm(rim_xy)
    rim_xy = rim_xy / rim_norm if rim_norm > 1e-9 else np.array([1., 0.])
    mouth_V = np.array([vessel.mouth_radius * rim_xy[0],
                        vessel.mouth_radius * rim_xy[1], vessel.height])
    mouth_W = p + R @ mouth_V
    if fill_fraction <= 0 or tilt_deg <= onset_deg or mouth_W[2] <= surface_z:
        return PourResult(False, np.empty((0, 3)), mouth_W, tilt_deg, onset_deg, np.zeros(3))
    horizontal = np.array([axis_W[0], axis_W[1], 0.])
    norm = np.linalg.norm(horizontal)
    # At exactly inverted orientation the horizontal direction is undefined.
    # Use a vertical downward jet in that limiting case.
    if norm < 1e-9:
        release_dir = np.array([0., 0., -1.])
    else:
        release_dir = horizontal / norm + np.array([0., 0., -.15])
        release_dir /= np.linalg.norm(release_dir)
    head_m = max(.005, fill_fraction * vessel.height * .6)
    activation = np.clip((tilt_deg - onset_deg) / 35., 0., 1.)
    speed = float(.75 * np.sqrt(2 * GRAVITY * head_m) * np.sqrt(activation))
    launch_velocity_W = v + np.cross(omega, R @ mouth_V) + speed * release_dir
    dz = mouth_W[2] - surface_z
    t_hit = (launch_velocity_W[2] + np.sqrt(launch_velocity_W[2]**2 + 2 * GRAVITY * dz)) / GRAVITY
    duration = min(float(t_hit), float(max_time))
    times = np.append(np.arange(0., duration, sample_dt), duration)
    points = (mouth_W[None, :] + times[:, None] * launch_velocity_W[None, :]
              + .5 * times[:, None]**2 * np.array([0., 0., -GRAVITY]))
    return PourResult(True, points, mouth_W, tilt_deg, onset_deg, launch_velocity_W, speed)


def target_opening_error(result, target_center_xy, target_opening_z, target_radius):
    """Check the descending jet's intersection with a circular opening."""
    if target_radius <= 0:
        raise ValueError('target_radius must be positive')
    if not result.active:
        return False, float('inf')
    pts = result.points_W
    for i in range(1, len(pts)):
        z0, z1 = pts[i-1, 2], pts[i, 2]
        if z0 >= target_opening_z >= z1 and z1 < z0:
            alpha = (target_opening_z - z0) / (z1 - z0)
            xy = pts[i-1, :2] + alpha * (pts[i, :2] - pts[i-1, :2])
            miss = float(np.linalg.norm(xy - np.asarray(target_center_xy)))
            return miss <= target_radius, miss
    return False, float('inf')


def render_pour_trajectory(meshcat, path, result):
    from pydrake.all import Rgba
    meshcat.Delete(path)
    if result.active:
        meshcat.SetLine(path, result.points_W.T, line_width=4., rgba=Rgba(.05, .50, .95, 1.))


def build_demo():
    """Original static three-vessel Meshcat demonstration."""
    from pydrake.all import (AddMultibodyPlantSceneGraph, Cylinder, DiagramBuilder,
        Meshcat, MeshcatVisualizer, Parser, Rgba, RigidTransform, RollPitchYaw, Simulator)
    examples = [
        ('test_tube', (.31, -.23, .54), (0, 63, 0), .85),
        ('erlenmeyer', (.46, .11, .55), (0, 54, 35), .65),
        ('boiling_flask', (.68, -.12, .59), (-52, 0, -25), .75),
    ]
    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(str(PACKAGE_XML))
    table = parser.AddModelsFromUrl('package://pouring_bot/table.sdf')[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName('body', table), RigidTransform(TABLE_CENTER))
    receiver = parser.AddModelsFromUrl('package://pouring_bot/beaker.sdf')[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName('body', receiver), RigidTransform([.72, .14, TABLE_TOP]))
    instances = []
    for name, pos, rpy, fraction in examples:
        model = parser.AddModelsFromUrl(f'package://pouring_bot/{name}.sdf')[0]
        pose = RigidTransform(RollPitchYaw(np.deg2rad(rpy)), pos)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName('body', model), pose)
        instances.append((name, model, fraction))
    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    diagram.ForcedPublish(simulator.get_context())
    pc = plant.GetMyContextFromRoot(simulator.get_context())
    for name, model, fraction in instances:
        spec = VESSELS[name]
        pose = plant.EvalBodyPoseInWorld(pc, plant.GetBodyByName('body', model))
        result = compute_pour_trajectory(pose, spec, fraction)
        render_pour_trajectory(meshcat, f'pouring/{name}/jet', result)
        height = max(.001, fraction * spec.height)
        path = f'pouring/{name}/schematic_fill'
        meshcat.SetObject(path, Cylinder(spec.fill_display_radius, height), Rgba(.12, .64, .96, .55))
        meshcat.SetTransform(path, pose @ RigidTransform([0., 0., height / 2]))
        print(f'{name}: fill={fraction:.0%}, tilt={result.tilt_deg:.1f} deg, exit speed={result.exit_speed_m_s:.3f} m/s')
    print('Meshcat:', meshcat.web_url())
    input('Press Enter to close.\n')

if __name__ == '__main__':
    build_demo()
