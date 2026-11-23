"""Generate a real SPH pouring scene and hollow OBJ boundaries from Drake dimensions.

The generated Bender2019 scene is the initial state for sph_drake_replay.py.
"""
import argparse
import csv
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


SOURCE_CUP = (0.035, 0.003, 0.110)
RECEIVER_CUP = (0.060, 0.004, 0.120)
RECEIVER_ORIGIN = (0.110, 0.0, 0.0)
CUP_SEGMENTS = 64
CUP_BOTTOM = 0.004


PARTICLE_RADIUS = 0.003


@dataclass(frozen=True)
class VesselProfile:
    """Outer and inner (z, radius) rings, from base to common open rim."""

    outer: tuple[tuple[float, float], ...]
    inner: tuple[tuple[float, float], ...]

    def __post_init__(self):
        for name in ("outer", "inner"):
            rings = tuple(tuple(float(v) for v in ring) for ring in getattr(self, name))
            if (len(rings) < 2 or any(len(r) != 2 for r in rings)
                    or not np.isfinite(rings).all()
                    or any(z < 0 or radius <= 0 for z, radius in rings)
                    or any(b[0] <= a[0] for a, b in zip(rings, rings[1:]))):
                raise ValueError(f"{name} needs finite (z, positive radius) rings with increasing z")
            object.__setattr__(self, name, rings)
        if not (self.outer[0][0] == 0 < self.inner[0][0]
                and self.outer[-1][0] == self.inner[-1][0]):
            raise ValueError("Vessel needs a base at z=0, a positive floor, and a common rim height")
        heights = sorted({z for z, _ in self.inner}
                         | {z for z, _ in self.outer if z >= self.inner[0][0]})
        if np.any(self.radius_at(heights) >= np.interp(heights, *np.asarray(self.outer).T)):
            raise ValueError("The inner cavity must remain strictly inside the outer wall")

    def radius_at(self, height):
        return np.interp(height, *np.asarray(self.inner).T)

    def contains(self, points):
        x, y, z = np.asarray(points).T
        half_segment = np.pi / CUP_SEGMENTS
        angle = np.arctan2(y, x) % (2 * half_segment) - half_segment
        return ((np.hypot(x, y) * np.cos(angle) < self.radius_at(z) * np.cos(half_segment))
                & (z > self.inner[0][0]) & (z < self.inner[-1][0]))


def cylindrical_profile(dimensions):
    inner, wall, height, bottom = dimensions
    return VesselProfile(((0, inner + wall), (height, inner + wall)),
                         ((bottom, inner), (height, inner)))


@dataclass(frozen=True)
class PouringConfig:
    """Cup dimensions and initial source-local fill bounds, in meters."""

    source_cup: tuple[float, float, float, float] = (*SOURCE_CUP, CUP_BOTTOM)
    receiver_cup: tuple[float, float, float, float] = (*RECEIVER_CUP, CUP_BOTTOM)
    receiver_origin: tuple[float, float, float] = RECEIVER_ORIGIN
    fill_start: tuple[float, float, float] = (-0.022, -0.022, 0.012)
    fill_end: tuple[float, float, float] = (0.022, 0.022, 0.080)

    particle_radius: float = PARTICLE_RADIUS
    max_time_step: float | None = None
    source_profile: VesselProfile | None = None
    receiver_profile: VesselProfile | None = None

    def __post_init__(self):
        for name, size in (("source_cup", 4), ("receiver_cup", 4),
                           ("receiver_origin", 3), ("fill_start", 3), ("fill_end", 3)):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != size or not all(map(math.isfinite, values)):
                raise ValueError(f"{name} requires {size} finite values in meters")
            object.__setattr__(self, name, values)

        for cup in (self.source_cup, self.receiver_cup):
            if min(cup) <= 0 or cup[3] >= cup[2]:
                raise ValueError("Cup dimensions must be positive with base thickness below rim height")

        radius = float(self.particle_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Particle radius must be positive and finite")
        object.__setattr__(self, "particle_radius", radius)
        if self.max_time_step is not None:
            step = float(self.max_time_step)
            if not math.isfinite(step) or step <= 0:
                raise ValueError("Maximum time step must be positive and finite")
            object.__setattr__(self, "max_time_step", step)

        # Cubic sampling needs round(extent / particle diameter) >= 3 per axis.
        min_extent = 5 * radius
        if any(hi - lo <= min_extent for lo, hi in zip(self.fill_start, self.fill_end)):
            raise ValueError(f"Initial fill must span over {min_extent:g} m on each axis")
        for name in ("source_profile", "receiver_profile"):
            profile = getattr(self, name)
            if isinstance(profile, dict):
                object.__setattr__(self, name, VesselProfile(**profile))
            elif profile is not None and not isinstance(profile, VesselProfile):
                raise ValueError(f"{name} must be a VesselProfile or an outer/inner mapping")

        profile = self.source_profile or cylindrical_profile(self.source_cup)
        lo, hi = self.fill_start, self.fill_end
        heights = [lo[2], hi[2], *(z for z, _ in profile.inner if lo[2] < z < hi[2])]
        corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in heights]
        if not profile.contains(corners).all():
            raise ValueError("Initial fill bounds must fit inside the source cavity")

DEFAULT_CONFIG = PouringConfig()


def write_cup(output, name, dimensions, default):
    # Native volume-map caches use the mesh basename. Preserve the existing
    # demo cache names and give each custom mesh its own stable name.
    if dimensions != default:
        digest = sha256(repr(dimensions).encode()).hexdigest()[:16]
        name = f"{name}_{digest}"
    path = output / f"{name}.obj"
    cup_mesh(path, *dimensions[:3], bottom=dimensions[3])
    return path


def cup_mesh(path, inner, wall, height, segments=CUP_SEGMENTS, bottom=CUP_BOTTOM):
    """Watertight, open-top cup wall and base; z=0 is outer base bottom."""
    profile_mesh(path, cylindrical_profile((inner, wall, height, bottom)), segments)


def profile_mesh(path, profile, segments=CUP_SEGMENTS):
    """Generate the closed solid shell with an uncapped cavity opening."""
    rings = (*profile.outer, *reversed(profile.inner))
    angles = [2 * math.pi * i / segments for i in range(segments)]
    vertices = [(radius * math.cos(a), radius * math.sin(a), z)
                for z, radius in rings for a in angles]
    center_inner = len(vertices)
    vertices.extend([(0.0, 0.0, profile.inner[0][0]), (0.0, 0.0, 0.0)])
    faces = []
    for ring in range(len(rings) - 1):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = ring * segments + i, ring * segments + j
            c, d = a + segments, b + segments
            faces.extend(((a, b, c), (b, d, c)))
    last = (len(rings) - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(((last + i, last + j, center_inner), (i, center_inner + 1, j)))
    with path.open('w') as f:
        f.write('# Hollow open-top cup, meters, Z up\n')
        for v in vertices:
            f.write('v %.9f %.9f %.9f\n' % v)
        for a,b,c in faces:
            f.write(f'f {a+1} {b+1} {c+1}\n')


def write_boundary(output, name, profile, dimensions, default):
    if profile is None:
        return write_cup(output, name, dimensions, default)
    digest = sha256(repr(profile).encode()).hexdigest()[:16]
    path = output / f"{name}_profile_{digest}.obj"
    profile_mesh(path, profile)
    return path


def prepare_scene(trajectory, output, template, *, settings=DEFAULT_CONFIG, export_fps=30.0, cfl_factor=None):
    if not math.isfinite(export_fps) or export_fps < 0:
        raise ValueError("Export FPS must be finite and nonnegative")
    if cfl_factor is not None and (not math.isfinite(cfl_factor) or cfl_factor <= 0):
        raise ValueError("CFL factor must be positive and finite")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory = Path(trajectory).resolve()
    with trajectory.open(newline='') as f:
        rows = list(csv.DictReader(f))
    source = [r for r in rows if r['body'] == 'source']
    if len(source) < 2:
        raise ValueError('Expected at least two Drake source pose samples')
    times = [float(r['time_s']) for r in source]
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('Trajectory times must be strictly increasing')
    (output/'source_trajectory.csv').write_text(trajectory.read_text())
    source_mesh = write_boundary(output, "source", settings.source_profile,
                                 settings.source_cup, DEFAULT_CONFIG.source_cup)
    receiver_mesh = write_boundary(output, "receiver", settings.receiver_profile,
                                   settings.receiver_cup, DEFAULT_CONFIG.receiver_cup)
    template = json.loads(Path(template).read_text())
    config = template['Configuration']
    config.update({'particleRadius': settings.particle_radius, 'gravitation': [0,0,-9.81],
                   'boundaryHandlingMethod': 2,  # Bender2019 volume maps
                   'enableVTKExport': export_fps > 0, 'dataExportFPS': export_fps or 30.0,
                   'particleAttributes': 'velocity;density'})
    if cfl_factor is not None:
        config['cflFactor'] = cfl_factor
    if settings.max_time_step is not None:
        config['timeStepSize'] = settings.max_time_step
        config['cflMaxTimeStepSize'] = settings.max_time_step
    first = source[0]
    origin = [float(first[k]) for k in ('x_m','y_m','z_m')]
    config['cameraPosition'] = [0.4, -0.5, 0.35]
    config['cameraLookat'] = [0.06, 0, 0.12]
    scene = {'Configuration':config, 'Materials':template['Materials'],
             'RigidBodies':[
                 {'geometryFile':str(source_mesh), 'translation':origin,
                  'rotationAxis':[0,1,0], 'rotationAngle':0, 'scale':[1,1,1],
                  'isDynamic':False, 'isAnimated':False,
                  'mapResolution':[48,48,48]},
                 {'geometryFile':str(receiver_mesh),
                  'translation':settings.receiver_origin, 'rotationAxis':[0,1,0],
                  'rotationAngle':0, 'scale':[1,1,1], 'isDynamic':False,
                  'mapResolution':[48,48,48]}],
             'FluidBlocks':[{'denseMode':0,
                             'start':[p + d for p, d in zip(origin, settings.fill_start)],
                             'end':[p + d for p, d in zip(origin, settings.fill_end)]}]}
    (output/'scene_static.json').write_text(json.dumps(scene,indent=2)+'\n')
    (output/'trial_manifest.json').write_text(json.dumps({
        'trajectory': 'source_trajectory.csv', 'scene':'scene_static.json',
        'duration_s':times[-1]-times[0],
        'status':'PREPARATION_ONLY_NO_MOVING_BOUNDARY_REPLAY',
        'note':'Static scene is a geometry/fluid smoke test, not a pouring result.'},indent=2)+'\n')
    return output / 'scene_static.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trajectory', default='drake_vessel_trajectory.csv')
    p.add_argument('--output', default='sph_pouring_trial')
    p.add_argument('--template', default='DoubleDamBreak_export.json')
    p.add_argument('--config', type=Path, help='JSON vessel geometry and fill')
    args = p.parse_args()
    settings = PouringConfig(**json.loads(args.config.read_text())) if args.config else DEFAULT_CONFIG
    scene = prepare_scene(args.trajectory, args.output, args.template, settings=settings)
    print('Prepared', scene)
    print('Replay vessel motion with sph_drake_replay.py.')

if __name__ == '__main__':
    main()
