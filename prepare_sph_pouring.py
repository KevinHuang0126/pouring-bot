"""Generate a real SPH pouring scene and hollow OBJ boundaries from Drake dimensions.

The generated Bender2019 scene is the initial state for sph_drake_replay.py.
"""
import argparse
import json
import math
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from drake_trajectory_export import load_trajectory


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
    segments: int = field(default=CUP_SEGMENTS, repr=False)

    def __post_init__(self):
        if not isinstance(self.segments, int) or self.segments < 3:
            raise ValueError("A profile requires at least three angular segments")
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
        half_segment = np.pi / self.segments
        angle = np.arctan2(y, x) % (2 * half_segment) - half_segment
        return ((np.hypot(x, y) * np.cos(angle) < self.radius_at(z) * np.cos(half_segment))
                & (z > self.inner[0][0]) & (z < self.inner[-1][0]))

    def capacity_m3(self, height=None):
        """Integrate the actual faceted cavity up to a local Z height."""
        z, r = np.asarray(self.inner).T
        height = z[-1] if height is None else np.clip(height, z[0], z[-1])
        end = np.minimum(z[1:], np.maximum(z[:-1], height))
        end_radius = np.interp(end, z, r)
        area_factor = self.segments * np.sin(2 * np.pi / self.segments) / 2
        return float(area_factor / 3 * np.sum(
            (end - z[:-1]) * (r[:-1]**2 + r[:-1] * end_radius + end_radius**2)
        ))


def initial_profile_fill(profile, fraction, particle_radius):
    """Seed a cubic lattice, quantizing volume by particle count."""
    spacing = 2 * particle_radius
    narrow = min(r for _, r in profile.inner) <= 5 * spacing
    # The volume-map boundary samples contribute more strongly in narrow
    # cavities. Leave a full lattice spacing there at rest.
    clearance = particle_radius * (2 if narrow else 1.5)
    capacity = profile.capacity_m3()
    target = fraction * capacity
    bottom, rim = profile.inner[0][0], profile.inner[-1][0]
    surface = brentq(lambda z: profile.capacity_m3(z) - target, bottom, rim)
    extent = int(np.ceil(max(r for _, r in profile.inner) / spacing))
    indices = np.arange(-extent, extent + 1)
    zs = np.arange(bottom + 3 * particle_radius, rim, spacing)
    half_segment = np.pi / profile.segments
    limits = [
        min(profile.radius_at([z - clearance, z + clearance,
                               *(h for h, _ in profile.inner if z - clearance < h < z + clearance)]))
        * np.cos(half_segment) for z in zs
    ]
    count = int(np.floor(target / spacing**3 + .5))
    if not count or not len(zs):
        raise ValueError("Requested fill is too small for the configured particle spacing")

    def lattice(offset):
        x, y = np.meshgrid(*((indices + phase) * spacing for phase in offset))
        xy = np.column_stack((x.ravel(), y.ravel()))
        phi = np.arctan2(xy[:, 1], xy[:, 0]) % (2 * half_segment) - half_segment
        radial = np.linalg.norm(xy, axis=1) * np.cos(phi) + clearance
        inside = radial[None, :] < np.asarray(limits)[:, None]
        cumulative = np.cumsum(inside.sum(axis=1))
        shortfall = max(0, count - cumulative[-1])
        top = min(int(np.searchsorted(cumulative, count)), len(zs) - 1)
        return (shortfall, top), xy, inside

    # Equal-spacing grids can differ in how many centers fit near a curved
    # wall. Keep the symmetric half-cell grid unless it is short of volume.
    packing = lattice((.5, .5))
    if packing[0][0] and not narrow:
        alternatives = ((0., 0.), (0., .5), (.25, .25))
        packing = min((packing, *map(lattice, alternatives)), key=lambda candidate: candidate[0])
    (_, top), xy, inside = packing
    remaining = count
    layers = []
    for z, layer in zip(zs[:top + 1], inside):
        candidates = xy[layer]
        if len(candidates) > remaining:
            # Populate the incomplete upper layer without a directional bias.
            candidates = candidates[np.random.default_rng(0).permutation(len(candidates))[:remaining]]
        if len(candidates):
            layers.append(np.column_stack((candidates, np.full(len(candidates), z))))
            remaining -= len(candidates)
    if not layers:
        raise ValueError("The initial fill lattice does not fit inside the cavity")
    points = np.concatenate(layers)
    initialized = len(points) * spacing**3
    if remaining:
        raise ValueError(
            f"Cannot seed {target * 1e6:.3f} mL at the current spacing and wall clearance. "
            f"The lattice fits {initialized * 1e6:.3f} mL. Reduce the fill fraction."
        )
    return points, {
        "usable_capacity_ml": capacity * 1e6,
        "requested_fill_fraction": fraction,
        "target_volume_ml": target * 1e6,
        "initialized_particle_count": len(points),
        "estimated_initialized_volume_ml": initialized * 1e6,
        "estimated_realized_fill_fraction": initialized / capacity,
        "volume_error_ml": (initialized - target) * 1e6,
        "target_surface_height_m": surface,
        "highest_particle_center_m": float(points[:, 2].max()),
        "minimum_wall_clearance_m": clearance,
        "nominal_particle_volume_ml": spacing**3 * 1e6,
    }


def write_particle_samples(path, points):
    """Write resting Partio GEO samples for native FluidModels."""
    count = len(points)
    with path.open("w") as stream:
        stream.write(f"PGEOMETRY V5\nNPoints {count} NPrims 1\n"
                     "NPointGroups 0 NPrimGroups 0\n"
                     "NPointAttrib 1 NVertexAttrib 0 NPrimAttrib 1 NAttrib 0\n"
                     "PointAttrib\nvelocity 3 vector 0 0 0\n")
        np.savetxt(stream, points, fmt="%.9f %.9f %.9f 1 (0 0 0)")
        stream.write("PrimitiveAttrib\ngenerator 1 index 1 papi\n")
        stream.write(f"Part {count} " + " ".join(map(str, range(count))))
        stream.write(" [0]\nbeginExtra\nendExtra\n")


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
    receiver_quaternion_wxyz: tuple[float, ...] = (1., 0., 0., 0.)
    fill_fraction: float | None = None

    def __post_init__(self):
        for name, size in (("source_cup", 4), ("receiver_cup", 4),
                           ("receiver_origin", 3), ("fill_start", 3), ("fill_end", 3)):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != size or not all(map(math.isfinite, values)):
                raise ValueError(f"{name} requires {size} finite values in meters")
            object.__setattr__(self, name, values)

        q = np.asarray(self.receiver_quaternion_wxyz, dtype=float)
        if q.shape != (4,) or not np.isfinite(np.linalg.norm(q)) or not np.linalg.norm(q):
            raise ValueError("Receiver quaternion must have four finite nonzero-norm components")
        object.__setattr__(self, "receiver_quaternion_wxyz", tuple(q / np.linalg.norm(q)))

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

        for name in ("source_profile", "receiver_profile"):
            profile = getattr(self, name)
            if isinstance(profile, dict):
                object.__setattr__(self, name, VesselProfile(**profile))
            elif profile is not None and not isinstance(profile, VesselProfile):
                raise ValueError(f"{name} must be a VesselProfile or an outer/inner mapping")

        if self.fill_fraction is not None:
            fraction = float(self.fill_fraction)
            if not 0 < fraction <= 1:
                raise ValueError("Initial fill fraction must be greater than zero and at most one")
            object.__setattr__(self, "fill_fraction", fraction)
            return
        # Cubic block sampling needs round(extent / particle diameter) >= 3.
        min_extent = 5 * radius
        if any(hi - lo <= min_extent for lo, hi in zip(self.fill_start, self.fill_end)):
            raise ValueError(f"Initial fill must span over {min_extent:g} m on each axis")
        profile = self.source_cavity()
        lo, hi = self.fill_start, self.fill_end
        heights = [lo[2], hi[2], *(z for z, _ in profile.inner if lo[2] < z < hi[2])]
        corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in heights]
        if not profile.contains(corners).all():
            raise ValueError("Initial fill bounds must fit inside the source cavity")

    def source_cavity(self):
        return self.source_profile or cylindrical_profile(self.source_cup)

    def receiver_rotation(self):
        return Rotation.from_quat(self.receiver_quaternion_wxyz, scalar_first=True)


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


def profile_mesh(path, profile, segments=None):
    """Generate the closed solid shell with an uncapped cavity opening."""
    segments = profile.segments if segments is None else segments
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
    key = repr(profile) if profile.segments == CUP_SEGMENTS else repr((profile, profile.segments))
    digest = sha256(key.encode()).hexdigest()[:16]
    path = output / f"{name}_profile_{digest}.obj"
    profile_mesh(path, profile)
    return path


def scene_rotation(rotation):
    vector = rotation.as_rotvec()
    angle = np.linalg.norm(vector)
    return {"rotationAngle": angle, "rotationAxis": (vector / angle).tolist() if angle else [0, 1, 0]}


def prepare_scene(trajectory, output, template, *, settings=DEFAULT_CONFIG, export_fps=30.0, cfl_factor=None):
    if not math.isfinite(export_fps) or export_fps < 0:
        raise ValueError("Export FPS must be finite and nonnegative")
    if cfl_factor is not None and (not math.isfinite(cfl_factor) or cfl_factor <= 0):
        raise ValueError("CFL factor must be positive and finite")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory = Path(trajectory).resolve()
    source = load_trajectory(trajectory).get("source", [])
    if len(source) < 2:
        raise ValueError("Expected at least two Drake source pose samples")
    times = [t for t, _ in source]
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
    first = source[0][1]
    origin = first[:3]
    source_rotation = Rotation.from_quat(first[3:], scalar_first=True)
    local_fill = source_rotation.magnitude() > 1e-12
    fill_origin = np.zeros(3) if local_fill else origin
    config['cameraPosition'] = [0.4, -0.5, 0.35]
    config['cameraLookat'] = [0.06, 0, 0.12]
    scene = {'Configuration':config, 'Materials':template['Materials'],
             'RigidBodies':[
                 {'geometryFile':str(source_mesh), 'translation':origin.tolist(),
                  **scene_rotation(source_rotation), 'scale':[1,1,1],
                  'isDynamic':False, 'isAnimated':False,
                  'mapResolution':[48,48,48]},
                 {'geometryFile':str(receiver_mesh),
                  'translation':settings.receiver_origin, **scene_rotation(settings.receiver_rotation()), 'scale':[1,1,1], 'isDynamic':False,
                  'mapResolution':[48,48,48]}],
             }
    if settings.fill_fraction is not None:
        points, metrics = initial_profile_fill(settings.source_cavity(), settings.fill_fraction,
                                               settings.particle_radius)
        samples = output / "initial_fluid.geo"
        write_particle_samples(samples, points)
        scene["FluidModels"] = [{
            "id": "Fluid", "particleFile": samples.name, "translation": origin.tolist(),
            **scene_rotation(source_rotation),
        }]
        scene["initialFill"] = metrics
    else:
        scene["FluidBlocks"] = [{
            "denseMode": 0,
            "start": [p + d for p, d in zip(fill_origin, settings.fill_start)],
            "end": [p + d for p, d in zip(fill_origin, settings.fill_end)],
        }]
        if local_fill:
            scene["initialFillFrame"] = "source"
    (output/'scene_static.json').write_text(json.dumps(scene,indent=2)+'\n')
    (output/'trial_manifest.json').write_text(json.dumps({
        'trajectory': 'source_trajectory.csv', 'scene':'scene_static.json',
        'duration_s':times[-1]-times[0], 'input_time_origin_s':times[0],
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
