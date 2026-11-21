"""Generate a real SPH pouring scene and hollow OBJ boundaries from Drake dimensions.

The generated Bender2019 scene is the initial state for sph_drake_replay.py.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def cup_mesh(path, inner, wall, height, segments=64, bottom=0.004):
    """Watertight, open-top cup wall and base; z=0 is outer base bottom."""
    outer = inner + wall
    rings = [(outer, 0.0), (outer, height), (inner, height),
             (inner, bottom), (0.0, bottom), (0.0, 0.0)]
    vertices = []
    # Four nonzero-radius rings followed by bottom and inner center vertices.
    for radius, z in rings[:4]:
        vertices.extend((radius*math.cos(2*math.pi*i/segments),
                         radius*math.sin(2*math.pi*i/segments), z)
                        for i in range(segments))
    center_inner = len(vertices)
    vertices.append((0.0, 0.0, bottom))
    center_outer = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    faces = []
    for ring in range(3):
        for i in range(segments):
            j = (i+1) % segments
            a, b = ring*segments+i, ring*segments+j
            c, d = (ring+1)*segments+i, (ring+1)*segments+j
            faces.extend(((a,b,c), (b,d,c)))
    for i in range(segments):
        j = (i+1) % segments
        faces.append((3*segments+i, 3*segments+j, center_inner))
        faces.append((i, center_outer, j))
    with path.open('w') as f:
        f.write('# Hollow open-top cup, meters, Z up\n')
        for v in vertices:
            f.write('v %.9f %.9f %.9f\n' % v)
        for a,b,c in faces:
            f.write(f'f {a+1} {b+1} {c+1}\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trajectory', default='drake_vessel_trajectory.csv')
    p.add_argument('--output', default='sph_pouring_trial')
    p.add_argument('--template', default='DoubleDamBreak_export.json')
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory = Path(args.trajectory).resolve()
    with trajectory.open(newline='') as f:
        rows = list(csv.DictReader(f))
    source = [r for r in rows if r['body'] == 'source']
    if len(source) < 2:
        raise ValueError('Expected at least two Drake source pose samples')
    times = [float(r['time_s']) for r in source]
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('Trajectory times must be strictly increasing')
    (output/'source_trajectory.csv').write_text(trajectory.read_text())
    cup_mesh(output/'source.obj', 0.035, 0.003, 0.110)
    cup_mesh(output/'receiver.obj', 0.060, 0.004, 0.120)
    template = json.loads(Path(args.template).read_text())
    config = template['Configuration']
    config.update({'particleRadius': 0.003, 'gravitation': [0,0,-9.81],
                   'boundaryHandlingMethod': 2,  # Bender2019 volume maps
                   'enableVTKExport': True, 'dataExportFPS': 30.0,
                   'particleAttributes': 'velocity;density'})
    first = source[0]
    origin = [float(first[k]) for k in ('x_m','y_m','z_m')]
    config['cameraPosition'] = [0.4, -0.5, 0.35]
    config['cameraLookat'] = [0.06, 0, 0.12]
    scene = {'Configuration':config, 'Materials':template['Materials'],
             'RigidBodies':[
                 {'geometryFile':str(output/'source.obj'), 'translation':origin,
                  'rotationAxis':[0,1,0], 'rotationAngle':0, 'scale':[1,1,1],
                  'isDynamic':False, 'isAnimated':False,
                  'mapResolution':[48,48,48]},
                 {'geometryFile':str(output/'receiver.obj'),
                  'translation':[0.110,0,0], 'rotationAxis':[0,1,0],
                  'rotationAngle':0, 'scale':[1,1,1], 'isDynamic':False,
                  'mapResolution':[48,48,48]}],
             'FluidBlocks':[{'denseMode':0,
                             'start':[origin[0]-0.022, origin[1]-0.022, origin[2]+0.012],
                             'end':[origin[0]+0.022, origin[1]+0.022, origin[2]+0.080]}]}
    (output/'scene_static.json').write_text(json.dumps(scene,indent=2)+'\n')
    (output/'trial_manifest.json').write_text(json.dumps({
        'trajectory': 'source_trajectory.csv', 'scene':'scene_static.json',
        'duration_s':times[-1]-times[0],
        'status':'PREPARATION_ONLY_NO_MOVING_BOUNDARY_REPLAY',
        'note':'Static scene is a geometry/fluid smoke test, not a pouring result.'},indent=2)+'\n')
    print('Prepared',output)
    print('Replay vessel motion with sph_drake_replay.py.')

if __name__ == '__main__':
    main()
