"""Generate and optionally test eight axisymmetric vessel families in Drake.

Solid surrogate collision models, not hollow fluid containers.
Run from the pouring-bot repository root with the Drake Python environment.
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'models' / 'generated'

# (normalized height, radius in meters) profiles. The bottom is z=0.
# Each adjacent pair becomes one *convex frustum collision component*.
# This preserves changes in width without convex-hulling the entire shape.

PROFILES = {
    # 1. Straight cylinder
    "cylinder": [
        (0.00, 0.026),
        (1.00, 0.026),
    ],

    # 2. Traditional Erlenmeyer: wide bottom, narrow top
    "conical": [
        (0.00, 0.037),
        (0.78, 0.014),
        (1.00, 0.014),
    ],

    # Rounded body with a longer, narrower cylindrical neck

    "round_bottom": [
        (0.00, 0.018),
        (0.12, 0.027),
        (0.30, 0.032),
        (0.48, 0.032),
        (0.60, 0.026),
        (0.68, 0.023),
        (1.00, 0.023),
    ],

    # 4. Shouldered bottle
    "shouldered": [
        (0.00, 0.033),
        (0.64, 0.033),
        (0.79, 0.015),
        (1.00, 0.015),
    ],

    # 5. Inverse Erlenmeyer:
    # Straight cylindrical bottom, flares outward at top
    'inverted_taper': [
        (0.00, 0.022),
        (0.80, 0.022),
        (0.92, 0.028),
        (1.00, 0.028),
    ],

    # 6. Funnel:
    # Wider cylindrical bottom instead of thin stem
    "funnel": [
        (0.00, 0.021),
        (0.43, 0.021),
        (0.88, 0.037),
        (1.00, 0.037),
    ],

    # Goblet:
    # Long straight lower section with a gentler upper flare
    "goblet": [
        (0.00, 0.022),
        (0.65, 0.022),
        (0.82, 0.028),
        (1.00, 0.028),
    ],

    # 8. Flared rim:
    # Mostly cylindrical with widened upper rim
    "flared_rim": [
        (0.00, 0.025),
        (0.70, 0.025),
        (0.90, 0.036),
        (1.00, 0.036),
    ],
    "waisted_bottle": [
        (0.00, 0.026),
        (0.20, 0.026),
        (0.35, 0.020),
        (0.50, 0.020),
        (0.72, 0.035),
        (1.00, 0.035),
    ],
}



def write_frustum(path, r0, r1, dz, segments=48):
    """Closed convex OBJ spanning local z=[0,dz]."""
    with path.open('w') as f:
        f.write('# Convex frustum, dimensions in meters\n')
        for z, r in ((0., r0), (dz, r1)):
            for i in range(segments):
                a = 2 * math.pi * i / segments
                f.write(f'v {r * math.cos(a):.9f} {r * math.sin(a):.9f} {z:.9f}\n')
        f.write('v 0 0 0\n')
        f.write(f'v 0 0 {dz:.9f}\n')
        c0, c1 = segments * 2 + 1, segments * 2 + 2
        for i in range(segments):
            j = (i+1) % segments
            b1, b2 = i+1, j+1
            t1, t2 = segments+i+1, segments+j+1
            f.write(f'f {b1} {b2} {t2}\n')
            f.write(f'f {b1} {t2} {t1}\n')
            f.write(f'f {c0} {b2} {b1}\n')
            f.write(f'f {c1} {t1} {t2}\n')


def calc_inertia(profile, height, mass):
    """Mass properties for the *solid of revolution*, not hollow glass."""
    sums = [0.0, 0.0, 0.0, 0.0]
    for (z0,r0),(z1,r1) in zip(profile, profile[1:]):
        n = 160
        dz = height * (z1-z0) / n
        for j in range(n):
            u = (j+.5)/n
            z = height * (z0 + (z1-z0)*u)
            r = r0 + (r1-r0)*u
            dv = math.pi*r*r*dz
            sums[0] += dv
            sums[1] += dv*z
            sums[2] += dv*z*z
            sums[3] += dv*r*r
    vol,z1,z2,r2 = sums
    com = z1/vol
    rho = mass/vol
    izz = rho*r2/2
    ixx = rho*(r2/4 + z2 - 2*com*z1 + com*com*vol)
    return com,ixx,izz


def generate(family, name, height, mass, radius_scale):
    if family not in PROFILES:
        raise ValueError(f'Unknown family: {family}')
    if not name.replace('_','').isalnum():
        raise ValueError('Name must contain only letters, numbers, underscores')
    if not (0.09 <= height <= 0.25 and mass > 0 and 0.6 <= radius_scale <= 1.25):
        raise ValueError('Require height 0.09-0.25 m, mass > 0, radius scale 0.6-1.25')
    profile = [(z, r*radius_scale) for z,r in PROFILES[family]]
    if max(r for _,r in profile)*2 > .095:
        raise ValueError('Vessel too wide for this initial generator')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    parts=[]
    for i,((z0,r0),(z1,r1)) in enumerate(zip(profile, profile[1:])):
        part_name=f'{name}_part_{i:02d}.obj'
        write_frustum(OUTPUT_DIR/part_name,r0,r1,height*(z1-z0))
        uri=f'package://pouring_bot/generated/{part_name}'
        z_offset=height*z0
        parts.append((i,z_offset,uri))
    com,ixx,izz=calc_inertia(profile,height,mass)
    sdf = [f'''<?xml version="1.0"?>
<sdf version="1.7" xmlns:drake="http://drake.mit.edu">
  <model name="{name}">
    <link name="body">
      <inertial>
        <pose>0 0 {com:.9f} 0 0 0</pose>
        <mass>{mass:.9f}</mass>
        <inertia>
          <ixx>{ixx:.12f}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{ixx:.12f}</iyy><iyz>0</iyz><izz>{izz:.12f}</izz>
        </inertia>
      </inertial>''']
    for i,z_offset,uri in parts:
        sdf.append(f'''
      <visual name="visual_{i}">
        <pose>0 0 {z_offset:.9f} 0 0 0</pose>
        <geometry><mesh><uri>{uri}</uri></mesh></geometry>
        <material><diffuse>0.65 0.90 0.98 0.55</diffuse></material>
      </visual>
      <collision name="collision_{i}">
        <pose>0 0 {z_offset:.9f} 0 0 0</pose>
        <geometry><mesh><uri>{uri}</uri><drake:declare_convex/></mesh></geometry>
        <drake:proximity_properties>
          <drake:mu_static>1.0</drake:mu_static>
          <drake:mu_dynamic>1.0</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>''')
    sdf.append('    </link>\n  </model>\n</sdf>\n')
    path = OUTPUT_DIR/f'{name}.sdf'
    path.write_text(''.join(sdf))
    ET.parse(path)
    print(f'Generated {family}: {path}')
    print(f'  height={height:.3f} m, base radius={profile[0][1]:.3f} m, '
          f'top radius={profile[-1][1]:.3f} m, max radius={max(r for _,r in profile):.3f} m')
    print(f'  mass={mass:.3f} kg, COM z={com:.3f} m; {len(parts)} collision segments')
    return profile


def run_station(name,height,mouth_radius,grasp_height,x,y):
    import builtins
    import simple_pour_trajectory as simple_pour
    import pour_planner
    import visualize_station as station
    simple_pour.VESSELS[name] = simple_pour.VesselSpec(
        name=name,height=height,mouth_radius=mouth_radius,
        fill_display_radius=0.7*mouth_radius)
    station.VESSELS = tuple(dict.fromkeys((*station.VESSELS,name)))
    station.GRASP_HEIGHTS[name] = grasp_height
    station.GRIP_FORCE_LIMITS[name] = 60.0
    station.RANDOMIZE_VESSEL = False
    station.RANDOMIZE_POSITION = False
    station.FIXED_VESSEL = name
    station.FIXED_SOURCE_X = x
    station.FIXED_SOURCE_Y = y
    pour_planner.VESSEL_TILT_ANGLES[name] = 95.0
    pour_planner.VESSEL_TILT_DURATIONS[name] = 10.0
    original = station.scenario_text
    def redirected(vessel_name,source_x,source_y):
        scenario=original(vessel_name,source_x,source_y)
        return scenario.replace(
            f'package://pouring_bot/{name}.sdf',
            f'package://pouring_bot/generated/{name}.sdf')
    station.scenario_text=redirected
    station.main()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=list(PROFILES)+['all'],default='inverted_taper')
    p.add_argument('--name',default=None)
    p.add_argument('--height',type=float,default=.15)
    p.add_argument('--mass',type=float,default=.12)
    p.add_argument('--radius-scale',type=float,default=1.)
    p.add_argument('--grasp-height',type=float,default=None)
    p.add_argument('--source-x',type=float,default=.508)
    p.add_argument('--source-y',type=float,default=-.047)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.family=='all':
        if a.run:
            p.error('--run requires a single family')
        for family in PROFILES:
            generate(family,f'shape_{family}_001',a.height,a.mass,a.radius_scale)
        return
    name=a.name or f'shape_{a.family}_001'
    profile=generate(a.family,name,a.height,a.mass,a.radius_scale)
    grasp_h=a.grasp_height if a.grasp_height is not None else a.height*.55
    if not 0 < grasp_h < a.height:
        p.error('--grasp-height must be within vessel height')
    if a.run:
        run_station(name,a.height,profile[-1][1],grasp_h,a.source_x,a.source_y)


if __name__=='__main__':
    main()
