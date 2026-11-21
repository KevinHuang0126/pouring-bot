"""Generate parametric vessel SDF models for Drake."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

OUTPUT_DIR = Path("models/generated")


def generate_vessel(
    name,
    body_radius,
    body_height,
    neck_radius,
    neck_height,
):
    if not all(v > 0 for v in (
        body_radius, body_height,
        neck_radius, neck_height,
    )):
        raise ValueError("All dimensions must be positive")

    if neck_radius > body_radius:
        raise ValueError(
            "Neck radius must not exceed body radius"
        )

    total_height = body_height + neck_height

    # Approximate mass, scaled with external volume.
    volume = (
        3.14159 * body_radius**2 * body_height
        + 3.14159 * neck_radius**2 * neck_height
    )
    mass = max(0.03, 0.15 * volume / 0.000466)

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <link name="body">
      <inertial>
        <pose>0 0 {total_height/2:.6f} 0 0 0</pose>
        <mass>{mass:.6f}</mass>
        <inertia>
          <ixx>0.0004</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.0004</iyy>
          <iyz>0</iyz>
          <izz>0.0002</izz>
        </inertia>
      </inertial>

      <visual name="base">
        <pose>0 0 {body_height/2:.6f} 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>{body_radius:.6f}</radius>
            <length>{body_height:.6f}</length>
          </cylinder>
        </geometry>
      </visual>

      <collision name="base">
        <pose>0 0 {body_height/2:.6f} 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>{body_radius:.6f}</radius>
            <length>{body_height:.6f}</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="neck">
        <pose>0 0 {body_height + neck_height/2:.6f} 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>{neck_radius:.6f}</radius>
            <length>{neck_height:.6f}</length>
          </cylinder>
        </geometry>
      </visual>

      <collision name="neck">
        <pose>0 0 {body_height + neck_height/2:.6f} 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>{neck_radius:.6f}</radius>
            <length>{neck_height:.6f}</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.sdf"
    path.write_text(sdf)
    ET.parse(path)

    print(f"Generated: {path}")
    print(f"Body radius: {body_radius:.3f} m")
    print(f"Body height: {body_height:.3f} m")
    print(f"Neck radius: {neck_radius:.3f} m")
    print(f"Neck height: {neck_height:.3f} m")
    print(f"Total height: {total_height:.3f} m")
    print(f"Mass: {mass:.3f} kg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="vessel_001")
    parser.add_argument("--body-radius", type=float, default=0.040)
    parser.add_argument("--body-height", type=float, default=0.080)
    parser.add_argument("--neck-radius", type=float, default=0.016)
    parser.add_argument("--neck-height", type=float, default=0.080)

    args = parser.parse_args()

    generate_vessel(
        args.name,
        args.body_radius,
        args.body_height,
        args.neck_radius,
        args.neck_height,
    )
