# Simple Pour Trajectory prototype

This package is designed to be copied into the existing `KevinHuang0126/pouring-bot` repository on the branch `wyatt/simple-pour-trajectory`.

## Files

- `simple_pour_trajectory.py` → repository root, beside `visualize_station.py`.
- `tests/test_simple_pour_trajectory.py` → `tests/` in the repository.

The standalone scene loads the existing `models/*.sdf` and `models/package.xml`. No models need to be copied or changed.

## Run

Use the same Python environment as the repository's Drake simulation. From the repository root:

```powershell
python simple_pour_trajectory.py
python -m unittest discover -s tests -p 'test_simple_pour_trajectory.py' -v
```

The script opens a Meshcat server with a table, receiving beaker, and three statically suspended/tilted source vessels. It draws schematic blue fill cores and ballistic jet curves.

## Future robot integration

At a robot update step, obtain the currently held vessel body pose `X_WV` with `plant.EvalBodyPoseInWorld(plant_context, vessel_body)`, then call:

```python
result = compute_pour_trajectory(
    X_WV, VESSELS['test_tube'], fill_fraction,
    linear_velocity_W=body_linear_velocity_W,
    angular_velocity_W=body_angular_velocity_W,
    surface_z=TABLE_TOP,
)
render_pour_trajectory(meshcat, 'pouring/held_vessel/jet', result)
hit, miss_distance_m = target_opening_error(
    result, target_center_xy=(0.72, 0.14),
    target_opening_z=0.29, target_radius=0.04,
)
```

Use the current **body-origin** linear velocity in the world frame and angular velocity in the world frame. For dynamic integration, supply the held body's measured/kinematic velocity. The jet is an instantaneous prediction from that state, not a history of emitted particles. Use the existing station's Meshcat instance for overlay; do not launch a second viewer.

## Assumptions and limitations

This is a heuristic tilt activation plus a single ballistic jet. The `25 + 35 * (1-fill_fraction)` threshold is a placeholder, not a geometrically derived spill condition. Fluid head, discharge speed, and jet launch direction are only approximate. The fill indicators are colored visual markers, not real fluid volume or free surfaces. The actual SDF labware are currently **solid collision cylinders/spheres with no modeled open interior**, so contact with a real poured fluid and vessel interior cannot be validated from these meshes. Water depletion, splashing, impact/contact, wetting, and actual collection in the receiving beaker are not modeled. The standalone visualization does not include the iiwa arm or execute robot motion. Receiver scoring assumes a horizontal circular opening and ignores rim/wall collisions.

For a more accurate validation model later, add actual hollow vessel collision geometry, vessel-specific tilt/spill onset derived from liquid volume and orientation, liquid conservation, and a more realistic fluid/particle simulator.
