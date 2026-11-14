"""Jacobian-based pouring with reduced-order liquid volume tracking.

Keeps the camera-localization branch's motion parameters and slip checks.
One FluidState is shared across clearance, transport, tilt, hold and recovery.
"""
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, JacobianWrtVariable, AngleAxis
from simple_pour_trajectory import (VESSELS, GRAVITY, compute_pour_trajectory,
    render_pour_trajectory, target_opening_error)
from fluid_model import create_fluid_state, compute_flow_rate_ml_s, print_fluid_summary

BEAKER_CENTER_XY = np.array([.72, .14])
BEAKER_OPENING_Z = .29
BEAKER_RADIUS = .04
POUR_MOUTH_Z = .46
VESSEL_CLEARANCE = .04
FILL_FRACTION = .75
MAX_TILT_DEG = 110.
VESSEL_TILT_ANGLES = {'test_tube': 110., 'erlenmeyer': 110., 'boiling_flask': 110.}
VESSEL_TILT_DURATIONS = {'test_tube': 12., 'erlenmeyer': 15., 'boiling_flask': 15.}
CLEARANCE_DURATION = 3.
TRANSPORT_DURATION = 6.
TILT_DURATION = 8.
POUR_HOLD_DURATION = 3.
RETURN_DURATION = 8.
CONTROL_DT = .02
SETTLE_DURATION = 4.
POSITION_GAIN = 5.
ORIENTATION_GAIN = 4.
DAMPING = .035
JOINT_SPEED_LIMIT = 1.
LINEAR_SPEED_LIMIT = .15
ANGULAR_SPEED_LIMIT = .65
POSITION_TOLERANCE = .025
ORIENTATION_TOLERANCE = np.deg2rad(8.)
GRASP_POSITION_TOLERANCE = .025
GRASP_ORIENTATION_TOLERANCE = np.deg2rad(20.)
# Set False for a controlled comparison using the original aiming trajectory.
DYNAMIC_AIMING = True
# Smooth the aiming correction at release onset to avoid a rim-offset jump.
AIM_BLEND_DEG = 3.
STREAM_PATH = 'pouring/live_stream'


def smoothstep(s):
    s = np.clip(s, 0., 1.)
    return 3*s**2 - 2*s**3


def limit_norm(vector, max_norm):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector * max_norm / norm if norm > max_norm else vector


def rotation_error_world(R_desired, R_actual):
    aa = RotationMatrix(R_desired.matrix() @ R_actual.matrix().T).ToAngleAxis()
    return aa.angle() * aa.axis()


def get_arm_indices(plant, iiwa):
    positions, velocities = [], []
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_joint(joint_index)
        positions.extend(range(joint.position_start(), joint.position_start() + joint.num_positions()))
        velocities.extend(range(joint.velocity_start(), joint.velocity_start() + joint.num_velocities()))
    return np.array(positions, dtype=int), np.array(velocities, dtype=int)


def get_gripper_pose(plant, plant_context):
    wsg = plant.GetModelInstanceByName('wsg')
    return plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName('body', wsg))


def get_vessel_pose(plant, plant_context, vessel_body):
    return plant.EvalBodyPoseInWorld(plant_context, vessel_body)


def vessel_pose_from_mouth(mouth_position, rotation, vessel_spec):
    mouth_V = np.array([0., 0., vessel_spec.height])
    return RigidTransform(rotation, np.asarray(mouth_position) - rotation.matrix() @ mouth_V)


def calculate_mouth_height(vessel_spec, mouth_start):
    return max(POUR_MOUTH_Z, float(mouth_start[2]), BEAKER_OPENING_Z + vessel_spec.height + VESSEL_CLEARANCE)


def compute_aimed_vessel_pose(rotation, vessel_spec, mouth_z, fill_fraction):
    """Aim a stationary predicted jet at the receiver; actual flow uses actual pose."""
    mouth_position = np.array([*BEAKER_CENTER_XY, mouth_z])
    pose = vessel_pose_from_mouth(mouth_position, rotation, vessel_spec)
    result = compute_pour_trajectory(pose, vessel_spec, fill_fraction)
    if result.active:
        p, v = result.mouth_W, result.launch_velocity_W
        dz = p[2] - BEAKER_OPENING_Z
        if dz > 0:
            t_hit = (v[2] + np.sqrt(v[2]**2 + 2*GRAVITY*dz)) / GRAVITY
            correction = BEAKER_CENTER_XY - (p[:2] + v[:2]*t_hit)
            if DYNAMIC_AIMING:
                correction *= smoothstep((result.tilt_deg - result.onset_deg) / AIM_BLEND_DEG)
            mouth_position[:2] += correction
            pose = vessel_pose_from_mouth(mouth_position, rotation, vessel_spec)
    return pose


def update_liquid_stream(meshcat, plant, plant_context, vessel_body, vessel_name, fill_fraction):
    pose = get_vessel_pose(plant, plant_context, vessel_body)
    velocity = plant.EvalBodySpatialVelocityInWorld(plant_context, vessel_body)
    result = compute_pour_trajectory(pose, VESSELS[vessel_name], fill_fraction,
        linear_velocity_W=velocity.translational(), angular_velocity_W=velocity.rotational())
    render_pour_trajectory(meshcat, STREAM_PATH, result)
    return result


def advance_fluid(meshcat, plant, plant_context, vessel_body, vessel_name, fluid, dt):
    """Integrate once per robot timestep, using the actual pose at its end."""
    result = update_liquid_stream(meshcat, plant, plant_context, vessel_body, vessel_name,
                                  0. if fluid.is_empty else fluid.fill_fraction)
    hit, _ = target_opening_error(result, BEAKER_CENTER_XY, BEAKER_OPENING_Z, BEAKER_RADIUS)
    rate = compute_flow_rate_ml_s(vessel_name, VESSELS[vessel_name].mouth_radius,
                                  result.exit_speed_m_s, result.active)
    fluid.advance(dt, rate, hit)
    if fluid.is_empty:
        meshcat.Delete(STREAM_PATH)
    return result


def check_grasp(plant, plant_context, vessel_body, X_GV_reference):
    X_GV_actual = get_gripper_pose(plant, plant_context).inverse() @ get_vessel_pose(plant, plant_context, vessel_body)
    position_error = np.linalg.norm(X_GV_actual.translation() - X_GV_reference.translation())
    orientation_error = np.linalg.norm(rotation_error_world(X_GV_reference.rotation(), X_GV_actual.rotation()))
    return (position_error < GRASP_POSITION_TOLERANCE and orientation_error < GRASP_ORIENTATION_TOLERANCE,
            position_error, orientation_error)


def command_cartesian_motion(simulator, station, station_context, plant, plant_context,
        meshcat, vessel_body, vessel_name, X_GV, pose_function, duration, phase_name,
        fluid, settle_duration=SETTLE_DURATION, stop_when_empty=False):
    """Track a desired vessel pose with the existing damped differential IK."""
    iiwa = plant.GetModelInstanceByName('iiwa')
    wsg = plant.GetModelInstanceByName('wsg')
    gripper_frame = plant.GetFrameByName('body', wsg)
    world = plant.world_frame()
    position_indices, velocity_indices = get_arm_indices(plant, iiwa)
    joint_lower = plant.GetPositionLowerLimits()[position_indices]
    joint_upper = plant.GetPositionUpperLimits()[position_indices]
    position_port = station.GetInputPort('iiwa.position')
    # Preserve the grip command set by visualize_station.py. Do not reset it.
    steps = max(1, int(np.ceil(duration / CONTROL_DT)))
    dt = duration / steps
    settle_steps = int(np.ceil(settle_duration / dt))
    start_time = simulator.get_context().get_time()
    X_VG = X_GV.inverse()

    def desired_gripper_pose(s):
        return pose_function(s) @ X_VG

    print(f'\n{phase_name}')
    X_WG_previous = desired_gripper_pose(0.)
    last_reported = -1
    for i in range(1, steps + settle_steps + 1):
        s = min(i / steps, 1.)
        X_WG_desired = desired_gripper_pose(s)
        X_WG_actual = get_gripper_pose(plant, plant_context)
        position_error = X_WG_desired.translation() - X_WG_actual.translation()
        linear_feedforward = (X_WG_desired.translation() - X_WG_previous.translation()) / dt
        v_desired = limit_norm(linear_feedforward + POSITION_GAIN * position_error, LINEAR_SPEED_LIMIT)
        orientation_error = rotation_error_world(X_WG_desired.rotation(), X_WG_actual.rotation())
        angular_feedforward = rotation_error_world(X_WG_desired.rotation(), X_WG_previous.rotation()) / dt
        omega_desired = limit_norm(angular_feedforward + ORIENTATION_GAIN * orientation_error, ANGULAR_SPEED_LIMIT)
        V_desired = np.concatenate([omega_desired, v_desired])
        J_full = plant.CalcJacobianSpatialVelocity(plant_context, JacobianWrtVariable.kV,
            gripper_frame, [0., 0., 0.], world, world)
        J = J_full[:, velocity_indices]
        qdot = J.T @ np.linalg.solve(J @ J.T + DAMPING**2 * np.eye(6), V_desired)
        max_speed = np.max(np.abs(qdot))
        if max_speed > JOINT_SPEED_LIMIT:
            qdot *= JOINT_SPEED_LIMIT / max_speed
        q_actual = plant.GetPositions(plant_context, iiwa).copy()
        q_command = np.clip(q_actual + dt*qdot, joint_lower + 1e-4, joint_upper - 1e-4)
        position_port.FixValue(station_context, q_command)
        simulator.AdvanceTo(start_time + i*dt)

        # Advance even on the last interval before a slip abort.
        result = advance_fluid(meshcat, plant, plant_context, vessel_body, vessel_name, fluid, dt)
        grasp_ok, slip_position, slip_orientation = check_grasp(plant, plant_context, vessel_body, X_GV)
        if not grasp_ok:
            print('\nWARNING: Vessel slipped relative to gripper.')
            print('Vessel:', vessel_name)
            print('Grasp position error:', round(slip_position, 4), 'm')
            print('Grasp orientation error:', round(np.rad2deg(slip_orientation), 2), 'deg')
            position_port.FixValue(station_context, plant.GetPositions(plant_context, iiwa).copy())
            meshcat.Delete(STREAM_PATH)
            return False
        progress = int(s*4)
        if progress > last_reported or i == steps + settle_steps:
            last_reported = progress
            print(f'  Progress: {100*s:.0f}% | Tilt: {result.tilt_deg:.1f} deg'
                  f' | Flow: {fluid.flow_rate_ml_s:.2f} mL/s'
                  f' | Remaining: {fluid.remaining_volume_ml:.2f} mL'
                  f' | Collected: {fluid.collected_volume_ml:.2f} mL'
                  f' | Spilled: {fluid.spilled_volume_ml:.2f} mL')
        if stop_when_empty and fluid.is_empty:
            print('Source empty: ending pouring phase early.')
            print(f'  Remaining: {fluid.remaining_volume_ml:.2f} mL | Collected: {fluid.collected_volume_ml:.2f} mL')
            # Stop at the measured arm configuration; recovery starts here.
            position_port.FixValue(station_context, plant.GetPositions(plant_context, iiwa).copy())
            return True
        X_WG_previous = X_WG_desired

    X_WG_final = get_gripper_pose(plant, plant_context)
    X_WG_goal = desired_gripper_pose(1.)
    final_position_error = np.linalg.norm(X_WG_goal.translation() - X_WG_final.translation())
    final_orientation_error = np.linalg.norm(rotation_error_world(X_WG_goal.rotation(), X_WG_final.rotation()))
    print('Final position error:', round(final_position_error, 4), 'm')
    print('Final orientation error:', round(np.rad2deg(final_orientation_error), 2), 'deg')
    if not (final_position_error < POSITION_TOLERANCE and final_orientation_error < ORIENTATION_TOLERANCE):
        print('WARNING: Desired Cartesian pose was not reached.')
        q_actual = plant.GetPositions(plant_context, iiwa).copy()
        print('Current arm configuration:', np.round(q_actual, 3))
        print('Lower joint limit margins:', np.round(q_actual - joint_lower, 3))
        print('Upper joint limit margins:', np.round(joint_upper - q_actual, 3))
        position_port.FixValue(station_context, q_actual)
        meshcat.Delete(STREAM_PATH)
        return False
    print('Motion completed successfully.')
    return True


def execute_pour(simulator, station, station_context, plant, plant_context, meshcat, vessel_body, vessel_name):
    """Run all pouring phases; return motion success as before, not transfer success."""
    vessel_spec = VESSELS[vessel_name]
    fluid = create_fluid_state(vessel_name, FILL_FRACTION)
    max_tilt_deg = VESSEL_TILT_ANGLES.get(vessel_name, MAX_TILT_DEG)
    tilt_duration = VESSEL_TILT_DURATIONS.get(vessel_name, TILT_DURATION)
    return_duration = max(RETURN_DURATION, tilt_duration)
    X_WG_start = get_gripper_pose(plant, plant_context)
    X_WV_start = get_vessel_pose(plant, plant_context, vessel_body)
    X_GV = X_WG_start.inverse() @ X_WV_start
    R_start = X_WV_start.rotation()
    mouth_start = X_WV_start.multiply(np.array([0., 0., vessel_spec.height]))
    mouth_z = calculate_mouth_height(vessel_spec, mouth_start)
    print('\n' + '='*55)
    print('JACOBIAN-BASED POURING WITH FLUID TRACKING')
    print('='*55)
    print('Vessel:', vessel_name)
    print(f'Vessel height: {vessel_spec.height:.3f} m')
    print(f'Mouth radius: {vessel_spec.mouth_radius:.3f} m')
    print('Initial mouth position:', np.round(mouth_start, 4))
    print(f'Desired mouth height: {mouth_z:.3f} m')
    print(f'Commanded tilt rotation: {max_tilt_deg:.1f} degrees')
    print(f'Tilt duration: {tilt_duration:.1f} s')
    print(f'Initial fill fraction: {FILL_FRACTION:.0%}')
    print(f'Initial liquid volume: {fluid.initial_volume_ml:.2f} mL')
    print(f'Vessel capacity: {fluid.capacity_ml:.2f} mL')

    def run_motion(pose_function, duration, phase_name, settle_duration=SETTLE_DURATION, stop_when_empty=False):
        ok = command_cartesian_motion(simulator, station, station_context, plant, plant_context,
            meshcat, vessel_body, vessel_name, X_GV, pose_function, duration, phase_name,
            fluid, settle_duration=settle_duration, stop_when_empty=stop_when_empty)
        if not ok:
            print('\nPartial fluid results (motion aborted):')
            print_fluid_summary(fluid)
            print('Pouring motion completed: False')
            print('Fluid transfer target reached:', transfer_target_reached())
        return ok

    def transfer_target_reached():
        return (fluid.collection_fraction >= .90 and fluid.spilled_volume_ml <= .05*fluid.initial_volume_ml)

    mouth_raised = mouth_start.copy()
    mouth_raised[2] = mouth_z
    def clearance_pose(s):
        a = smoothstep(s)
        return vessel_pose_from_mouth((1-a)*mouth_start + a*mouth_raised, R_start, vessel_spec)
    if not run_motion(clearance_pose, CLEARANCE_DURATION, 'PHASE 6: Raising vessel for clearance...'):
        return False

    mouth_goal = np.array([*BEAKER_CENTER_XY, mouth_z])
    def transport_pose(s):
        a = smoothstep(s)
        return vessel_pose_from_mouth((1-a)*mouth_raised + a*mouth_goal, R_start, vessel_spec)
    if not run_motion(transport_pose, TRANSPORT_DURATION, 'PHASE 7: Transporting to beaker...'):
        return False

    def tilt_pose(s):
        theta = np.deg2rad(max_tilt_deg * smoothstep(s))
        rotation = RotationMatrix.MakeYRotation(theta) @ R_start
        # Aiming-only floor; actual release uses true fill, including zero.
        aiming_fill = max(fluid.fill_fraction, .02) if DYNAMIC_AIMING else FILL_FRACTION
        return compute_aimed_vessel_pose(rotation, vessel_spec, mouth_z, aiming_fill)
    if not run_motion(tilt_pose, tilt_duration, 'PHASE 8: Tilting vessel...', stop_when_empty=True):
        return False

    if not fluid.is_empty:
        fixed_pour_pose = tilt_pose(1.)
        def hold_pose(s):
            return tilt_pose(1.) if DYNAMIC_AIMING else fixed_pour_pose
        if not run_motion(hold_pose, POUR_HOLD_DURATION, 'PHASE 9: Holding pouring pose...',
                          settle_duration=0., stop_when_empty=True):
            return False
    else:
        print('PHASE 9: Skipped - source already empty.')
    print(f'Remaining: {fluid.remaining_volume_ml:.2f} mL | Collected: {fluid.collected_volume_ml:.2f} mL | Spilled: {fluid.spilled_volume_ml:.2f} mL')

    if fluid.is_empty:
        # Recover from the actual arm pose, rather than jumping to the
        # original maximum-tilt waypoint which may never have been reached.
        recovery_start = get_gripper_pose(plant, plant_context) @ X_GV
        R_recovery = recovery_start.rotation()
        mouth_recovery = recovery_start.multiply(np.array([0., 0., vessel_spec.height]))
        relative_rotation = RotationMatrix(R_start.matrix() @ R_recovery.matrix().T).ToAngleAxis()
        def return_pose(s):
            a = smoothstep(s)
            R = RotationMatrix(AngleAxis(a * relative_rotation.angle(), relative_rotation.axis())) @ R_recovery
            mouth = (1-a)*mouth_recovery + a*mouth_goal
            return vessel_pose_from_mouth(mouth, R, vessel_spec)
    else:
        def return_pose(s):
            return tilt_pose(1.-s)
    if not run_motion(return_pose, return_duration, 'PHASE 10: Returning upright...'):
        return False
    meshcat.Delete(STREAM_PATH)
    print_fluid_summary(fluid)
    print('Pouring motion completed: True')
    print('Fluid transfer success:', transfer_target_reached())
    return True
