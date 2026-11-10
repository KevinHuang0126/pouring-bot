
import numpy as np

from pydrake.all import (
    InverseKinematics,
    RollPitchYaw,
    RotationMatrix,
    Solve,
)


def solve_pregrasp_ik(plant, plant_context, target_position):
    """Solve IK for a downward-facing gripper above the vessel."""

    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg = plant.GetModelInstanceByName("wsg")

    gripper_frame = plant.GetFrameByName("body", wsg)
    world_frame = plant.world_frame()

    q_initial = plant.GetPositions(plant_context).copy()

    ik = InverseKinematics(plant)
    q = ik.q()
    prog = ik.prog()


    # Approximate midpoint between fingertips.
    # The fingers extend along the gripper's local -Z axis.
    # Midpoint between the two finger-body origins.
    p_grasp = np.array([0.0, 0.0771, 0.0])
    # Position the grasp center at the desired world location.
    tolerance = 0.005

    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=p_grasp,
        frameA=world_frame,
        p_AQ_lower=target_position - tolerance,
        p_AQ_upper=target_position + tolerance,
    )

    # Initial candidate: orient the gripper's local Z axis downward.
    # Rotate the gripper so its local -X axis points downward.
    # Local -Z points vertically downward.
    R_desired = RotationMatrix()

    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=R_desired,
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=np.deg2rad(5),
    )

    # Favor a configuration near the initial joint positions.
    prog.AddQuadraticErrorCost(
        np.eye(plant.num_positions()),
        q_initial,
        q,
    )

    prog.SetInitialGuess(q, q_initial)

    result = Solve(prog)

    if not result.is_success():
        print("Orientation-aware IK failed.")
        return None

    plant.SetPositions(plant_context, result.GetSolution(q))
    q_iiwa = plant.GetPositions(plant_context, iiwa).copy()

    print("Orientation-aware IK succeeded!")
    print("Joint angles:", np.round(q_iiwa, 3))

    return q_iiwa
